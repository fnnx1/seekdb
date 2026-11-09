#!/usr/bin/env python3
"""Run the official public mysqltest cases.

The v4 package (mysqltest/) has 20 cases, one .test file per case, a .result
JSON array with one assertion per SQL statement, and driver directives
(--restart, --mode ann) that are not SQL and do not consume an assertion.
Each case starts with a fresh data directory; --restart closes the process
normally and reuses the same directory.
"""
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))
IMAGE = os.environ.get("MVDB_IMAGE", "minivectordb:check")
USE_DOCKER = os.environ.get("USE_DOCKER") == "1"
CASES = os.path.join(ROOT, "mysqltest")

# Error codes that our engine returns for the public semantic error names.
ERROR_CODES = {
    "duplicate_key": 1062,
    "unsupported_sql": 1064,
    "dimension_mismatch": 1366,
}


def parse_case(test_path):
    sqls = []
    segments = [[]]
    with open(test_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("--"):
                if low.startswith("--restart"):
                    if segments[-1]:
                        segments.append([])
                # --mode ann: this engine returns exact results, which satisfy
                # the public ANN expectations; no protocol change is needed.
                continue
            sqls.append(line)
            segments[-1].append(line)
    return sqls, [s for s in segments if s]


def run_segment(sqls, d):
    if USE_DOCKER:
        cmd = ["docker", "run", "--rm", "-i", "-v", d + ":/data", IMAGE]
    else:
        cmd = [BIN, d]
    p = subprocess.run(cmd, input="\n".join(sqls) + "\n", text=True,
                       capture_output=True, timeout=120)
    if p.returncode != 0:
        raise AssertionError("rc=%s stderr=%s" % (p.returncode, p.stderr[-800:]))
    return p.stdout.splitlines()


def convert(text, sample):
    if isinstance(sample, bool):
        return text
    if isinstance(sample, int):
        return int(text)
    if isinstance(sample, float):
        return float(text)
    return text


def compare_value(actual, expected):
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5)
    return actual == expected


def run_case(name, test_path, result_path, statements):
    sqls, segments = parse_case(test_path)
    with open(result_path, encoding="utf-8") as f:
        expected = json.load(f)
    if len(sqls) != len(expected):
        raise AssertionError("%s: %d SQL vs %d assertions" % (name, len(sqls), len(expected)))
    if statements is not None and len(sqls) != statements:
        raise AssertionError("%s: %d SQL vs manifest statements %d" % (name, len(sqls), statements))

    d = tempfile.mkdtemp(prefix="mvdb-public-")
    try:
        lines = []
        for seg in segments:
            lines.extend(run_segment(seg, d))
        pos = 0
        for idx, (sql, exp) in enumerate(zip(sqls, expected)):
            if "columns" in exp:
                if pos >= len(lines):
                    raise AssertionError("%s stmt %d: missing header" % (name, idx + 1))
                header = lines[pos].split("\t")
                pos += 1
                if header != exp["columns"]:
                    raise AssertionError("%s stmt %d columns %r != %r\nSQL: %s" %
                                         (name, idx + 1, header, exp["columns"], sql))
                actual_rows = []
                for _ in exp["rows"]:
                    if pos >= len(lines):
                        raise AssertionError("%s stmt %d: missing row" % (name, idx + 1))
                    fields = lines[pos].split("\t")
                    pos += 1
                    if len(fields) != len(exp["columns"]):
                        raise AssertionError("%s stmt %d: field count %d != %d" %
                                             (name, idx + 1, len(fields), len(exp["columns"])))
                    actual_rows.append(tuple(convert(f, e) for f, e in zip(fields, exp["rows"][0])))
                if exp.get("unordered"):
                    if sorted(actual_rows) != sorted(tuple(r) for r in exp["rows"]):
                        raise AssertionError("%s stmt %d unordered rows %r != %r\nSQL: %s" %
                                             (name, idx + 1, actual_rows, exp["rows"], sql))
                else:
                    for a, e in zip(actual_rows, exp["rows"]):
                        if len(a) != len(e) or any(not compare_value(x, y) for x, y in zip(a, e)):
                            raise AssertionError("%s stmt %d rows %r != %r\nSQL: %s" %
                                                 (name, idx + 1, actual_rows, exp["rows"], sql))
            elif "error" in exp:
                if pos >= len(lines):
                    raise AssertionError("%s stmt %d: missing error\nSQL: %s" % (name, idx + 1, sql))
                line = lines[pos]
                pos += 1
                m = re.match(r"ERROR (\d+) \(", line)
                if not m:
                    raise AssertionError("%s stmt %d: expected error, got %r\nSQL: %s" %
                                         (name, idx + 1, line, sql))
                want = ERROR_CODES.get(exp["error"])
                if want is not None and int(m.group(1)) != want:
                    raise AssertionError("%s stmt %d: error code %s != expected %s\nSQL: %s" %
                                         (name, idx + 1, m.group(1), want, sql))
            else:
                if pos >= len(lines):
                    raise AssertionError("%s stmt %d: missing result\nSQL: %s" % (name, idx + 1, sql))
                line = lines[pos]
                pos += 1
                m = re.match(r"Query OK, (\d+) rows? affected$", line)
                if not m:
                    raise AssertionError("%s stmt %d: not Query OK: %r\nSQL: %s" %
                                         (name, idx + 1, line, sql))
                if "affected_rows" in exp and int(m.group(1)) != exp["affected_rows"]:
                    raise AssertionError("%s stmt %d affected %s != %s\nSQL: %s" %
                                         (name, idx + 1, m.group(1), exp["affected_rows"], sql))
        if pos != len(lines):
            raise AssertionError("%s: %d unexpected output lines: %r" %
                                 (name, len(lines) - pos, lines[pos:]))
        print("PASS public/%s (%d statements, %d segments)" %
              (name, len(sqls), len(segments)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    with open(os.path.join(CASES, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    cases = manifest.get("cases", [])
    for c in cases:
        run_case(c["id"],
                 os.path.join(CASES, c["test"]),
                 os.path.join(CASES, c["result"]),
                 c.get("statements"))
    print("ALL PUBLIC CASES PASSED (%d)" % len(cases))


if __name__ == "__main__":
    main()
