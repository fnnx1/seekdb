#!/usr/bin/env python3
"""Run the official public mysqltest cases against the engine.

The public package (mysqltest/) contains one .test file per case (one SQL per
non-empty line) and a .result JSON array with one assertion per SQL.  A new
empty data directory is used for every case.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))
IMAGE = os.environ.get("MVDB_IMAGE", "minivectordb:check")
USE_DOCKER = os.environ.get("USE_DOCKER") == "1"
CASES = os.path.join(ROOT, "mysqltest")


def load_case(name):
    with open(os.path.join(CASES, "cases", name + ".test"), encoding="utf-8") as f:
        tests = [line.strip() for line in f if line.strip()]
    with open(os.path.join(CASES, "cases", name + ".result"), encoding="utf-8") as f:
        expected = json.load(f)
    if len(tests) != len(expected):
        raise AssertionError("%s: %d tests vs %d assertions" % (name, len(tests), len(expected)))
    return tests, expected


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


def run_case(name):
    tests, expected = load_case(name)
    d = tempfile.mkdtemp(prefix="mvdb-public-")
    try:
        if USE_DOCKER:
            cmd = ["docker", "run", "--rm", "-i", "-v", d + ":/data", IMAGE]
        else:
            cmd = [BIN, d]
        p = subprocess.run(cmd, input="\n".join(tests) + "\n", text=True,
                           capture_output=True, timeout=120)
        if p.returncode != 0:
            raise AssertionError("%s: rc=%s stderr=%s" % (name, p.returncode, p.stderr[-500:]))
        lines = p.stdout.splitlines()
        pos = 0
        for idx, (sql, exp) in enumerate(zip(tests, expected)):
            if "columns" in exp:
                if pos >= len(lines):
                    raise AssertionError("%s stmt %d: missing header\n%s" % (name, idx + 1, p.stdout))
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
        print("PASS public/%s (%d statements)" % (name, len(tests)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    names = []
    with open(os.path.join(CASES, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    for c in manifest.get("cases", []):
        names.append(c["id"])
    for n in names:
        run_case(n)
    print("ALL PUBLIC CASES PASSED (%d)" % len(names))


if __name__ == "__main__":
    main()
