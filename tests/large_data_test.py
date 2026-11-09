#!/usr/bin/env python3
"""Large-data smoke test: 100k x 128D load, exact TopK, restart, peak RSS."""
import os
import resource
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))
N = 100_000
D = 128


def make_sql(path):
    with open(path, "w") as f:
        f.write("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(%d));\n" % D)
        for b in range(0, N, 1000):
            vals = []
            for i in range(b, b + 1000):
                base = (i * 131) % 1000
                vec = ",".join("%.5f" % (((base + j * 17) % 1000) / 1000.0)
                               for j in range(D))
                vals.append("(%d,'[%s]')" % (i, vec))
            f.write("INSERT INTO t VALUES " + ",".join(vals) + ";\n")
        f.write("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[%s]') LIMIT 10;\n"
                % ",".join("0.5" for _ in range(D)))


def run_file(path, d):
    with open(path) as fin:
        p = subprocess.run([BIN, d], stdin=fin, text=True,
                           capture_output=True, timeout=180)
    if p.returncode != 0:
        raise AssertionError("rc=%s stderr=%s" % (p.returncode, p.stderr[-500:]))
    return p.stdout


def run_sql(payload, d):
    p = subprocess.run([BIN, d], input=payload, text=True,
                       capture_output=True, timeout=180)
    if p.returncode != 0:
        raise AssertionError("rc=%s stderr=%s" % (p.returncode, p.stderr[-500:]))
    return p.stdout


def main():
    d = tempfile.mkdtemp(prefix="mvdb-large-")
    try:
        sql = os.path.join(d, "load.sql")
        make_sql(sql)
        out = run_file(sql, d)
        lines = out.strip().splitlines()[-11:]
        assert len(lines) == 11, lines
        assert lines[0] == "id", lines
        q = ("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[%s]') LIMIT 10;\n"
             % ",".join("0.5" for _ in range(D)))
        out2 = run_sql(q, d)
        assert out2.strip().splitlines() == lines, (lines, out2)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
        assert rss < 400.0, "peak RSS %.1f MB" % rss
        print("PASS large_data_test: %d rows, peak RSS %.1f MB" % (N, rss))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
