#!/usr/bin/env python3
"""Random-SQL smoke test: malformed input must produce errors, not crashes."""
import os
import random
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))
N = 20_000

KW = ["SELECT", "INSERT", "INTO", "VALUES", "FROM", "WHERE", "AND", "OR",
      "NOT", "LIKE", "IN", "BETWEEN", "ORDER", "BY", "LIMIT", "OFFSET",
      "CREATE", "TABLE", "PRIMARY", "KEY", "BIGINT", "VARCHAR", "VECTOR",
      "UPDATE", "SET", "DELETE", "DROP", "INDEX", "SHOW", "TABLES",
      "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "AS", "ASC", "DESC",
      "L2_DISTANCE"]
NAMES = ["t", "u", "x", "id", "name", "v", "idx", "a", "b"]
VALS = ["1", "2", "0", "-1", "1.5", "'a'", "'[1,2]'", "'[0,0]'", "NULL", "("]
PUNCT = [";", ",", "(", ")", "=", "<", ">", "<=", ">=", "<>", "*", "'"]


def make_sql():
    random.seed(12345)
    out = []
    for _ in range(N):
        n = random.randint(1, 14)
        parts = []
        for _ in range(n):
            r = random.random()
            if r < 0.45:
                parts.append(random.choice(KW))
            elif r < 0.75:
                parts.append(random.choice(NAMES))
            elif r < 0.9:
                parts.append(random.choice(VALS))
            else:
                parts.append(random.choice(PUNCT))
        s = " ".join(parts)
        if random.random() < 0.9:
            s += ";"
        out.append(s)
    return "\n".join(out) + "\n"


def main():
    d = tempfile.mkdtemp(prefix="mvdb-fuzz-")
    try:
        p = subprocess.run([BIN, d], input=make_sql(), text=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=120)
        assert p.returncode == 0, "crash rc=%s stderr=%s" % (p.returncode, p.stderr[-800:])
        print("PASS fuzz_smoke: %d random statements, no crash" % N)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
