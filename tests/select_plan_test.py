#!/usr/bin/env python3
"""Compare reused SELECT templates with fresh parsing, including error paths."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("MINIVECTORDB", str(ROOT / "minivectordb"))


def main():
    env = dict(os.environ)
    env.pop("MVDB_LEGACY_STDIO", None)
    with tempfile.TemporaryDirectory(prefix="mvdb-select-plan-") as directory:
        proc = subprocess.Popen([BIN, directory], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, env=env)
        ident = 0
        count = 0
        try:
            assert json.loads(proc.stdout.readline()) == {"type": "ready"}

            def execute(sql):
                nonlocal ident
                ident += 1
                proc.stdin.write(json.dumps(dict(id=ident, op="execute", sql=sql)) + "\n")
                proc.stdin.flush()
                response = json.loads(proc.stdout.readline())
                response.pop("id", None)
                return response

            def compare(sql):
                nonlocal count
                # A leading comment excludes template reuse, preserving the
                # independent Lexer/Parser path in the same executable.
                cached = execute(sql)
                parsed = execute("/* parse afresh */ " + sql)
                assert cached == parsed, (sql[:200], cached, parsed)
                count += 1
                return cached

            assert execute("CREATE TABLE t(id BIGINT PRIMARY KEY, grp BIGINT, v VECTOR(3))")["ok"]
            assert execute("INSERT INTO t VALUES " + ",".join(
                "(%d,%d,'[%d,%d,%d]')" % (i, i % 25, i % 7, i % 9, i % 11)
                for i in range(100)))["ok"]
            values = ["[1,2,3]", "[3,2,1]", "[0,0,0]", "[1e-40,.00001,-2]",
                      "[0x1p+0,2,3]", "[+1,-2,3]", "[1,2]", "[nan,2,3]",
                      "[1,2,3,]", "[1 x,2,3]", "[1,2,3]more",
                      "[1,2,3\\n]", "[1,2,3'bad]"]
            shapes = [
                "SELECT id FROM t WHERE grp=4 ORDER BY L2_DISTANCE(v,'%s') LIMIT 10",
                "SELECT id,'a''b' AS s FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10",
                "SELECT id,'a\\nb' AS s FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10",
                "SELECT id FROM t /*hello*/ ORDER BY L2_DISTANCE(v,'%s') DESC LIMIT 3,5",
                'SELECT id FROM t ORDER BY L2_DISTANCE(v,"%s") LIMIT 10',
                " SELECT id FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10",
                "SELECT id FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10 /*tail*/",
                "SELECT id FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10; --tail",
            ]
            for shape in shapes:
                for _ in range(3):
                    for value in values:
                        compare(shape % value)

            # Force eviction and then revisit existing templates with new data.
            for rep in range(3):
                for group in range(40):
                    compare("SELECT id FROM t WHERE grp=%d ORDER BY "
                            "L2_DISTANCE(v,'[%d,2,3]') LIMIT 4" % (group, rep))

            shape = "SELECT id FROM t ORDER BY L2_DISTANCE(v,'%s') LIMIT 10"
            compare(shape % "[1,2,3]")
            # Oversized hits fall back to normal parsing and do not poison
            # the retained query vector or change dimension error messages.
            compare(shape % ("[" + ",".join(["1"] * 5000) + "]"))
            compare(shape % ("[" + " " * 131072 + "1,2,3]"))
            compare(shape % "[3,2,1]")
            assert execute("DROP TABLE t")["ok"]
            compare(shape % "[2,2,3]")
            assert execute("CREATE TABLE t(id BIGINT PRIMARY KEY,v VECTOR(4))")["ok"]
            assert execute("INSERT INTO t VALUES (5,'[1,2,3,4]')")["ok"]
            compare(shape % "[1,2,3]")
            assert compare(shape % "[1,2,3,4]")["rows"] == [[5]]
            assert execute("DROP TABLE t")["ok"]
            assert execute("CREATE TABLE t(id BIGINT PRIMARY KEY,w VECTOR(4))")["ok"]
            compare(shape % "[1,2,3,4]")
            print("PASS %d SELECT template parsing, eviction, errors and schema changes" % count)
        finally:
            if proc.poll() is None:
                proc.stdin.write('{"id":0,"op":"shutdown"}\n')
                proc.stdin.flush()
                assert json.loads(proc.stdout.readline()).get("ok")
                proc.wait(timeout=10)
            proc.stdin.close()
            proc.stdout.close()
            assert proc.returncode == 0


if __name__ == "__main__":
    main()
