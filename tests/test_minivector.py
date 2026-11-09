#!/usr/bin/env python3
"""Dynamic tests for MiniVectorDB.

Run from the repository root after `make`:
    python3 tests/test_minivector.py
Set MINIVECTORDB to override the binary path.
"""
import os
import shutil
import signal
import subprocess
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))


def run(sql, d, timeout=60):
    p = subprocess.run([BIN, d], input=sql + "\n", text=True,
                       capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise AssertionError("rc=%s stderr=%s" % (p.returncode, p.stderr))
    return p.stdout


def sel(sql, d):
    out = run(sql, d).strip("\n")
    return out.splitlines() if out else []


def fresh():
    return tempfile.mkdtemp(prefix="mvdb-test-")


def crash(sql, d, wait=0.35):
    p = subprocess.Popen([BIN, d], stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         text=True)
    p.stdin.write(sql + "\n")
    p.stdin.flush()
    time.sleep(wait)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()


def sigterm(sql, d, wait=0.35):
    p = subprocess.Popen([BIN, d], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    p.stdin.write(sql + "\n")
    p.stdin.flush()
    time.sleep(wait)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=10)


def test_basic_vector_ties():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(20), v VECTOR(3));"
            "INSERT INTO t VALUES (1,'a','[1,2,3]'),(2,'b','[4,5,6]'),(3,'c','[0,0,0]');", d)
        assert sel("SELECT * FROM t ORDER BY id;", d) == [
            "id\tname\tv", "1\ta\t[1,2,3]", "2\tb\t[4,5,6]", "3\tc\t[0,0,0]"]
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2,3]') LIMIT 2;", d) == [
            "id", "1", "3"]
        assert sel("SELECT id FROM t WHERE id >= 2 ORDER BY id DESC;", d) == ["id", "3", "2"]
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2,3]') LIMIT 1,1;", d) == ["id", "3"]
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2,3]') DESC LIMIT 2;", d) == ["id", "2", "3"]
        assert sel("SELECT id, L2_DISTANCE(v,'[1,2,3]') AS d FROM t ORDER BY d LIMIT 2;", d) == [
            "id\td", "1\t0.000000", "3\t3.741657"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_filtered_ties():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(2), v VECTOR(2));"
            "INSERT INTO t VALUES (5,'x','[1,0]'),(2,'x','[1,0]'),(9,'y','[1,0]'),(1,'x','[0,1]');", d)
        assert sel("SELECT id FROM t WHERE g = 'x' ORDER BY L2_DISTANCE(v,'[1,0]') LIMIT 10;", d) == [
            "id", "2", "5", "1"]
        assert sel("SELECT id FROM t WHERE g = 'x' AND id >= 2 ORDER BY id;", d) == ["id", "2", "5"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tx():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(10));"
            "INSERT INTO t VALUES (1,'a'),(2,'b');", d)
        run("BEGIN; INSERT INTO t VALUES (3,'c'); UPDATE t SET name='aa' WHERE id=1;"
            "DELETE FROM t WHERE id=2; COMMIT;", d)
        assert sel("SELECT id,name FROM t ORDER BY id;", d) == ["id\tname", "1\taa", "3\tc"]
        run("BEGIN; UPDATE t SET name='zz' WHERE id=3; DELETE FROM t WHERE id=1; ROLLBACK;", d)
        assert sel("SELECT id,name FROM t ORDER BY id;", d) == ["id\tname", "1\taa", "3\tc"]
        out = run("BEGIN; INSERT INTO t VALUES (4,'d'),(1,'dup'); COMMIT;", d)
        assert "ERROR 1062" in out, out
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "3"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_vector_update_rollback_restart():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));"
            "INSERT INTO t VALUES (1,'[1,2]'),(2,'[3,4]');", d)
        # transaction update then rollback: id 1 must still be the old vector
        run("BEGIN; UPDATE t SET v='[9,9]' WHERE id=1; ROLLBACK;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2]') LIMIT 1;", d) == ["id", "1"]
        # committed vector update survives restart
        run("UPDATE t SET v='[0,0]' WHERE id=2;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0]') LIMIT 1;", d) == ["id", "2"]
        assert sel("SELECT id,v FROM t WHERE id=2;", d) == ["id\tv", "2\t[0,0]"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_restart():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));"
            "INSERT INTO t VALUES (1,'[1,2]'),(2,'[3,4]');", d)
        run("INSERT INTO t VALUES (3,'[5,6]'); UPDATE t SET v='[9,9]' WHERE id=2;", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2", "3"]
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[9,9]');", d)[1] == "2"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_crash():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY); INSERT INTO t VALUES (1),(2);", d)
        crash("BEGIN; INSERT INTO t VALUES (3);", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2"]
        crash("INSERT INTO t VALUES (4);", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2", "4"]
        crash("BEGIN; INSERT INTO t VALUES (5); COMMIT;", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2", "4", "5"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_vector_crash_and_restart():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));"
            "INSERT INTO t VALUES (1,'[1,2]'),(2,'[3,4]');", d)
        crash("BEGIN; INSERT INTO t VALUES (3,'[5,6]'); COMMIT;", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2", "3"]
        crash("BEGIN; INSERT INTO t VALUES (4,'[7,8]');", d)
        assert sel("SELECT id FROM t ORDER BY id;", d) == ["id", "1", "2", "3"]
        crash("UPDATE t SET v='[9,9]' WHERE id=1;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[9,9]') LIMIT 1;", d) == ["id", "1"]
        crash("BEGIN; UPDATE t SET v='[0,0]' WHERE id=2;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[3,4]') LIMIT 1;", d) == ["id", "2"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_close_rollback():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY); BEGIN; INSERT INTO t VALUES (1);", d)
        assert sel("SELECT id FROM t;", d) == ["id"]
        sigterm("CREATE TABLE u (id BIGINT PRIMARY KEY); BEGIN; INSERT INTO u VALUES (1);", d)
        assert sel("SELECT id FROM u;", d) == ["id"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_errors_schema():
    d = fresh()
    try:
        out = run("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(2), v VECTOR(2));"
                  "INSERT INTO t VALUES (1,'ab','[1,2]');"
                  "INSERT INTO t VALUES (1,'x','[1,2]');"
                  "INSERT INTO t VALUES (2,'abc','[1,2]');"
                  "INSERT INTO t VALUES (3,'x','[1,2,3]');", d)
        assert "ERROR 1062" in out and "ERROR 1406" in out and "ERROR 1366" in out, out
        assert sel("SELECT id FROM t;", d) == ["id", "1"]
        out = run("SELECT nope FROM t; SELECT id FROM nope; SELECT id,;", d)
        assert "ERROR 1054" in out and "ERROR 1146" in out and "ERROR 1064" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_index_maintenance():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));"
            "CREATE VECTOR INDEX IF NOT EXISTS idx ON t(v);"
            "INSERT INTO t VALUES (1,'[1,2]'),(2,'[3,4]');", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2]') LIMIT 1;", d) == ["id", "1"]
        run("UPDATE t SET v='[0,0]' WHERE id=2;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[0,0]') LIMIT 1;", d) == ["id", "2"]
        run("DROP INDEX IF EXISTS idx ON t;", d)
        assert sel("SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2]') LIMIT 1;", d) == ["id", "1"]
        run("CREATE INDEX idx2 ON t(v); DROP INDEX idx2 ON t;", d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_like_filter():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(20));"
            "INSERT INTO t VALUES (1,'alpha'),(2,'beta'),(3,'alps'),(4,'a_b');", d)
        assert sel("SELECT id FROM t WHERE name LIKE 'a%' ORDER BY id;", d) == ["id", "1", "3", "4"]
        assert sel("SELECT id FROM t WHERE name LIKE 'a_p%' ORDER BY id;", d) == ["id", "1", "3"]
        assert sel("SELECT id FROM t WHERE id > 1 AND name LIKE '%a' ORDER BY id;", d) == ["id", "2"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_update_affected_rows():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(10), v VECTOR(2));"
            "INSERT INTO t VALUES (1,'a','[1,2]');", d)
        assert "Query OK, 0 rows affected" in run("UPDATE t SET name='a' WHERE id=1;", d)
        assert "Query OK, 0 rows affected" in run("UPDATE t SET v='[1,2]' WHERE id=1;", d)
        assert "Query OK, 1 row affected" in run("UPDATE t SET name='b' WHERE id=1;", d)
        assert "Query OK, 0 rows affected" in run("UPDATE t SET name='b' WHERE id=1;", d)
        assert "Query OK, 1 row affected" in run("DELETE FROM t WHERE id=1;", d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_show_tables_and_truncate():
    d = fresh()
    try:
        run("CREATE TABLE a (id BIGINT PRIMARY KEY);"
            "CREATE TABLE b (id BIGINT PRIMARY KEY);"
            "INSERT INTO a VALUES (1),(2);", d)
        assert sel("SHOW TABLES;", d) == ["Tables_in_test", "a", "b"]
        assert sel("SHOW TABLES LIKE 'a%';", d) == ["Tables_in_test", "a"]
        run("TRUNCATE TABLE a;", d)
        assert sel("SELECT id FROM a;", d) == ["id"]
        run("INSERT INTO a VALUES (3);", d)
        assert sel("SELECT id FROM a;", d) == ["id", "3"]
        run("DROP TABLE b;", d)
        assert sel("SHOW TABLES;", d) == ["Tables_in_test", "a"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_if_exists_index():
    d = fresh()
    try:
        out = run("DROP TABLE IF EXISTS t; CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));"
                  "CREATE VECTOR INDEX idx ON t(v); INSERT INTO t VALUES (1,'[1,2]');", d)
        assert out.count("Query OK") == 4, out
        out = run("DROP TABLE t; SELECT id FROM t;", d)
        assert "ERROR 1146" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_performance_smoke():
    d = fresh()
    try:
        run("CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(8));", d)
        rows = []
        for i in range(10000):
            vals = ",".join(str(((i * 7 + j * 13) % 1000) / 10.0) for j in range(8))
            rows.append("(%d,'[%s]')" % (i, vals))
        t0 = time.time()
        for b in range(0, len(rows), 100):
            run("INSERT INTO t VALUES " + ",".join(rows[b:b + 100]) + ";", d)
        write_s = time.time() - t0
        q = "SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2,3,4,5,6,7,8]') LIMIT 10;"
        t0 = time.time()
        for _ in range(100):
            assert len(sel(q, d)) == 11
        query_s = time.time() - t0
        assert write_s < 120 and query_s < 120, (write_s, query_s)
        print("  perf smoke: write %.2fs, 100 queries %.2fs" % (write_s, query_s))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t0 = time.time()
        t()
        print("PASS %-32s %.2fs" % (t.__name__, time.time() - t0))
    print("ALL TESTS PASSED (%d)" % len(tests))


if __name__ == "__main__":
    main()
