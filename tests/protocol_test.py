#!/usr/bin/env python3
"""Protocol-level tests for MiniVectorDB's stdin/stdout interface."""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))


def run(payload, d):
    p = subprocess.run([BIN, d], input=payload, text=True,
                       capture_output=True, timeout=30)
    if p.returncode != 0:
        raise AssertionError("rc=%s stderr=%s" % (p.returncode, p.stderr))
    return p.stdout


def fresh():
    return tempfile.mkdtemp(prefix="mvdb-proto-")


def test_multiple_statements_one_line():
    d = fresh()
    try:
        out = run("CREATE TABLE t (id BIGINT PRIMARY KEY); INSERT INTO t VALUES (1),(2); "
                  "SELECT id FROM t ORDER BY id;\n", d)
        assert "Query OK, 0 rows affected" in out
        assert "id\n1\n2" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_comments_and_crlf():
    d = fresh()
    try:
        payload = ("-- leading comment\r\n"
                   "CREATE TABLE t (id BIGINT PRIMARY KEY); # hash comment\r\n"
                   "/* block\ncomment */ INSERT INTO t VALUES (1);\r\n"
                   "SELECT id FROM t;\r\n")
        out = run(payload, d)
        assert "id\n1" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_semicolon_in_string_and_comment():
    d = fresh()
    try:
        out = run("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(20));"
                  "INSERT INTO t VALUES (1,'a;b');"
                  "SELECT s FROM t;\n", d)
        assert "s\na;b" in out, out
        out = run("SELECT 1; -- a ; comment\nSELECT 2;\n", d)
        assert "1\n1\n2\n2" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_trailing_semicolon_and_select_constant():
    d = fresh()
    try:
        out = run("SELECT 1\n", d)
        assert out.strip().splitlines() == ["1", "1"], out
        out = run("SELECT 1 AS x;\n", d)
        assert out.strip().splitlines() == ["x", "1"], out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_database_use_and_empty_statements():
    d = fresh()
    try:
        out = run("CREATE DATABASE IF NOT EXISTS test; USE test; ;; "
                  "DROP DATABASE IF EXISTS test;\n", d)
        assert out.count("Query OK") == 3, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_error_then_continue():
    d = fresh()
    try:
        out = run("CREATE TABLE t (id BIGINT PRIMARY KEY); SELECT nope FROM t; SELECT 1;\n", d)
        assert "ERROR 1054" in out, out
        assert out.rstrip().endswith("1\n1"), out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_bigint_min_and_max():
    d = fresh()
    try:
        out = run("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"
                  "INSERT INTO t VALUES (9223372036854775807, -9223372036854775808);"
                  "SELECT id, v FROM t;\n", d)
        assert "9223372036854775807" in out, out
        assert "-9223372036854775808" in out, out
        out = run("INSERT INTO t VALUES (9223372036854775808, 0);\n", d)
        assert "ERROR 1064" in out, out
        out = run("INSERT INTO t VALUES (-9223372036854775809, 0);\n", d)
        assert "ERROR 1064" in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_exit_and_eof():
    d = fresh()
    try:
        out = run("SELECT 1; exit\nSELECT 2;\n", d)
        assert "1\n1" in out and "2\n2" not in out, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("ALL PROTOCOL TESTS PASSED (%d)" % len(tests))


if __name__ == "__main__":
    main()
