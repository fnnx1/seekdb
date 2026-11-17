#!/usr/bin/env python3
"""Real-threshold JSONL regression for file-backed vector scan buffering.

Default data: 8193 x 4096D float32 vectors (128 MiB + 16 KiB).  Constant
dyadic vectors let the oracle compute distances independently as 64*abs(x-q).
There are many equal distances, reverse insertion order, and scattered updates.

  MINIVECTORDB=/absolute/path/to/binary python3 tests/spill_scan_test.py

This deliberately crosses the production 128 MiB threshold.  It is a correctness
test, not a performance benchmark.  --data-parent selects the filesystem; data
is removed unless --keep-data is supplied.  MVDB_SYNC_COMMIT is inherited.
"""

import argparse
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import tempfile


DIM = 4096
NROWS = 8193
SPILL_BYTES = 128 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("MINIVECTORDB", str(ROOT / "minivectordb"))
TIMEOUT = 180
VECTORS = {}


def vector(numerator):
    """Every coordinate is numerator/8, exactly representable in float32."""
    if numerator not in VECTORS:
        value = format(numerator / 8, ".3f").rstrip("0").rstrip(".")
        if value in ("", "-0"):
            value = "0"
        VECTORS[numerator] = "[" + ",".join([value] * DIM) + "]"
    return VECTORS[numerator]


def row_sql(key, group, numerator):
    return "(%d,%d,'%s')" % (key, group, vector(numerator))


class Engine:
    def __init__(self, directory, kill_on_ready=False):
        env = dict(os.environ)
        env.pop("MVDB_LEGACY_STDIO", None)
        self.stderr = tempfile.TemporaryFile()
        self.closed = False
        self.ident = 0
        self.proc = subprocess.Popen(
            [BIN, str(directory)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            line = self.receive()
            if kill_on_ready:
                self.kill_immediately()
            assert json.loads(line) == {"type": "ready"}, line
        except BaseException:
            self.kill()
            raise

    def receive(self):
        if not self.selector.select(TIMEOUT):
            raise TimeoutError("engine response timed out")
        line = self.proc.stdout.readline()
        if not line:
            self.stderr.seek(0)
            raise AssertionError("engine exited: " + self.stderr.read().decode(errors="replace"))
        assert line.endswith(b"\n"), ("incomplete response", line[:200])
        return line

    def send(self, sql=None, op="execute", mode="exact"):
        self.ident += 1
        request = dict(id=self.ident, op=op)
        if sql is not None:
            request.update(sql=sql, search_mode=mode)
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        self.proc.stdin.flush()
        return self.ident

    @staticmethod
    def decode(line, ident, ok=True):
        response = json.loads(line)
        assert response.get("id") == ident, response
        assert response.get("ok") is ok, response
        if not ok:
            assert response.get("error", {}).get("code"), response
        return response

    def request(self, sql=None, op="execute", ok=True, mode="exact"):
        ident = self.send(sql, op=op, mode=mode)
        return self.decode(self.receive(), ident, ok)

    def request_then_kill(self, sql):
        ident = self.send(sql)
        line = self.receive()
        # No JSON decode, query, stat, sleep or shutdown between ACK and kill.
        self.kill_immediately()
        return self.decode(line, ident)

    def close_handles(self):
        if self.closed:
            return
        self.closed = True
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.stderr.close()

    def kill_immediately(self):
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=TIMEOUT)
        assert self.proc.returncode == -signal.SIGKILL, self.proc.returncode
        self.close_handles()

    def kill(self):
        if self.closed:
            return
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=TIMEOUT)
        self.close_handles()

    def shutdown(self):
        self.request(op="shutdown")
        assert self.proc.wait(timeout=TIMEOUT) == 0, self.proc.returncode
        self.close_handles()


def assert_affected(response, count):
    assert response.get("affected_rows") == count, response


def check_query(engine, model, numerator, where="", predicate=lambda key, group: True,
                descending=False, offset=0, limit=13, mode="exact"):
    # DIM=4096: sqrt(DIM)*(n/8-q/8) = 8*abs(n-q).
    ordered = sorted(
        ((8 * abs(n - numerator), key, group)
         for key, (group, n) in model.items() if predicate(key, group)),
        key=lambda row: ((-row[0] if descending else row[0]), row[1]))
    want = [[key, group, distance] for distance, key, group in ordered[offset:offset + limit]]
    sql = ("SELECT id,g,L2_DISTANCE(v,'%s') AS distance FROM t %s "
           "ORDER BY distance %s LIMIT %d OFFSET %d" %
           (vector(numerator), where, "DESC" if descending else "ASC", limit, offset))
    response = engine.request(sql, mode=mode)
    assert response.get("columns") == ["id", "g", "distance"], response
    got = response.get("rows")
    assert isinstance(got, list) and len(got) == len(want), (sql[:100], got, want)
    for actual, expected in zip(got, want):
        assert actual[:2] == expected[:2], ("ID/tie/filter mismatch", sql[:100], got, want)
        assert isinstance(actual[2], (int, float)) and math.isclose(
            actual[2], expected[2], rel_tol=1e-5, abs_tol=1e-5), (actual, expected)


def check_suite(engine, model, label):
    for q in (-3, 0, 9):
        check_query(engine, model, q)
    check_query(engine, model, 1, offset=7, limit=21)
    check_query(engine, model, -2, descending=True, offset=3, limit=17)
    check_query(engine, model, 2, "WHERE g = 2", lambda key, group: group == 2)
    check_query(engine, model, -1, "WHERE g >= 2 AND g < 4", lambda key, group: 2 <= group < 4,
                offset=5, limit=19)
    check_query(engine, model, 0, "WHERE g = 999", lambda key, group: group == 999)
    check_query(engine, model, 0, limit=0)
    # The public ANN contract requires exact tiny fixtures.  This large test
    # is intentionally an exact oracle; no approximate-recall assumption here.
    response = engine.request("SELECT id,g FROM t ORDER BY id")
    assert response.get("rows") == [[key, model[key][0]] for key in sorted(model)], label
    # Projection follows the separate materialize/get path, while TopK above
    # exercises both parallel full scans and filtered/single-thread scans.
    key = sorted(model)[len(model) // 2]
    response = engine.request("SELECT id,v FROM t WHERE id = %d" % key)
    assert response.get("rows") == [[key, [model[key][1] / 8] * DIM]], label
    print("PASS", label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parent", type=Path)
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="mvdb-spill-regression-", dir=args.data_parent))
    engine = None
    model = {}
    try:
        engine = Engine(directory)
        engine.request("CREATE TABLE t(id BIGINT PRIMARY KEY,g BIGINT,v VECTOR(%d))" % DIM)
        engine.request("BEGIN")
        batch = []
        for i in range(NROWS):
            key, group, n = NROWS - i, i % 5, (i * 7) % 17 - 8
            model[key] = (group, n)
            batch.append(row_sql(key, group, n))
            if len(batch) == 64 or i + 1 == NROWS:
                assert_affected(engine.request("INSERT INTO t VALUES " + ",".join(batch)), len(batch))
                batch.clear()
        engine.request("COMMIT")
        files = list(directory.glob("vec_*.bin"))
        assert len(files) == 1 and files[0].stat().st_size > SPILL_BYTES, files
        check_suite(engine, model, "initial real-threshold spill")

        # Scattered append-only updates force buffer misses and nonmonotonic
        # vector offsets, mixed with still-contiguous original vector ranges.
        for key, group, n in ((1, 3, 13), (4096, 2, -11), (8193, 4, 0)):
            assert_affected(engine.request("UPDATE t SET g=%d,v='%s' WHERE id=%d" %
                                           (group, vector(n), key)), 1)
            model[key] = (group, n)
        assert_affected(engine.request("DELETE FROM t WHERE id=17 OR id=8000"), 2)
        del model[17], model[8000]
        check_suite(engine, model, "autocommit updates and deletes after spill")

        # First row of the failed INSERT is new; duplicate second row must not
        # leak it.  Earlier successful statements in the transaction remain
        # visible, then rollback restores all of them.
        before = dict(model)
        engine.request("BEGIN")
        assert_affected(engine.request("UPDATE t SET g=2,v='%s' WHERE id=4096" % vector(15)), 1)
        model[4096] = (2, 15)
        assert_affected(engine.request("DELETE FROM t WHERE id=5000"), 1)
        del model[5000]
        assert_affected(engine.request("INSERT INTO t VALUES " + row_sql(9001, 2, 0)), 1)
        model[9001] = (2, 0)
        engine.request("INSERT INTO t VALUES " + row_sql(9002, 1, -2) + "," +
                       row_sql(4096, 1, -2), ok=False)
        check_suite(engine, model, "transaction visibility and failed multirow atomicity")
        engine.request("ROLLBACK")
        model = before
        check_suite(engine, model, "rollback restores spilled vector references")
        engine.request("INSERT INTO t VALUES " + row_sql(9010, 1, 3) + "," +
                       row_sql(4096, 1, 3), ok=False)
        check_suite(engine, model, "failed autocommit has no partial row")

        engine.request("BEGIN")
        assert_affected(engine.request("UPDATE t SET g=2,v='%s' WHERE id=1" % vector(-15)), 1)
        model[1] = (2, -15)
        assert_affected(engine.request("DELETE FROM t WHERE id=4096"), 1)
        del model[4096]
        additions = ((9023, 2, 0), (9022, 3, 0), (9021, 2, 8))
        assert_affected(engine.request("INSERT INTO t VALUES " + ",".join(
            row_sql(*entry) for entry in additions)), len(additions))
        model.update({key: (group, n) for key, group, n in additions})
        check_suite(engine, model, "mixed transaction before durable acknowledgement")
        engine.request_then_kill("COMMIT")

        # Kill the recovery process at ready without an intervening SELECT or
        # shutdown.  The next process must recover its checkpoint again.
        engine = Engine(directory, kill_on_ready=True)
        engine = Engine(directory)
        check_suite(engine, model, "commit ACK kill and interrupted-ready recovery")

        # Losing an uncommitted vector append/update/delete must preserve the
        # committed snapshot and its exact distances through another restart.
        engine.request("BEGIN")
        engine.request("UPDATE t SET v='%s' WHERE id=1" % vector(31))
        engine.request("DELETE FROM t WHERE id=9023")
        engine.request("INSERT INTO t VALUES " + row_sql(9999, 2, 0))
        engine.kill_immediately()
        engine = Engine(directory)
        check_suite(engine, model, "uncommitted spill mutations discarded on second recovery")
        engine.shutdown()
        engine = Engine(directory)
        check_suite(engine, model, "normal shutdown and final repeated recovery")
        engine.shutdown()
        print("PASS spill_integration_test: real 128 MiB spill, exact ties/filters/offsets, "
              "atomicity, rollback, ACK SIGKILL, and repeated recovery", flush=True)
    finally:
        if engine is not None:
            engine.kill()
        if args.keep_data:
            print("Retained test data:", directory, flush=True)
        else:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
