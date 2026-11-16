#!/usr/bin/env python3
"""Check acknowledged-commit recovery after immediate process SIGKILL.

The critical path kills the engine as soon as a complete ACK line arrives,
BEFORE decoding JSON, issuing a query, inspecting the WAL, or normal shutdown.
This tests process crashes with the same running kernel; it does not simulate
power loss, kernel crashes, or loss of the filesystem's dirty page cache.

MINIVECTORDB selects the executable. MVDB_SYNC_COMMIT is inherited unchanged:
  python3 tests/commit_crash_test.py
  MVDB_SYNC_COMMIT=1 python3 tests/commit_crash_test.py

Temporary databases default to the repository filesystem. --compaction adds a
larger optional case that crosses the 64 MiB shutdown-checkpoint threshold.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("MINIVECTORDB", str(ROOT / "minivectordb"))
TIMEOUT = 30


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
            line = self.receive_line()
            if kill_on_ready:
                # A recovered process must survive another crash without any
                # intervening SELECT, shutdown, or filesystem inspection.
                self.kill_immediately()
            assert json.loads(line) == {"type": "ready"}, line
        except BaseException:
            self.kill()
            raise

    def receive_line(self):
        if not self.selector.select(TIMEOUT):
            raise TimeoutError("commit-crash engine response timed out")
        line = self.proc.stdout.readline()
        if not line:
            self.stderr.seek(0)
            raise AssertionError("engine exited: " + self.stderr.read().decode(errors="replace"))
        assert line.endswith(b"\n"), ("incomplete response", line)
        return line

    def send(self, sql=None, op="execute"):
        self.ident += 1
        payload = dict(id=self.ident, op=op)
        if sql is not None:
            payload.update(sql=sql, search_mode="exact")
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        self.proc.stdin.flush()
        return self.ident

    @staticmethod
    def decode(line, ident, ok):
        response = json.loads(line)
        assert response.get("id") == ident, response
        assert response.get("ok") is ok, response
        if not ok:
            assert response.get("error", {}).get("code"), response
        return response

    def request(self, sql=None, op="execute", ok=True):
        ident = self.send(sql, op)
        return self.decode(self.receive_line(), ident, ok)

    def request_then_kill(self, sql, ok=True):
        ident = self.send(sql)
        line = self.receive_line()
        # Keep these adjacent: no JSON decoding, queries, logging, sleeping,
        # WAL reads/stat calls, or cooperative shutdown before the SIGKILL.
        self.kill_immediately()
        return self.decode(line, ident, ok)

    def close_handles(self):
        if self.closed:
            return
        self.closed = True
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.stderr.close()

    def kill_immediately(self):
        # os.kill avoids even Popen.poll() between ACK receipt and SIGKILL.
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
        assert self.proc.wait(timeout=TIMEOUT) == 0
        self.close_handles()


def check_rows(engine, expected):
    response = engine.request("SELECT id,n FROM t ORDER BY id")
    assert response.get("columns") == ["id", "n"], response
    want = [[key, expected[key]] for key in sorted(expected)]
    assert response.get("rows") == want, (response.get("rows"), want)


def tiny_commit(directory):
    """200 two-BIGINT rows fit below a typical 4 KiB stdio WAL buffer.

    Before any optional filesystem-alignment padding, the typed row record
    and commit marker together use 3,651 bytes. A missing fflush must not hide
    the final marker merely because its fwrite succeeded in user space.
    """
    expected = {i: i * 3 for i in range(200)}
    engine = Engine(directory)
    try:
        engine.request("CREATE TABLE t(id BIGINT PRIMARY KEY,n BIGINT)")
        engine.request("BEGIN")
        response = engine.request("INSERT INTO t VALUES " + ",".join(
            "(%d,%d)" % pair for pair in expected.items()))
        assert response.get("affected_rows") == 200, response
        engine.request_then_kill("COMMIT")
    finally:
        engine.kill()

    # Kill the recovered instance immediately on readiness. Its recovery
    # checkpoint and WAL transition must remain valid for the next restart.
    Engine(directory, kill_on_ready=True)
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
    finally:
        engine.kill()
    return expected


def autocommit_cycles(directory, expected, cycles):
    """Every DML ACK is followed immediately by SIGKILL, then verification."""
    for cycle in range(cycles):
        key = 1000 + cycle
        value = -100 - cycle
        operations = [
            ("INSERT INTO t VALUES(%d,%d)" % (key, value), 1,
             lambda: expected.__setitem__(key, value)),
            ("UPDATE t SET n=%d WHERE id=%d" % (value * 2, key), 1,
             lambda: expected.__setitem__(key, value * 2)),
            ("DELETE FROM t WHERE id=%d" % key, 1,
             lambda: expected.__delitem__(key)),
        ]
        for sql, affected, update_expected in operations:
            engine = Engine(directory)
            try:
                check_rows(engine, expected)
                response = engine.request_then_kill(sql)
                assert response.get("affected_rows") == affected, response
                update_expected()
            finally:
                engine.kill()
            # Repeat a crash immediately after recovery, not just after DML.
            if cycle % 3 == 0:
                Engine(directory, kill_on_ready=True)
            engine = Engine(directory)
            try:
                check_rows(engine, expected)
            finally:
                engine.kill()


def transaction_errors(directory, expected):
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
        engine.request("BEGIN")
        engine.request("INSERT INTO t VALUES(900001,77)")
        engine.request("UPDATE t SET n=-999 WHERE id=0")
        engine.request("DELETE FROM t WHERE id=1")
        # The first row of this failing statement must never leak into WAL;
        # all other changes in this transaction are still uncommitted.
        engine.request_then_kill("INSERT INTO t VALUES(900002,1),(0,2)", ok=False)
    finally:
        engine.kill()
    Engine(directory, kill_on_ready=True)
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
        engine.request("BEGIN")
        engine.request("INSERT INTO t VALUES(900003,88)")
        engine.request("INSERT INTO t VALUES(900004,1),(0,2)", ok=False)
        # A failed statement does not prevent committing prior good work.
        engine.request_then_kill("COMMIT")
        expected[900003] = 88
    finally:
        engine.kill()
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
        engine.request("BEGIN")
        engine.request("INSERT INTO t VALUES(900005,99)")
        engine.request("UPDATE t SET n=-888 WHERE id=0")
        engine.request_then_kill("ROLLBACK")
    finally:
        engine.kill()
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
        # Also reject a bad autocommit multirow statement atomically.
        engine.request_then_kill("INSERT INTO t VALUES(900006,1),(0,2)", ok=False)
    finally:
        engine.kill()
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
        # Explicitly check an acknowledged statement inside an open
        # transaction: its ACK must not turn it into a committed record.
        engine.request("BEGIN")
        engine.request_then_kill("INSERT INTO t VALUES(900007,123)")
    finally:
        engine.kill()
    engine = Engine(directory)
    try:
        check_rows(engine, expected)
    finally:
        engine.kill()


def compaction_case(directory):
    """Optional 65 MiB payload exercises normal-close checkpoint compaction."""
    text = "x" * 65536
    count = 65 * 16
    engine = Engine(directory)
    try:
        engine.request("CREATE TABLE big(id BIGINT PRIMARY KEY,payload VARCHAR(65536))")
        for start in range(0, count, 16):
            engine.request("INSERT INTO big VALUES " + ",".join(
                "(%d,'%s')" % (key, text) for key in range(start, start + 16)))
        # These inspections belong only to this normal-shutdown compaction
        # case; the immediate-kill paths above never inspect a live WAL.
        assert (Path(directory) / "wal.bin").stat().st_size > 64 * 1024 * 1024
        engine.shutdown()
        assert (Path(directory) / "wal.bin").stat().st_size == 0
        assert (Path(directory) / "snapshot.bin").stat().st_size > 64 * 1024 * 1024
    finally:
        engine.kill()
    Engine(directory, kill_on_ready=True)
    engine = Engine(directory)
    try:
        response = engine.request("SELECT id FROM big ORDER BY id")
        assert response.get("rows") == [[i] for i in range(count)], response
        response = engine.request(
            "SELECT id,payload FROM big WHERE id=0 OR id=%d ORDER BY id" % (count - 1))
        assert response.get("rows") == [[0, text], [count - 1, text]]
    finally:
        engine.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=8,
                        help="number of repeated INSERT/UPDATE/DELETE crash cycles")
    parser.add_argument("--data-parent", type=Path, default=ROOT,
                        help="existing parent directory for temporary databases")
    parser.add_argument("--compaction", action="store_true",
                        help="also test a WAL exceeding the 64 MiB close/checkpoint threshold")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if not args.data_parent.is_dir():
        parser.error("--data-parent must be an existing directory")
    with tempfile.TemporaryDirectory(prefix="mvdb-commit-crash-", dir=args.data_parent) as directory:
        expected = tiny_commit(directory)
        autocommit_cycles(directory, expected, args.cycles)
        transaction_errors(directory, expected)
    if args.compaction:
        with tempfile.TemporaryDirectory(prefix="mvdb-commit-compact-", dir=args.data_parent) as directory:
            compaction_case(directory)
    mode = os.environ.get("MVDB_SYNC_COMMIT", "<default>")
    print("PASS commit_crash_test: tiny 200-row commit, %d autocommit crash cycles, "
          "failed/uncommitted work, recovery-ready crashes%s; MVDB_SYNC_COMMIT=%s" %
          (args.cycles, ", >64MiB compaction" if args.compaction else "", mode))


if __name__ == "__main__":
    main()
