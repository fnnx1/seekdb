#!/usr/bin/env python3
"""Check WAL commit boundaries, old-format replay and acknowledged durability.

Temporary databases live on the repository filesystem so Btrfs NOCOW is
exercised when available. Other filesystems must retain the unpadded format.
MINIVECTORDB can select a different engine binary, including sanitizer builds.
"""
from array import array
import json
import os
from pathlib import Path
import selectors
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib


ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("MINIVECTORDB", str(ROOT / "minivectordb"))
HEADER = struct.Struct("<IQIB")
NOCOW = 0x00800000


def file_flags(path, set_flags=None):
    """Return Linux inode flags, or None when this filesystem lacks them."""
    if not sys.platform.startswith("linux"):
        return None
    import fcntl
    flags = array("L", [0 if set_flags is None else set_flags])
    # Linux FS_IOC_{GET,SET}FLAGS encode sizeof(long), including on 32-bit.
    command = ((2 if set_flags is None else 1) << 30 |
               flags.itemsize << 16 | ord("f") << 8 |
               (1 if set_flags is None else 2))
    try:
        with Path(path).open("rb") as source:
            fcntl.ioctl(source, command, flags, True)
        return flags[0]
    except OSError:
        return None


class Engine:
    def __init__(self, directory):
        env = dict(os.environ)
        env.pop("MVDB_LEGACY_STDIO", None)
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([BIN, str(directory)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr,
                                     env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        self.ident = 0
        try:
            assert self.receive() == {"type": "ready"}
        except BaseException:
            self.kill()
            raise

    def receive(self):
        if not self.selector.select(30):
            raise TimeoutError("WAL test engine response timed out")
        line = self.proc.stdout.readline()
        if not line:
            self.stderr.seek(0)
            raise AssertionError("engine exited: " + self.stderr.read().decode(errors="replace"))
        return json.loads(line)

    def request(self, sql=None, op="execute"):
        self.ident += 1
        request = dict(id=self.ident, op=op)
        if sql is not None:
            request.update(sql=sql, search_mode="exact")
        self.proc.stdin.write(json.dumps(request).encode() + b"\n")
        self.proc.stdin.flush()
        response = self.receive()
        assert response.get("ok"), (sql, response)
        return response.get("rows", [])

    def close_handles(self):
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.stderr.close()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.close_handles()

    def shutdown(self):
        self.request(op="shutdown")
        assert self.proc.wait(timeout=30) == 0
        self.close_handles()


def records(raw):
    """Independently validate lengths, sequence numbers and IEEE CRC32."""
    result = []
    position = 0
    previous = 0
    while position < len(raw):
        assert len(raw) - position >= HEADER.size, "incomplete test WAL header"
        length, seq, crc, kind = HEADER.unpack_from(raw, position)
        end = position + HEADER.size + length
        assert end <= len(raw), "incomplete test WAL payload"
        payload = raw[position + HEADER.size:end]
        assert seq > previous, (previous, seq)
        assert zlib.crc32(struct.pack("<QB", seq, kind) + payload) == crc
        result.append((kind, position, end))
        previous, position = seq, end
    return result


def check_alignment(directory):
    wal = Path(directory) / "wal.bin"
    raw = wal.read_bytes()
    parsed = records(raw)
    flags = file_flags(wal)
    nocow = flags is not None and bool(flags & NOCOW)
    if nocow:
        alignment = max(4096, wal.stat().st_blksize, os.sysconf("SC_PAGESIZE"))
        assert all(end % alignment == 0 for kind, _, end in parsed if kind == 2)
        assert any(kind == 4 for kind, _, _ in parsed), "NOCOW WAL lacks padding"
    else:
        assert all(kind != 4 for kind, _, _ in parsed), "fallback WAL was padded"
    return raw, parsed, nocow


def ids(engine):
    return [row[0] for row in engine.request("SELECT id FROM t ORDER BY id")]


def insert(engine, pk):
    engine.request("INSERT INTO t VALUES(%d,%d,'[%d,1,2,3]')" % (pk, pk % 2, pk))


def populate(directory):
    engine = Engine(directory)
    try:
        engine.request("CREATE TABLE t(id BIGINT PRIMARY KEY,grp BIGINT,v VECTOR(4))")
        for pk in range(3):
            engine.request("BEGIN")
            insert(engine, pk)
            engine.request("COMMIT")
        assert ids(engine) == [0, 1, 2]
        engine.request("BEGIN")
        insert(engine, 999)  # Kill with a live, unacknowledged transaction.
        return check_alignment(directory)
    finally:
        engine.kill()


def check_recovery(directory, wanted):
    engine = Engine(directory)
    try:
        assert ids(engine) == wanted
        assert engine.request("SELECT id FROM t ORDER BY "
                              "L2_DISTANCE(v,'[0,1,2,3]') LIMIT 10") == [[pk] for pk in wanted[:10]]
    finally:
        engine.kill()


def copy_database(source, target, raw, nocow=False):
    for path in Path(source).iterdir():
        if path.is_file() and path.name != "wal.bin":
            shutil.copyfile(path, Path(target) / path.name)
    wal = Path(target) / "wal.bin"
    wal.touch()
    if nocow:
        flags = file_flags(wal)
        assert flags is not None and file_flags(wal, flags | NOCOW) is not None
    wal.write_bytes(raw)


def test_replay_and_truncated_commits():
    with tempfile.TemporaryDirectory(prefix="mvdb-wal-source-", dir=ROOT) as source:
        raw, parsed, nocow = populate(source)
        ends = [end for kind, _, end in parsed if kind == 2]
        assert len(ends) == 4  # CREATE TABLE followed by three transactions.
        checks = 0
        # Before/inside the 17-byte commit marker and exactly after it. A
        # truncated final group must never discard earlier confirmed groups.
        for count, end in enumerate(ends[1:], 1):
            for offset in (-18, -17, -16, -1, 0):
                with tempfile.TemporaryDirectory(prefix="mvdb-wal-cut-", dir=ROOT) as target:
                    copy_database(source, target, raw[:end + offset], nocow)
                    check_recovery(target, list(range(count if offset == 0 else count - 1)))
                    checks += 1
        # A historical unpadded WAL uses identical record framing and CRCs.
        # Construct it independently of any old engine executable.
        legacy = b"".join(raw[start:end] for kind, start, end in parsed if kind != 4)
        assert records(legacy) and all(kind != 4 for kind, _, _ in records(legacy))
        with tempfile.TemporaryDirectory(prefix="mvdb-wal-legacy-", dir=ROOT) as target:
            copy_database(source, target, legacy, nocow)
            check_recovery(target, [0, 1, 2])
        print("PASS WAL CRC, %d commit truncations and unpadded legacy replay (%s)" %
              (checks, "NOCOW" if nocow else "filesystem fallback"))


def test_repeated_restarts():
    with tempfile.TemporaryDirectory(prefix="mvdb-wal-restarts-", dir=ROOT) as directory:
        populate(directory)
        for cycle in range(12):
            engine = Engine(directory)
            try:
                assert ids(engine) == list(range(3 + cycle))
                engine.request("BEGIN")
                insert(engine, 999)
                engine.request("ROLLBACK")
                engine.request("BEGIN")
                insert(engine, 3 + cycle)
                engine.request("COMMIT")
                check_alignment(directory)
                if cycle % 2 == 0:
                    engine.request("BEGIN")
                    insert(engine, 998)
                    engine.kill()
                else:
                    engine.shutdown()
            except BaseException:
                if engine.proc.poll() is None:
                    engine.kill()
                raise
        check_recovery(directory, list(range(15)))
        print("PASS 12 SIGKILL/shutdown restarts, confirmed commits, rollback and abandoned transactions")


def test_tmpfs_fallback():
    shared_memory = Path("/dev/shm")
    if not shared_memory.is_dir() or not os.access(shared_memory, os.W_OK):
        print("SKIP /dev/shm filesystem fallback (directory unavailable)")
        return
    with tempfile.TemporaryDirectory(prefix="mvdb-wal-fallback-", dir=shared_memory) as directory:
        raw, parsed, nocow = populate(directory)
        assert not nocow, "/dev/shm unexpectedly enabled NOCOW"
        assert raw and all(kind != 4 for kind, _, _ in parsed)
        check_recovery(directory, [0, 1, 2])
        print("PASS /dev/shm unpadded filesystem fallback and SIGKILL recovery")


if __name__ == "__main__":
    test_replay_and_truncated_commits()
    test_repeated_restarts()
    test_tmpfs_fallback()
