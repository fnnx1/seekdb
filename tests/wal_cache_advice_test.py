#!/usr/bin/env python3
"""Check bounded WAL cache advice without adding a default commit sync.

A temporary LD_PRELOAD shim records WAL advice/sync calls and can reject cache
advice. Recovery uses the existing harness's immediate ACK-to-SIGKILL path.
MINIVECTORDB selects the executable; no global sync or cache drop is used.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from commit_crash_test import Engine


ROOT = Path(__file__).resolve().parent.parent
PAGE = os.sysconf("SC_PAGE_SIZE")
PAYLOAD = "x" * 65536
SHIM = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static bool wal_fd(int fd) {
  char proc[64], path[4096];
  std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
  const ssize_t n = ::readlink(proc, path, sizeof(path) - 1);
  if (n < 0) return false;
  path[n] = 0;
  const char *base = std::strrchr(path, '/');
  return std::strcmp(base ? base + 1 : path, "wal.bin") == 0;
}
static void record(const char *kind, int fd, off_t offset, off_t len, int advice) {
  const char *path = std::getenv("MVDB_TEST_ADVICE_LOG");
  if (!path || !wal_fd(fd)) return;
  struct stat status{};
  if (::fstat(fd, &status) != 0) std::abort();
  FILE *out = std::fopen(path, "a");
  if (!out) std::abort();
  std::fprintf(out, "%s %lld %lld %lld %d\n", kind, (long long)offset,
               (long long)len, (long long)status.st_size, advice);
  if (std::fclose(out) != 0) std::abort();
}
extern "C" int posix_fadvise(int fd, off_t offset, off_t len, int advice) {
  record("advice", fd, offset, len, advice);
  if (wal_fd(fd) && std::getenv("MVDB_TEST_ADVICE_ERROR")) return EINVAL;
  static auto real = reinterpret_cast<int (*)(int, off_t, off_t, int)>(
      ::dlsym(RTLD_NEXT, "posix_fadvise"));
  return real(fd, offset, len, advice);
}
extern "C" int fdatasync(int fd) {
  record("sync", fd, 0, 0, 0);
  static auto real = reinterpret_cast<int (*)(int)>(::dlsym(RTLD_NEXT, "fdatasync"));
  return real(fd);
}
'''


@contextmanager
def environment(values):
    before = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def records(log, kind):
    if not log.exists():
        return []
    return [tuple(map(int, parts[1:])) for line in log.read_text().splitlines()
            if (parts := line.split())[0] == kind]


def check_calls(log, advice_count, sync_count, partial=False):
    advice = records(log, "advice")
    assert len(advice) == advice_count, (advice_count, advice)
    syncs = records(log, "sync")
    assert len(syncs) == sync_count, (sync_count, syncs)
    for offset, length, size, hint in advice:
        assert offset == 0, ("must revisit the complete prefix", advice)
        assert 0 < length == size // PAGE * PAGE, ("must preserve partial append page", advice)
        assert hint == os.POSIX_FADV_DONTNEED, advice
    if partial:
        assert any(size % PAGE for _, _, size, _ in advice), "partial append page not exercised"
    return advice


def insert_batch(start):
    return "INSERT INTO t VALUES " + ",".join(
        "(%d,'%s')" % (key, PAYLOAD) for key in range(start, start + 8))


def expect_rows(engine, count):
    response = engine.request("SELECT id,payload FROM t ORDER BY id")
    rows = response.get("rows", [])
    assert len(rows) == count, (len(rows), count)
    for key, row in enumerate(rows):
        assert row == [key, PAYLOAD], ("recovered row mismatch", key)


def run_case(parent, library, strict=False, fail_advice=False, partial=False):
    with tempfile.TemporaryDirectory(prefix="mvdb-wal-advice-", dir=parent) as temporary:
        root = Path(temporary)
        directory, log = root / "db", root / "calls.txt"
        with environment({"LD_PRELOAD": str(library), "MVDB_TEST_ADVICE_LOG": str(log),
                          "MVDB_TEST_ADVICE_ERROR": "1" if fail_advice else None,
                          "MVDB_SYNC_COMMIT": "1" if strict else None}):
            engine = Engine(directory)
            try:
                engine.request("CREATE TABLE t(id BIGINT PRIMARY KEY,payload VARCHAR(65536))")
                create_advice = int((directory / "wal.bin").stat().st_size >= PAGE) if strict else 0
                check_calls(log, create_advice, int(strict))
                for batch in range(4):
                    sql = insert_batch(batch * 8)
                    if batch == 3:
                        # No log/stat/query/JSON decoding between final ACK and SIGKILL.
                        engine.request_then_kill(sql)
                    else:
                        engine.request(sql)
                    advice_count = create_advice + batch + 1 if strict else (batch + 1) // 2
                    check_calls(log, advice_count, batch + 2 if strict else 0)
            finally:
                engine.kill()
            advice_count = create_advice + 4 if strict else 2
            sync_count = 5 if strict else 0
            initial = check_calls(log, advice_count, sync_count, partial=partial)
            assert initial[-1][1] > initial[0][1], initial

            # Startup snapshots/truncates/reopens the old WAL. A second crash at
            # readiness must preserve all acknowledged rows before any query.
            Engine(directory, kill_on_ready=True)
            engine = Engine(directory)
            try:
                expect_rows(engine, 32)
                engine.request(insert_batch(32))
                # Fresh WAL has less than 1 MiB: no premature advice from an
                # old offset/counter. Strict mode still synchronizes each ACK.
                check_calls(log, advice_count + int(strict), sync_count + int(strict))
                engine.request_then_kill(insert_batch(40))
            finally:
                engine.kill()
            check_calls(log, advice_count + (2 if strict else 1),
                        sync_count + (2 if strict else 0), partial=partial)
            engine = Engine(directory)
            try:
                expect_rows(engine, 48)
            finally:
                engine.kill()
    print("PASS WAL cache advice: mode=%s, advice_error=%s, partial_tail=%s" %
          ("strict" if strict else "default", fail_advice, partial), flush=True)


def main():
    if not sys.platform.startswith("linux"):
        print("SKIP WAL cache advice (requires Linux LD_PRELOAD)")
        return
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        print("SKIP WAL cache advice (C++ compiler unavailable)")
        return
    with tempfile.TemporaryDirectory(prefix="mvdb-wal-advice-shim-") as temporary:
        temp = Path(temporary)
        source, library = temp / "shim.cpp", temp / "shim.so"
        source.write_text(SHIM)
        subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC",
                        str(source), "-ldl", "-o", str(library)], check=True)
        run_case(ROOT, library)
        run_case(ROOT, library, fail_advice=True)
        run_case(ROOT, library, strict=True, fail_advice=True)
        shared_memory = Path("/dev/shm")
        if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
            run_case(shared_memory, library, fail_advice=True, partial=True)
        else:
            print("SKIP partial WAL append page (/dev/shm unavailable)")
    print("PASS wal_cache_advice_test: threshold, prefix/page bounds, sync policy, "
          "advice errors, ACK-SIGKILL recovery and truncate/reopen")


if __name__ == "__main__":
    main()
