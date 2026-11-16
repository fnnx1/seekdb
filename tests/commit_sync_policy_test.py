#!/usr/bin/env python3
"""Linux commit flush policy and interrupted-checkpoint correctness checks.

Uses a temporary LD_PRELOAD shim, temporary databases, and SIGKILL only.
MINIVECTORDB selects the engine; no global sync or cache dropping is used.
"""
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
BIN = str(Path(os.environ.get("MINIVECTORDB", ROOT / "minivectordb")).resolve())
SHIM = r'''
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <dlfcn.h>
#include <unistd.h>

static bool named(const char *path, const char *name) {
  if (!path) return false;
  const char *slash = std::strrchr(path, '/');
  return std::strcmp(slash ? slash + 1 : path, name) == 0;
}
static bool wal_fd(int fd) {
  char proc[64], path[4096];
  std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
  const ssize_t n = ::readlink(proc, path, sizeof(path) - 1);
  if (n < 0) return false;
  path[n] = 0;
  return named(path, "wal.bin");
}
static bool enabled(const char *name) {
  const char *value = std::getenv(name);
  return value && std::strcmp(value, "1") == 0;
}
static void mark(const char *message) { (void)::write(2, message, std::strlen(message)); }
static void stop_at(const char *point) {
  static bool stopped = false;
  const char *requested = std::getenv("MVDB_TEST_STOP_POINT");
  if (!stopped && requested && std::strcmp(requested, point) == 0) {
    stopped = true;
    mark("MVDB_TEST_CHECKPOINT_STOP\n");
    ::raise(SIGSTOP);
  }
}
extern "C" int fdatasync(int fd) {
  static auto real = reinterpret_cast<int (*)(int)>(::dlsym(RTLD_NEXT, "fdatasync"));
  static bool failed = false;
  if (wal_fd(fd) && enabled("MVDB_TEST_FDATASYNC_ERROR")) {
    mark("MVDB_TEST_WAL_FDATASYNC\n");
    if (!failed) { failed = true; errno = EIO; return -1; }
  }
  return real(fd);
}
extern "C" int fflush(FILE *file) {
  static auto real = reinterpret_cast<int (*)(FILE *)>(::dlsym(RTLD_NEXT, "fflush"));
  static bool failed = false;
  if (file && wal_fd(::fileno(file)) && enabled("MVDB_TEST_FFLUSH_ERROR") && !failed) {
    failed = true;
    mark("MVDB_TEST_WAL_FFLUSH_ERROR\n");
    errno = EIO;
    return EOF;
  }
  return real(file);
}
extern "C" int rename(const char *old_path, const char *new_path) {
  static auto real = reinterpret_cast<int (*)(const char *, const char *)>(::dlsym(RTLD_NEXT, "rename"));
  const bool snapshot = named(old_path, "snapshot.tmp") && named(new_path, "snapshot.bin");
  if (snapshot) stop_at("snapshot_rename_before");
  const int result = real(old_path, new_path);
  if (snapshot && result == 0) stop_at("snapshot_rename_after");
  return result;
}
extern "C" FILE *fopen(const char *path, const char *mode) {
  static auto real = reinterpret_cast<FILE *(*)(const char *, const char *)>(::dlsym(RTLD_NEXT, "fopen"));
  const bool truncate = named(path, "wal.bin") && mode && std::strcmp(mode, "wb") == 0;
  if (truncate) stop_at("wal_truncate_before");
  FILE *result = real(path, mode);
  if (truncate && result) stop_at("wal_truncate_after");
  return result;
}
'''


class Engine:
    def __init__(self, directory, environment=None, ready=True):
        env = dict(os.environ)
        for name in ("MVDB_LEGACY_STDIO", "MVDB_SYNC_COMMIT", "LD_PRELOAD",
                     "MVDB_TEST_FDATASYNC_ERROR", "MVDB_TEST_FFLUSH_ERROR",
                     "MVDB_TEST_STOP_POINT"):
            env.pop(name, None)
        env.update(environment or {})
        self.error = tempfile.TemporaryFile()
        self.process = subprocess.Popen([BIN, str(directory)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.error, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.ident = 0
        self.log = ""
        if ready:
            try:
                assert self.receive() == {"type": "ready"}
            except BaseException:
                self.kill()
                raise

    def receive(self):
        if not self.selector.select(15):
            raise TimeoutError("engine response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("engine exited before response")
        return json.loads(line)

    def request(self, sql):
        self.ident += 1
        request = dict(id=self.ident, op="execute", sql=sql, search_mode="exact")
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()
        return self.receive()

    def ok(self, sql):
        response = self.request(sql)
        assert response.get("ok") is True, (sql, response)
        return response.get("rows", [])

    def kill(self):
        if self.error.closed:
            return
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=15)
        self.error.seek(0)
        self.log = self.error.read().decode(errors="replace")
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.error.close()


def expect_ids(directory, expected):
    engine = Engine(directory)
    try:
        assert engine.ok("SELECT id FROM t ORDER BY id") == [[pk] for pk in expected]
    finally:
        engine.kill()


def wait_stopped(process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        pid, status = os.waitpid(process.pid, os.WNOHANG | os.WUNTRACED)
        if pid:
            if os.WIFSTOPPED(status):
                assert os.WSTOPSIG(status) == signal.SIGSTOP
                return
            process.returncode = os.waitstatus_to_exitcode(status)
            raise AssertionError("engine exited before checkpoint hook")
        time.sleep(0.01)
    raise TimeoutError("checkpoint hook did not stop engine")


def main():
    if not sys.platform.startswith("linux"):
        print("SKIP commit sync policy (requires Linux LD_PRELOAD and SIGSTOP)")
        return
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        print("SKIP commit sync policy (C++ compiler unavailable)")
        return
    with tempfile.TemporaryDirectory(prefix="mvdb-sync-policy-") as temporary:
        temp = Path(temporary)
        source, library = temp / "shim.cpp", temp / "shim.so"
        source.write_text(SHIM)
        subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC",
                        str(source), "-ldl", "-o", str(library)], check=True)
        fixture = temp / "fixture"
        engine = Engine(fixture)
        try:
            engine.ok("CREATE TABLE t(id BIGINT PRIMARY KEY,v VECTOR(4))")
            for pk in (1, 2, 3):
                engine.ok("BEGIN")
                engine.ok("INSERT INTO t VALUES(%d,'[%d,1,2,3]')" % (pk, pk))
                engine.ok("COMMIT")
        finally:
            engine.kill()

        default = temp / "default"
        shutil.copytree(fixture, default)
        engine = Engine(default, {"LD_PRELOAD": str(library), "MVDB_TEST_FDATASYNC_ERROR": "1"})
        try:
            engine.ok("BEGIN")
            engine.ok("INSERT INTO t VALUES(100,'[1,2,3,4]')")
            engine.ok("COMMIT")
        finally:
            engine.kill()  # Immediately after acknowledged COMMIT.
        assert "MVDB_TEST_WAL_FDATASYNC" not in engine.log, engine.log
        expect_ids(default, [1, 2, 3, 100])
        print("PASS default commits skip WAL fdatasync and survive immediate SIGKILL")

        for name, injection, strict, marker in (
                ("strict", "MVDB_TEST_FDATASYNC_ERROR", "1", "MVDB_TEST_WAL_FDATASYNC"),
                ("flush-error", "MVDB_TEST_FFLUSH_ERROR", "0", "MVDB_TEST_WAL_FFLUSH_ERROR")):
            directory = temp / name
            shutil.copytree(fixture, directory)
            engine = Engine(directory, {"LD_PRELOAD": str(library), injection: "1",
                                        "MVDB_SYNC_COMMIT": strict})
            try:
                engine.ok("BEGIN")
                engine.ok("INSERT INTO t VALUES(200,'[1,2,3,4]')")
                for _ in range(2):
                    response = engine.request("COMMIT")
                    assert response.get("ok") is False and response.get("error"), response
            finally:
                engine.kill()
            assert engine.log.count(marker) == 1, engine.log
            # Failed/unacknowledged COMMIT may have reached the kernel; only
            # earlier acknowledged rows must be guaranteed after recovery.
            recovered = Engine(directory)
            try:
                rows = recovered.ok("SELECT id FROM t ORDER BY id")
                assert all([pk] in rows for pk in (1, 2, 3)), rows
            finally:
                recovered.kill()
            print("PASS %s reports first failure and rejects later COMMIT" % name)

        for point in ("snapshot_rename_before", "snapshot_rename_after",
                      "wal_truncate_before", "wal_truncate_after"):
            directory = temp / point
            shutil.copytree(fixture, directory)
            engine = Engine(directory, {"LD_PRELOAD": str(library), "MVDB_TEST_STOP_POINT": point}, ready=False)
            try:
                wait_stopped(engine.process)
            finally:
                engine.kill()
            assert engine.log.count("MVDB_TEST_CHECKPOINT_STOP") == 1, engine.log
            expect_ids(directory, [1, 2, 3])
            print("PASS SIGKILL at startup checkpoint %s" % point)


if __name__ == "__main__":
    main()
