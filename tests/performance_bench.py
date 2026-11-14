#!/usr/bin/env python3
"""Reproducible local exact-search benchmark; this is not the official score.

All SQL and JSONL requests are prepared before timing. Each request waits for
its response, including BEGIN / INSERT / COMMIT separately for every batch.
The default workload is 10,000 x 128D, 100 commits, 500 filtered and 500 full
exact Top-10 searches. The independent exhaustive integer oracle checks every
answer, including primary-key tie ordering. No ANN results are accepted.

Example (alternates the order of binaries between rounds):
  python3 tests/performance_bench.py --binary /tmp/baseline \
      --binary ./minivectordb --rounds 7 --cpus 0,1 --json /tmp/perf.json

Vectors use random multiples of 1/16 in [-4, 4). Their squared distances are
exact in float32 for dimensions <= 1024, so the oracle does not mistake a
float32 rounding difference for an incorrect result. Unlike the historical
/tmp/bench_parts.py workload, these vectors do not repeat every 200 rows.
The generator, oracle compilation, result comparison, /proc sampling and
post-acknowledgement SIGKILL recovery check are outside measured phases.
I/O is Linux per-process /proc accounting, not the platform's unknown B metric.
RSS is the engine's VmHWM; driver/oracle memory is excluded. Fresh data
directories default to the repository filesystem, since /tmp can be tmpfs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import selectors
import statistics
import struct
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
ORACLE = r"""
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
template<class T> T read(std::istream &in) {
  T x{}; in.read(reinterpret_cast<char*>(&x), sizeof(x));
  if (!in) std::exit(2); return x;
}
int main(int argc, char **argv) {
  std::ifstream in(argv[1], std::ios::binary);
  const auto n=read<uint32_t>(in), d=read<uint32_t>(in);
  const auto q=read<uint32_t>(in), k=read<uint32_t>(in);
  std::vector<int16_t> rows(size_t(n)*d);
  std::vector<int64_t> ids(n);
  std::vector<int32_t> groups(n);
  for (uint32_t i=0; i<n; ++i) {
    ids[i]=read<int64_t>(in); groups[i]=read<int32_t>(in);
    in.read(reinterpret_cast<char*>(&rows[size_t(i)*d]), d*sizeof(int16_t));
  }
  std::vector<int16_t> query(d);
  for (uint32_t t=0; t<q; ++t) {
    const auto group=read<int32_t>(in);
    in.read(reinterpret_cast<char*>(query.data()), d*sizeof(int16_t));
    std::vector<std::pair<int64_t,int64_t>> candidates;
    for (uint32_t i=0; i<n; ++i) {
      if (group>=0 && groups[i]!=group) continue;
      int64_t distance=0;
      for (uint32_t j=0; j<d; ++j) {
        const int64_t delta=int64_t(rows[size_t(i)*d+j])-query[j];
        distance+=delta*delta;
      }
      candidates.emplace_back(distance,ids[i]);
    }
    const auto take=std::min(size_t(k),candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin()+take,
                      candidates.end());
    for (size_t i=0; i<take; ++i)
      std::cout << (i ? " " : "") << candidates[i].second;
    std::cout << '\n';
  }
}
"""


def encode(ident, sql=None, op="execute"):
    request = {"id": ident, "op": op}
    if sql is not None:
        request.update(sql=sql, search_mode="exact")
    return json.dumps(request, separators=(",", ":")).encode() + b"\n"


def prepare(args, directory):
    started = time.perf_counter()
    rng = random.Random(args.seed)
    ids = list(range(args.rows))
    rng.shuffle(ids)  # PK ordering must not depend on insertion order.
    rows = [(pk, pk % args.groups,
             [rng.randrange(-64, 64) for _ in range(args.dims)]) for pk in ids]
    # Identical vectors deliberately exercise (distance, PK) tie handling.
    for i in range(min(20, len(rows))):
        pk, group, _ = rows[i]
        rows[i] = (pk, group, [0] * args.dims)
    queries = [(i % args.groups if i < args.queries // 2 else -1,
                [rng.randrange(-64, 64) for _ in range(args.dims)])
               for i in range(args.queries)]
    if queries:
        queries[0] = (0, [0] * args.dims)
        queries[args.queries // 2] = (-1, [0] * args.dims)

    def vector_text(vector):
        return "[" + ",".join(format(x / 16, ".4f") for x in vector) + "]"

    ident = 1
    schema = encode(ident, "CREATE TABLE t (id BIGINT PRIMARY KEY, grp BIGINT, "
                    "v VECTOR(%d));" % args.dims)
    writes = []
    for start in range(0, args.rows, args.batch):
        records = ",".join("(%d,%d,'%s')" % (pk, group, vector_text(vector))
                           for pk, group, vector in rows[start:start + args.batch])
        for sql in ("BEGIN;", "INSERT INTO t VALUES " + records + ";", "COMMIT;"):
            ident += 1
            writes.append(encode(ident, sql))
    searches = []
    for group, vector in queries:
        condition = " WHERE grp = %d" % group if group >= 0 else ""
        sql = ("SELECT id FROM t%s ORDER BY L2_DISTANCE(v,'%s') LIMIT %d;"
               % (condition, vector_text(vector), args.k))
        ident += 1
        searches.append(encode(ident, sql))
    preparation_s = time.perf_counter() - started
    oracle_path = directory / "oracle.bin"
    with oracle_path.open("wb") as output:
        output.write(struct.pack("<IIII", args.rows, args.dims,
                                 args.queries, args.k))
        for pk, group, vector in rows:
            output.write(struct.pack("<qi", pk, group))
            output.write(struct.pack("<%dh" % args.dims, *vector))
        for group, vector in queries:
            output.write(struct.pack("<i", group))
            output.write(struct.pack("<%dh" % args.dims, *vector))
    source = directory / "oracle.cpp"
    source.write_text(ORACLE)
    oracle = directory / "oracle"
    subprocess.run([args.cxx, "-O2", "-std=c++17", str(source), "-o", str(oracle)],
                   check=True, capture_output=True)
    started = time.perf_counter()
    expected = subprocess.check_output([str(oracle), str(oracle_path)], text=True)
    expected = [[int(value) for value in line.split()]
                for line in expected.splitlines()]
    if len(expected) != args.queries:
        raise AssertionError("oracle returned an incomplete result")
    return schema, writes, searches, expected, preparation_s, time.perf_counter() - started


def sample_process(pid):
    result = {}
    for line in Path("/proc/%d/status" % pid).read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            result[line.split(":")[0] + "_bytes"] = int(line.split()[1]) * 1024
    for line in Path("/proc/%d/io" % pid).read_text().splitlines():
        key, value = line.split(":")
        result[key] = int(value)
    return result


def delta(after, before):
    return {key: after[key] - before[key] for key in after
            if key not in ("VmHWM_bytes", "VmRSS_bytes")}


class Engine:
    def __init__(self, binary, directory, cpus, timeout):
        self.stderr = tempfile.TemporaryFile()
        environment = dict(os.environ)
        environment.pop("MVDB_LEGACY_STDIO", None)
        self.process = subprocess.Popen(
            [binary, str(directory)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, env=environment,
            preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        self.timeout = timeout
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        if self.receive() != {"type": "ready"}:
            raise AssertionError("invalid ready handshake")

    def receive(self):
        if not self.selector.select(self.timeout):
            raise TimeoutError("engine response exceeded %ss" % self.timeout)
        line = self.process.stdout.readline()
        if not line:
            self.stderr.seek(0)
            raise RuntimeError("engine exited: " + self.stderr.read().decode(errors="replace"))
        return json.loads(line)

    def request(self, payload):
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        response = self.receive()
        if response.get("ok") is not True:
            raise AssertionError(response)
        return response

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=self.timeout)
        self.close()

    def close(self):
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()


def check_answers(responses, expected):
    actual = [[row[0] for row in response.get("rows", [])] for response in responses]
    for i, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise AssertionError("exact TopK query %d: got %r, expected %r" % (i, got, want))
    return hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest()


def run_one(binary, args, workload):
    schema, writes, searches, expected, _, _ = workload
    with tempfile.TemporaryDirectory(prefix="mvdb-perf-", dir=args.data_parent) as temp:
        started = time.perf_counter()
        engine = Engine(binary, temp, args.cpus, args.timeout)
        startup_s = time.perf_counter() - started
        try:
            started = time.perf_counter()
            engine.request(schema)
            schema_s = time.perf_counter() - started
            before = sample_process(engine.process.pid)
            started = time.perf_counter()
            for payload in writes:
                engine.request(payload)
            write_s = time.perf_counter() - started
            written = sample_process(engine.process.pid)
            responses = []
            started = time.perf_counter()
            for payload in searches[:args.queries // 2]:
                responses.append(engine.request(payload))
            filtered_s = time.perf_counter() - started
            started = time.perf_counter()
            for payload in searches[args.queries // 2:]:
                responses.append(engine.request(payload))
            unfiltered_s = time.perf_counter() - started
            queried = sample_process(engine.process.pid)
            digest = check_answers(responses, expected)
            disk = {p.name: p.stat().st_size for p in Path(temp).iterdir() if p.is_file()}
            # Kill after all acknowledged commits, without any graceful close.
            # Recovery must reproduce the already verified exact answers.
            engine.kill()
            started = time.perf_counter()
            engine = Engine(binary, temp, args.cpus, args.timeout)
            recovery_s = time.perf_counter() - started
            recovered = [engine.request(payload) for payload in searches]
            check_answers(recovered, expected)
            started = time.perf_counter()
            engine.request(encode(99999999, op="shutdown"))
            if engine.process.wait(timeout=args.timeout) != 0:
                raise AssertionError("nonzero shutdown status")
            shutdown_s = time.perf_counter() - started
            engine.close()
            return dict(write_s=write_s, filtered_s=filtered_s,
                        unfiltered_s=unfiltered_s, query_s=filtered_s + unfiltered_s,
                        engine_peak_rss_bytes=queried["VmHWM_bytes"],
                        startup_s=startup_s, schema_s=schema_s, recovery_s=recovery_s,
                        shutdown_s=shutdown_s, write_io=delta(written, before),
                        query_io=delta(queried, written), files_before_crash=disk,
                        exact_queries_verified=len(expected),
                        recovery_queries_verified=len(expected), result_sha256=digest)
        finally:
            if engine.process.poll() is None:
                engine.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", action="append", help="repeat to compare binaries")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dims", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--cpus", help="comma-separated engine CPU affinity; default first two allowed CPUs")
    parser.add_argument("--json", type=Path, help="save all raw rounds and medians")
    parser.add_argument("--data-parent", type=Path, default=ROOT,
                        help="writable directory for fresh databases (default repository root)")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()
    for name in ("rounds", "rows", "dims", "queries", "batch", "groups", "k"):
        if getattr(args, name) <= 0:
            parser.error("--%s must be positive" % name)
    if args.dims > 1024 or args.queries % 2:
        parser.error("--dims must be <=1024; --queries must be even")
    allowed = os.sched_getaffinity(0)
    args.cpus = set(map(int, args.cpus.split(","))) if args.cpus else set(sorted(allowed)[:2])
    if not args.cpus or not args.cpus <= allowed:
        parser.error("--cpus must be a nonempty subset of %s" % sorted(allowed))
    if not args.data_parent.is_dir():
        parser.error("--data-parent must be an existing writable directory")
    binaries = [str(Path(binary).resolve()) for binary in
                (args.binary or [ROOT / "minivectordb"])]
    if any(not os.access(binary, os.X_OK) for binary in binaries):
        parser.error("every --binary must exist and be executable")
    with tempfile.TemporaryDirectory(prefix="mvdb-perf-driver-") as directory:
        workload = prepare(args, Path(directory))
        report = dict(workload=dict(rows=args.rows, dims=args.dims, queries=args.queries,
                                   batch=args.batch, commits=(args.rows + args.batch - 1)//args.batch,
                                   groups=args.groups, k=args.k, seed=args.seed,
                                   engine_cpus=sorted(args.cpus), dataset="random-dyadic-float32",
                                   data_parent=str(args.data_parent.resolve()),
                                   data_filesystem=subprocess.check_output(
                                       ["stat", "-f", "-c", "%T", str(args.data_parent)], text=True).strip(),
                                   mode="exact", requests_preencoded=True),
                      preparation_s=workload[4], oracle_s=workload[5], binaries={})
        runs = {binary: [] for binary in binaries}
        for repeat in range(args.rounds):
            offset = repeat % len(binaries)
            for binary in binaries[offset:] + binaries[:offset]:
                result = run_one(binary, args, workload)
                runs[binary].append(result)
                print("round %d %s: write %.6fs, filtered %.6fs, full %.6fs, RSS %.2fMiB"
                      % (repeat + 1, binary, result["write_s"], result["filtered_s"],
                         result["unfiltered_s"], result["engine_peak_rss_bytes"] / 2**20),
                      file=sys.stderr, flush=True)
        for binary, values in runs.items():
            keys = ("write_s", "filtered_s", "unfiltered_s", "query_s",
                    "engine_peak_rss_bytes", "startup_s", "schema_s", "recovery_s", "shutdown_s")
            report["binaries"][binary] = dict(
                median={key: statistics.median(value[key] for value in values) for key in keys},
                rounds=values)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
