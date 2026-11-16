#!/usr/bin/env python3
"""Write/recovery cross-check with ordinary eight-decimal random float32 data.

Prepare requests before timing, alternate binaries, and verify 50 exact-query
answers against the first binary before and after SIGKILL. This checks WAL
changes independently of the dyadic benchmark's integer oracle; it does not
provide an independent float-distance oracle or an official score.
MVDB_SYNC_COMMIT is inherited. SIGKILL leaves the host kernel/page cache alive;
this does not test power-loss durability. Old binaries may always synchronize.
"""
import os
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

import performance_bench as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binaries", nargs="+")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cpus", default="4,5")
    parser.add_argument("--data-parent", type=Path, default=bench.ROOT)
    args = parser.parse_args()
    cpus = set(map(int, args.cpus.split(",")))
    rng = random.Random(73199)
    rows = ["(%d,%d,'[%s]')" % (i, i % 10, ",".join(
        "%.8f" % rng.uniform(-1, 1) for _ in range(128))) for i in range(10000)]
    schema = bench.encode(0, "CREATE TABLE t(id BIGINT PRIMARY KEY,grp BIGINT,v VECTOR(128))")
    writes, queries = [], []
    for start in range(0, 10000, 100):
        for sql in ["BEGIN", "INSERT INTO t VALUES " + ",".join(rows[start:start + 100]), "COMMIT"]:
            writes.append(bench.encode(len(writes) + 1, sql))
    for i in range(50):
        queries.append(bench.encode(10000 + i, "SELECT id FROM t%s ORDER BY "
                       "L2_DISTANCE(v,'[%s]') LIMIT 10" % (
                           " WHERE grp=3" if i % 2 else "",
                           ",".join("%.8f" % rng.uniform(-1, 1) for _ in range(128)))))

    runs = {binary: [] for binary in args.binaries}
    expected = None
    for rep in range(args.rounds):
        shift = rep % len(args.binaries)
        for binary in args.binaries[shift:] + args.binaries[:shift]:
            with tempfile.TemporaryDirectory(prefix="mvdb-decimal-", dir=args.data_parent) as directory:
                engine = bench.Engine(binary, directory, cpus, 30)
                try:
                    engine.request(schema)
                    before = bench.sample_process(engine.process.pid)
                    start = time.perf_counter()
                    for request in writes:
                        engine.request(request)
                    elapsed = time.perf_counter() - start
                    after = bench.sample_process(engine.process.pid)
                    answers = [engine.request(request) for request in queries]
                    if expected is None:
                        expected = answers
                    assert answers == expected
                    files = {path.name: path.stat().st_size for path in Path(directory).iterdir()
                             if path.is_file()}
                    engine.kill()
                    engine = bench.Engine(binary, directory, cpus, 30)
                    assert [engine.request(request) for request in queries] == expected
                    runs[binary].append(dict(write_s=elapsed, io=bench.delta(after, before), files=files,
                                            requested_commit_sync=os.environ.get("MVDB_SYNC_COMMIT", "0") == "1",
                                            crash_test="SIGKILL-with-kernel-alive",
                                            power_loss_tested=False))
                    print(rep, binary, elapsed, files, file=sys.stderr, flush=True)
                finally:
                    engine.kill()
    print(json.dumps({binary: dict(median_write_s=statistics.median(
        run["write_s"] for run in values), runs=values) for binary, values in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
