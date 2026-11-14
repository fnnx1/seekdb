# Exact TopK performance, 2026-11-14

The candidate preserves exact search and acknowledged-commit durability.
Compared with commit `03e528d`, the unfiltered query phase is 2.23x faster,
all queries are 1.92x faster, and durable writes are 1.53x faster in the
controlled local benchmark. These are measured phase times, not the formal E
score. No formal evaluation or submission was made.

## Measurement

Seven rounds per binary, alternating which runs first. Both engines were
restricted to CPU 0 and CPU 1 on the same Intel Core Ultra 7 265K host. Fresh
databases were placed on NVMe/btrfs, with 100 separate BEGIN/INSERT/COMMIT
transactions of 100 rows each. The driver waited for every response.

The dataset contains 10,000 x 128D deterministic random float32 vectors, ten
groups, shuffled primary keys and deliberately duplicated vectors to exercise
ties. There are 500 filtered and 500 unfiltered exact Top-10 queries. All
requests were generated and JSON-encoded before timing. The first query's
cache construction is included. The driver, oracle and restart time are
excluded from the write/query phase times and reported separately.

The vectors are multiples of 1/16, permitting an independent exhaustive
integer oracle whose distances are exactly representable in float32. Every
query result was checked, followed by SIGKILL, recovery and another check of
all 1,000 results. Each binary passed 7,000 initial and 7,000 recovered queries.

| Metric, median | 03e528d | Candidate | Change |
| --- | ---: | ---: | ---: |
| Durable write, 100 commits | 265.33 ms | 173.36 ms | 1.53x faster |
| 500 filtered queries | 26.81 ms | 18.57 ms | 1.44x faster |
| 500 unfiltered queries | 64.12 ms | 28.74 ms | 2.23x faster |
| All 1,000 queries | 90.88 ms | 47.28 ms | 1.92x faster |
| Engine peak RSS | 15.15 MiB | 14.34 MiB | 5.4% lower |
| WAL logical size | 5,355,200 B | 5,355,200 B | unchanged format/size |
| Write-phase physical reads | 2,043,904 B | 0 B | tail-page rereads eliminated |

I/O uses `/proc/<pid>/io`, whose accounting is not the platform's unknown B
metric. Median physical write accounting was 12,333,056 B and 12,316,672 B;
btrfs allocation and metadata work make these larger and noisier than the
logical WAL size. Retaining the partial append page costs at most one page.

The old `/tmp/bench_parts.py` included Python vector/SQL generation in its
timers, used only 200 distinct vectors, and placed data on tmpfs. Its numbers
cannot establish engine throughput or real storage flush latency. Decimal
precision also affects parsing cost: the dyadic workload here has shorter
literals than the separate eight-decimal experimental workload. Compare
binaries within one workload, not times from those different scripts.

## Implementation

* Buffered protocol input and `from_chars(double)` avoid per-character stdio
  and per-coordinate locale conversion. The input loop handles interrupted
  reads, preserving graceful SIGTERM shutdown.
* Stored row cells occupy 16 bytes instead of 88. Strings and temporary owned
  vectors have explicit copy/move ownership; stored vectors retain offsets.
  INSERT avoids constructing defaults for supplied columns. Spilling vectors
  releases the previous 128 MiB memory allocation.
* A derived cache contains compact scan metadata and one byte per coordinate
  with signed codes in [-63,63]. All cache allocation capacities together are
  capped at 32 MiB per table. Large tables use the existing streaming scan.
* For reconstructed vectors x' and q', the triangle inequality gives
  `distance(x,q) >= distance(x',q') - error(x) - error(q)`. The scan rejects a
  row only when this lower bound exceeds the current TopK radius, widened for
  floating-point rounding. Every survivor is ranked using the original
  float32 distance kernel and primary-key tie rule. Out-of-range queries,
  descending searches and unsupported dimensions use exhaustive distances.
* Every acknowledged commit still calls `fdatasync`. WAL cache eviction stops
  at the last complete page, retaining the partial page needed by the next
  append. No batching across acknowledgements or delayed durability is used.

Quantized rejection can weaken on outliers, near ties and other distributions;
correctness is preserved while throughput can approach the exhaustive scan.
The formal T/B/M normalization and driver behavior are unavailable, so a
formal E multiplier or leaderboard placement cannot be derived from this run.
The full 4M-row case has not been run; the existing 100k-row load/restart test
passed at 88.5 MiB peak RSS.

## Reproduce and verify

```bash
git show 03e528d:src/minivectordb.cpp > /tmp/minivectordb-baseline.cpp
g++ -O2 -std=c++17 -fopenmp -o /tmp/minivectordb-baseline /tmp/minivectordb-baseline.cpp
make
python3 tests/performance_bench.py --binary /tmp/minivectordb-baseline \
  --binary ./minivectordb --rounds 7 --cpus 0,1 \
  --data-parent "$PWD" --json /tmp/minivectordb-performance.json
./scripts/check.sh
```

The full local check passed: 19 dynamic tests, 8 legacy protocol tests, 20,000
fuzz statements, 100k-row load/restart, exact rejection regressions, ASan/UBSan
and ownership/snapshot checks, plus the official handshake and 20 public
JSONL cases. The new regression also exposed and fixed an existing temporary
string lifetime bug when parsing negative SQL numbers.

Raw rounds and candidate source hash: [2026-11-14-exact-pruning.json](2026-11-14-exact-pruning.json).
Check output: [2026-11-14-check.log](2026-11-14-check.log).
