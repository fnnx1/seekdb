# Additional exact-search performance, 2026-11-16

Compared with the current `b1d105c` baseline (which already includes paired
integer dot products and survivor prefetch), the candidate improves unfiltered
queries by 1.59–1.74x and the mixed query workload by 1.44–1.52x. Exact TopK,
primary-key tie ordering, WAL encoding and per-commit `fdatasync` remain intact.
No formal platform evaluation was run; the unknown E normalization prevents a
score forecast.

## Reproducible results

Both workloads have 10,000 x 128D float32 rows, ten groups, 100 transactions
of 100 rows each and 1,000 exact Top-10 queries (500 filtered). Both binaries
use CPU 0 and CPU 1 on the same Core Ultra 7 265K host. Data lives on NVMe/btrfs.
The driver prepares every SQL/JSON request before timing and waits for each
response. Seven fresh-database rounds per binary alternate execution order.
The first query's cache build is included. The table reports medians.

| Workload | Metric | b1d105c | Candidate | Speedup |
| --- | --- | ---: | ---: | ---: |
| Random vectors with repeated random vectors for ties | 500 filtered queries | 16.09 ms | 12.85 ms | 1.25x |
| Same | 500 unfiltered queries | 25.93 ms | 16.31 ms | 1.59x |
| Same | All 1,000 queries | 41.78 ms | 28.96 ms | 1.44x |
| Same | Durable write, 100 commits | 175.88 ms | 159.59 ms | 1.10x |
| Historical 20-zero-vector cluster | 500 filtered queries | 15.78 ms | 12.41 ms | 1.27x |
| Same | 500 unfiltered queries | 25.54 ms | 14.66 ms | 1.74x |
| Same | All 1,000 queries | 41.36 ms | 27.23 ms | 1.52x |
| Same | Durable write, 100 commits | 150.52 ms | 148.86 ms | 1.01x |

Engine peak RSS remains approximately 14.3 MiB (changes below 0.1 MiB across
these medians). Both versions produce the same 5,355,200-byte WAL; query I/O
and write-phase physical reads are zero. Physical write accounting remains
noisy on btrfs. The small write changes do not establish a stable write gain.

The benchmark now defaults to repeated random vectors (`--ties copies`). The
previous default inserted twenty zeros, creating a center cluster that alters
which candidates survive exact bounds. `--ties zeros` preserves that workload
for historical comparisons. Both variants still check ties using an independent
integer-distance oracle. All initial and SIGKILL-recovered answers passed:
14,000 initial and 14,000 recovered queries per binary across both workloads.

Other exploratory runs measured mixed-query gains between about 1.4x and 1.6x;
host scheduling and clock changes affect absolute times. The improvement is
distribution dependent and should not be described as a universal 2x gain.

## Why the implementation helps

The previous scan called one integer dot product per row, reduced SIMD lanes
to one integer, loaded per-row norm/error metadata, and performed a floating
comparison. The 128D cache now stores four-coordinate stripes from eight rows
in each 32-byte span. A SIMD lane accumulates one row's dot product, so one
kernel computes eight code distances and produces an eight-bit survivor mask.
Rejected blocks avoid row-metadata lookups and individual function calls.

The cache is stably grouped by its first non-primary BIGINT column. A matching
integer equality predicate scans a contiguous range of blocks. Other filters,
primary-key predicates and boundary rows gather one lane from that same code
buffer. There is one quantized coordinate copy, two uint32 row mappings and
one int bias per row. All persistent derived capacities remain within the
existing 32 MiB per-table limit; large tables retain streaming scans.

For code distance C, scale s, row reconstruction error e_i and query error e_q,
the triangle inequality gives `s * sqrt(C) <= distance(x,q) + e_i + e_q`.
The cache's largest e_i yields one conservative threshold for all lanes.
The current TopK radius is widened for float32 rounding, converted to integer
code units, rounded upward and compared inclusively. The bound can admit
extra candidates; every surviving candidate is still ranked with the original
float32 L2 kernel. Each thread keeps its own heap and threshold, and queued
survivors are rechecked after that threshold tightens.

Short decimal literals additionally use an exact integer mantissa divided by
an exact power of ten (at most fifteen digits), which preserves the rounded
double then float32 conversion. Exponents, longer literals and uncommon
syntax retain the existing parser. Plain SQL strings copy one span when no
unescaping is needed, and Parser references its caller's SQL buffer.

## Experiments excluded from the candidate

* Unrolling the existing per-row dot product gave only about 2% mixed-query
  benefit. Four-row processing helped less than the eight-row layout.
* Progressive 32-coordinate rejection approximately doubled query time on
  the random workload: reduction and branch costs outweighed early exits.
* O3, single-thread scans and optional AVX-VNNI did not beat the retained
  AVX2 implementation consistently. Small result-materialization changes
  were also below a useful measurement threshold.
* WAL preallocation improved some write medians by roughly 10–30%, but
  increased physical writes and lacked a stable 1.5x result. Aligned padding
  added WAL bytes without a consistent gain. O_DSYNC prototypes caused more
  synchronous writes and regressed substantially. None changed the WAL path.

## Validation and reproduction

`scripts/check.sh` passed the full regression, sanitizer, 100k-row restart,
20,000-statement fuzz and official 20-public-case suite. New checks compare
SIMD lane distances and masks with independent scalar integer arithmetic,
including inclusive tie thresholds and extreme code values, and compare
decimal float32 bit patterns with libc conversion. Query regressions cover
grouped/ungrouped predicates, other BIGINT columns, general predicates without
a grouping column, uneven block and worker boundaries, deleted rows,
transactions, floating extremes and cache invalidation. The existing exact
suite also passed with `OMP_THREAD_LIMIT=1`.

```bash
git show b1d105c:src/minivectordb.cpp > /tmp/minivectordb-baseline.cpp
g++ -O2 -std=c++17 -fopenmp -o /tmp/minivectordb-baseline /tmp/minivectordb-baseline.cpp
make
python3 tests/performance_bench.py --binary /tmp/minivectordb-baseline \
  --binary ./minivectordb --rounds 7 --cpus 0,1 --ties copies \
  --json /tmp/block-scan-copies.json
python3 tests/performance_bench.py --binary /tmp/minivectordb-baseline \
  --binary ./minivectordb --rounds 7 --cpus 0,1 --ties zeros \
  --json /tmp/block-scan-zeros.json
./scripts/check.sh
```

Raw rounds and source hash:
[random ties](2026-11-16-block-scan-copies.json),
[historical zero ties](2026-11-16-block-scan-zeros.json).
Full check output: [2026-11-16-check.txt](2026-11-16-check.txt).
