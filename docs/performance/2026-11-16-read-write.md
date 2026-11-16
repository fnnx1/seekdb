# Further read and durable-write optimization, 2026-11-16

This round does **not** establish another 1.5x improvement over `ae9fc6c`
(the same engine source as `1342e8b`). The selected changes improve the
default mixed-query workload by 1.04x and durable writes by 1.09x in the final
seven-round run. Write gains vary with filesystem activity. No platform
evaluation or submission was performed, and these timings do not predict E.

## Final measurements

Both engines use `g++ -O2 -std=c++17 -fopenmp`, two CPU cores on the same
Core Ultra 7 265K host, and fresh databases on NVMe/btrfs. Requests are
pre-generated; every BEGIN, INSERT and COMMIT waits for its response. Each
binary runs seven times with alternating execution order. First-query cache
construction is included. All rounds, including slower ones, are retained.

| Workload | Metric | Baseline | Candidate | Median ratio | Median paired ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Random dyadic, duplicate random vectors | 100 durable commits | 145.73 ms | 133.78 ms | 1.09x | 1.15x |
| Same | 500 filtered queries | 12.42 ms | 11.74 ms | 1.06x | 1.06x |
| Same | 500 unfiltered queries | 15.55 ms | 15.32 ms | 1.01x | 1.03x |
| Same | All 1,000 queries | 27.98 ms | 26.99 ms | 1.04x | 1.04x |
| Historical zero-vector tie cluster | 100 durable commits | 168.75 ms | 132.07 ms | 1.28x | 1.21x |
| Same | All 1,000 queries | 26.73 ms | 25.75 ms | 1.04x | 1.04x |
| Ordinary random eight-decimal floats | 100 durable commits | 139.57 ms | 124.49 ms | 1.12x | 1.05x |

The paired column takes the median of baseline/candidate ratios within each
alternating round. Its difference from the ratio of medians shows substantial
write variability; a stable 20–30% write gain is not established. An earlier
ordinary-decimal run measured 1.16x by medians but only 1.07x paired; its timings
are retained in `read-write-experiments/notes.json`.

For the default workload, Linux physical `write_bytes` falls from 12,267,520
to 11,157,504 bytes (9.0%). The zero-cluster run falls 13.0%, and the decimal
run falls 13.6%. These are `/proc` measurements, not the platform's unknown B.
The logical WAL grows from 5,355,200 to 5,738,496 bytes (7.2%) due to padding.
Query disk I/O remains zero. Peak engine RSS is 14.32 to 14.34 MiB in the
default run, within 0.03 MiB. Ordinary startup increases 4.66 to 5.31 ms,
including checked directory durability; recovery falls 22.02 to 17.21 ms.

## Selected changes

* Cache up to 16 SELECT AST templates when SQL is byte-identical except for
  one quoted vector literal. Only the vector is reparsed; table and column
  names bind on every execution. Escaped strings and bare vectors use the
  original parser. Key size, literal span and retained vector dimensions are
  bounded. SQL errors, changing dimensions and DDL remain observable.
* Replace heap pop/push with one root replacement and sift-down. Distance
  order, descending queries and ascending primary-key ties are preserved.
* Best-effort NOCOW for an empty WAL when supported by the filesystem. A
  CRC-protected type-4 no-op before each commit marker aligns its end to a
  page boundary. Later appends avoid rewriting acknowledged commit sectors.
  Every commit still calls `fdatasync`; unsupported filesystems retain the
  existing unpadded WAL. Existing readers ignore type 4, and the new reader
  also accepts historical unpadded logs.
* Preserve alignment on NOCOW WAL reopen, reject unsafe unaligned direct
  appends, stop startup after failed truncation, and reject subsequent writes
  after any WAL write/flush/sync failure. WAL directory creation is checked
  before readiness, and snapshot directory-sync failure prevents truncation.

## Where the time remains

Separate exploratory measurements on CPU 6/7 found about 9–10 ms per 1,000
minimal JSONL round trips, compared with 28–29 ms for normal exact queries.
An in-process split measured approximately 13.3 ms for already parsed search
and result construction. SQL parsing was 2.35 ms, JSON decoding 0.21 ms and
serialization 0.36 ms. These runs have different placement and warm-cache
conditions from the final measurements; their times are not added together.
They show that a 1.5x end-to-end query gain requires much more than shaving
another few percent from distance arithmetic.

The following candidates were measured and excluded:

| Experiment | Result |
| --- | --- |
| Quantized candidate warm-up followed by exact conservative scanning | 128/256-row variants neutral or slower; 512 rows +3%; full pass +23% time |
| Lossless byte shuffle/LZ WAL compression | Dyadic WAL 5.35 to 2.35 MB, little time gain; ordinary decimals slower and only 8% smaller |
| Preinitialized NOCOW/O_DIRECT | About 1.29x write, but startup +6.5 ms and recovery 22.8 to 92 ms |
| Typed INSERT bypass | About 1.09x on tmpfs, under 3% expected on disk; added parser complexity |
| SWAR decimal parsing | About 1.02x; short decimal cases regress |
| O3 without fast-math/native | About 1.006x mixed query; retain O2 |
| One OpenMP thread | Mixed queries slower by about 12%; retain two |
| `GOMP_SPINCOUNT=0` | Mixed queries slower by 1–3%, including shared driver/engine CPUs |

Raw read experiments are in `read-write-experiments/`. Compression and
O_DIRECT remain isolated prototypes and are not enabled by the candidate.

## Reproduction and verification

```
git show ae9fc6c:src/minivectordb.cpp > /tmp/mvdb-baseline.cpp
g++ -O2 -std=c++17 -fopenmp /tmp/mvdb-baseline.cpp -o /tmp/mvdb-baseline
make
python3 tests/performance_bench.py --binary /tmp/mvdb-baseline \
  --binary ./minivectordb --rounds 7 --cpus 0,1 --json /tmp/copies.json
python3 tests/performance_bench.py --binary /tmp/mvdb-baseline \
  --binary ./minivectordb --rounds 7 --cpus 0,1 --ties zeros --json /tmp/zeros.json
python3 tests/write_decimal_bench.py /tmp/mvdb-baseline ./minivectordb \
  --rounds 7 --cpus 4,5 > /tmp/decimal.json
./scripts/check.sh
```

Run performance commands sequentially, without the clean-build checks in
parallel. The two dyadic workloads verified all 28,000 initial and 28,000
SIGKILL-recovered queries across both binaries using an independent integer
oracle. The decimal workload cross-checks 50 exact-query responses per run
against the baseline and after recovery; it does not have an independent
floating-point oracle.

The full check passed: 19 dynamic tests, exact pruning regressions, 440 SELECT
template comparisons, 15 WAL truncation positions, legacy WAL replay, 12
alternating SIGKILL/normal restarts, tmpfs fallback, eight protocol tests,
20,000 fuzz statements, 100,000-row restart, ASan/UBSan kernels and ownership,
and all 20 public JSONL cases. Isolated fault injection additionally checked
partial writes, sync failure and failed truncation. Check output is retained
in `2026-11-16-read-write-check.txt`.

Final raw runs: `2026-11-16-read-write-{copies,zeros,decimal}.json`.

Source SHA256:

* Baseline: `4228f802a394208756b861a1ac43eeb1ddfa6dd0378e348d008680acf43b186e`
* Candidate: `40f4dfcfb62db852d34d2f6dec00e9748ac5eaeae09dd9a64fc79f18c717d57e`
