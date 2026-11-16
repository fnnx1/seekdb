# Process-crash commit policy, 2026-11-16

The implementation exceeds the requested approximate 1.5x target under the
failure model the user confirmed on 2026-11-16: an acknowledged COMMIT must
survive immediate database-process SIGKILL while the host kernel and its file
cache remain available. Power failure, kernel failure and disk corruption are
outside this default contract. **The comparison includes this authorized
change from per-commit disk synchronization; it is not a 2.7x algorithm gain
under identical power-loss guarantees.** No formal platform score is claimed.

## Implementation

The only engine changes relative to the saved working-tree baseline are a
constructor setting, one boolean, and an early return after successful WAL
`fflush`:

* The default, or `MVDB_SYNC_COMMIT=0`, writes complete WAL operation records
  and the commit marker, checks `fflush`, then allows the success response.
* `MVDB_SYNC_COMMIT=1` additionally executes the previous per-commit
  `fdatasync` and synchronized-page eviction path.
* WAL write/flush failures remain sticky: subsequent COMMIT cannot acknowledge
  later records across a failed WAL tail.
* CRCs, transaction grouping, undo, exact float32 search, primary-key ties,
  NOCOW/padding, file format and checkpoint ordering are unchanged. Snapshot
  and directory synchronization are retained.

A response not received by the client does not prove rollback: an incomplete
or unacknowledged commit may or may not have reached a complete WAL marker.

## Measurement

Both binaries use `g++ -O2 -std=c++17 -fopenmp` on the same Core Ultra 7 265K
host, NVMe/btrfs data directories, engine CPUs 0/1 and driver CPU 2. SQL and
JSON requests are prepared outside timing. Each BEGIN, INSERT and COMMIT
waits for its response. Every database starts fresh; the first query's cache
construction is included. Seven alternating rounds are retained in full.
There were no parallel benchmarks or correctness suites during measurement.
The baseline always uses per-commit fdatasync; the candidate defaults to the
process-crash policy. Raw JSON records each binary's actual policy and source
hash separately from the driver's requested environment setting.

For 10,000 x 128D, 100 transactions of 100 rows, and 1,000 exact Top-10
queries (500 filtered), medians are:

| Metric | Previous default | New default | Ratio of medians | Median paired ratio |
| --- | ---: | ---: | ---: | ---: |
| 100 acknowledged commits | 127.963 ms | 31.734 ms | 4.03x | 4.03x |
| 1,000 mixed exact queries | 27.676 ms | 27.185 ms | 1.02x | 1.02x |
| Write + query | 160.281 ms | 59.350 ms | **2.70x** | **2.72x** |
| Sum including startup, schema, recovery and shutdown | 197.196 ms | 83.193 ms | 2.37x | 2.39x |
| Startup | 5.466 ms | 5.308 ms | 1.03x | 1.06x |
| Recovery to ready | 28.968 ms | 16.708 ms | 1.73x | 1.75x |

Combined metrics are calculated per round before taking medians; adding
component medians does not reproduce their median. The combined phase sum
includes startup, CREATE TABLE, writes, initial queries, recovery to ready and
shutdown. It excludes workload/oracle preparation, result verification and
the extra post-recovery verification queries. It is not whole-process wall
clock time and does not wait for all background device writeback.

Every paired write+query ratio exceeds 2.36x for the default random-copy tie
workload. This is not just a shift of work into startup or recovery. Query
code is unchanged, and no separate query optimization is established.

Additional checks against special-data dependence:

| Workload | Metric | Previous default | New default | Ratio of medians | Median paired ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Historical zero-vector ties | 100 commits | 135.702 ms | 32.457 ms | 4.18x | 3.85x |
| Same | 1,000 mixed queries | 30.930 ms | 32.168 ms | 0.96x | 1.00x |
| Same | Write + query | 171.243 ms | 66.788 ms | 2.56x | 2.53x |
| Ordinary random eight-decimal float32 | 100 commits | 129.816 ms | 36.191 ms | 3.59x | 3.60x |

The decimal script cross-checks 50 query responses before and after recovery
per run; it does not time ordinary-float queries or supply an independent
floating-point oracle. The zero-tie query medians and paired ratios differ;
no universal query speedup is claimed.

A three-round control with `MVDB_SYNC_COMMIT=1` for both binaries measures
133.756 -> 123.543 ms writes and 162.364 -> 154.041 ms write+query (1.05x total).
This small/noisy difference does not establish an optimization in synchronized
mode. The large default-mode improvement comes from the new commit policy.

Default-workload engine peak RSS medians are 14.34 -> 14.35 MiB and both WALs
are 5,738,496 bytes. Engine RSS excludes kernel page-cache memory: the new
mode retains dirty WAL pages, so these numbers do not establish unchanged
total/cgroup memory. Linux write accounting does not certify completed device
writeback at ACK time, and no B/M-score improvement is inferred.

## Correctness and regression

`MVDB_SYNC_COMMIT=0 ./scripts/check.sh` completed with **ALL CHECKS PASSED**:

* 19 dynamic tests, exact-pruning regressions and 440 SELECT-template/error/DDL
  comparisons.
* WAL CRCs, 15 commit truncation positions, legacy WAL replay, repeated
  SIGKILL/normal restarts and tmpfs fallback.
* Immediate-ACK SIGKILL tests in both default and per-commit-sync modes. The
  critical test kills before JSON decoding, subsequent queries or WAL reads.
* LD_PRELOAD fault injection confirms no default WAL fdatasync call, actual
  fdatasync in synchronized mode, and sticky failure after either fflush or
  fdatasync EIO. Four startup checkpoint interruption points (before/after
  snapshot rename and before/after WAL truncation) recover acknowledged rows.
* Eight legacy protocol tests, 20,000 fuzz statements, a 100,000-row restart,
  ASan/UBSan engine/kernel/ownership checks, the official host handshake and
  all 20 public JSONL protocol cases.

Separate runs also passed the immediate-kill test on `/dev/shm`, a WAL larger
than 64 MiB through close-time compaction, and the WAL alignment suite with
`MVDB_SYNC_COMMIT=1`. No Docker or formal platform evaluation was run.

The two seven-round dyadic benchmarks validate 56,000 initial/recovered exact
query responses with an independent integer oracle across both binaries. The
three-round sync control adds 12,000 oracle-checked responses. The seven-round
ordinary-decimal run cross-checks 1,400 initial/recovered responses against
the previous implementation.

## Artifacts and reproduction

* [Default random-copy rounds](2026-11-16-process-crash-copies.json)
* [Historical zero-tie rounds](2026-11-16-process-crash-zeros.json)
* [Ordinary-decimal rounds](2026-11-16-process-crash-decimal.json)
* [Per-commit-sync control](2026-11-16-process-crash-strict-control.json)
* [Full regression log](2026-11-16-process-crash-check.txt)
* [Engine-only implementation diff](2026-11-16-process-crash-implementation.diff)

The baseline includes all pre-existing working-tree optimizations. Its source
can be reconstructed from this exact candidate with the saved reverse diff:

```bash
cp src/minivectordb.cpp /tmp/mvdb-before.cpp
patch --quiet -R /tmp/mvdb-before.cpp < docs/performance/2026-11-16-process-crash-implementation.diff
g++ -O2 -std=c++17 -fopenmp /tmp/mvdb-before.cpp -o /tmp/mvdb-before
make
MVDB_SYNC_COMMIT=0 python3 tests/performance_bench.py --binary /tmp/mvdb-before --binary ./minivectordb --rounds 7 --cpus 0,1 --driver-cpu 2 --json /tmp/process-copies.json
MVDB_SYNC_COMMIT=0 python3 tests/performance_bench.py --binary /tmp/mvdb-before --binary ./minivectordb --rounds 7 --cpus 0,1 --driver-cpu 2 --ties zeros --json /tmp/process-zeros.json
MVDB_SYNC_COMMIT=0 taskset -c 2 python3 tests/write_decimal_bench.py /tmp/mvdb-before ./minivectordb --rounds 7 --cpus 0,1 > /tmp/process-decimal.json
MVDB_SYNC_COMMIT=1 python3 tests/performance_bench.py --binary /tmp/mvdb-before --binary ./minivectordb --rounds 3 --cpus 0,1 --driver-cpu 2 --json /tmp/sync-control.json
MVDB_SYNC_COMMIT=0 ./scripts/check.sh
MVDB_SYNC_COMMIT=0 python3 tests/commit_crash_test.py --compaction
```

Source SHA256:

* Previous working-tree baseline: `40f4dfcfb62db852d34d2f6dec00e9748ac5eaeae09dd9a64fc79f18c717d57e`
* Candidate: `eef9d7d5981b5564e90394f9c2dc0f4b7cb4e007f8347e37d3163b2392df8435`

The final rebuilt `minivectordb` matches the measured candidate byte for byte
(SHA256 `f64ff488c88c4d4877e853fec2edc6803e87ffe0f800d5d8786b044104f9ed0c`).
