# FTS performance round 4 (2026-10-30)

Presence-only fulltext queries previously advanced each posting individually and decoded unused BM25 payload columns. Short natural-language unions now retain per-term batch cursors, drain disjoint ordered runs, and deduplicate equal heads together. The optimizer and code generator omit term-frequency/document-length payloads only when relevance is not evaluated. Scored, Boolean, weighted, high-dimensional and unsupported-ID paths retain their existing behavior.

The batch projection borrows child-owned ID storage until the next call. It never refills a child after retaining any output, and restores the shared scan expression's reserved buffers before the next refill. This second rule prevents a storage write through an old projected pointer from overwriting an unconsumed child head. Numeric presence values use the expression's write-buffer API. LIMIT/OFFSET, partial final batches, rescans, lookup and MVCC continue through the original storage machinery.

IK arbitration now uses ordered chains with a monotonic cursor rather than a locked hash map. Active CJK trie states use a small contiguous array with stable compaction, explicitly preserving each state's dictionary pointer. Dictionary ranges use binary search only after validating strict ordering and non-overlap under the original comparator. BENG ASCII stopwords use a collision-free encoding of the original built-in list; Unicode, other charsets and unsupported inputs retain collation-aware lookup. No query-result cache or benchmark-specific term/data handling is introduced.

## Validation

Release compilation and linking passed with Clang 17 and two workers. Candidate4 binary SHA256: `6de4423ca976c1a733b8cd550f881840b4abd4928c553dc6d03cdc62a4c74918`.

- 237 differential SQL statements and 11 native mysqltest cases passed, including scalar/1/2/7/32/128-row batches, binary IDs, no-primary-key tables, DML, rollback and prepared-statement reuse.
- 27,665 full-corpus score rows, unrounded floating-point values, exact ranked IDs and LIMIT/OFFSET results matched the previous binary on the same physical corpus.
- An independent 18,000-row fixture passed 88 comparisons: 84 successful cases / 155,947 rows and four unchanged unsupported ES errors. Dense ties, threshold changes and weighted fallbacks are covered.
- 3,120 concurrent TOKENIZE calls across eight sessions passed. Another deterministic corpus of 240 strings / 720 IK smart, IK max-word and BENG results matched exactly, including raw token order and frequencies.

Early candidates exposed two projection bugs: an uninitialized numeric buffer and a shared-frame alias at batch size one. Both were corrected before release; existing regressions reproduce them. An exact-score check after rebuilding the benchmark corpus initially compared different physical/statistics snapshots. Restarting the original binary on the current corpus produced exactly the candidate's values; a fresh original snapshot was used without rounding or tolerance.

## Measurement

Reference: `d17bedf63bca97fe98d5d9f036686530c5c762de`, binary SHA256 `21c0ad220e3c054a4ab3438549b39b19a64c3fa4137e1b4409dd0e322795ae03`.

One uninterrupted old/new/new/old sequence ran the unchanged official benchmark with 20,000 rows, batch 500, 3,000 tokenize rounds, 200 query rounds, three samples, warmup 30 and fresh loading on every run. The dedicated server used the same 2 GiB configuration; no compilation or other test load ran concurrently. All four reports are retained, including outliers. Build units are seconds; query and tokenize units are milliseconds.

| Metric | Previous version mean | This version mean | Improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 6.28000 | 6.02550 | 4.05% |
| build_ik_content_sec | 4.86150 | 4.59350 | 5.51% |
| build_beng_en_sec | 2.64150 | 2.58650 | 2.08% |
| tokenize_ik_avg_ms | 0.15090 | 0.14500 | 3.91% |
| tokenize_beng_avg_ms | 0.11210 | 0.11020 | 1.69% |
| query_cn_avg_ms | 0.73760 | 0.65820 | 10.76% |
| query_beng_avg_ms | 0.97560 | 0.88625 | 9.16% |
| query_mixed_avg_ms | 0.81740 | 0.69000 | 15.59% |
| query_limit_avg_ms | 0.25215 | 0.24600 | 2.44% |

Mean category improvements versus the previous local version: build 3.88%, tokenize 2.80%, query 9.49%. This comparison does not use the platform's scoring baseline and does not predict a platform score. Timing noise and fixed client overhead remain visible in the raw reports.

An experimental encoded-key bulk-build sort was removed because it did not establish a repeatable benefit. Build profiling confirmed it was exercised. The existing encoder also has an arbitrary-binary empty/NUL composite-key edge case, so this change does not enable that path. Mini-mode DDL concurrency stays unchanged: its single-worker setting is an upstream low-memory regression fix. Reusing inverted postings to build the document-word auxiliary index could avoid repeated tokenization, but requires coordinated snapshots, persistent dependencies and recovery; no partial version is included.

Benchmark SQL, corpus generation, timer, scoring and baseline files are unchanged. Raw reports retain the reused Linux build cache's older Git HEAD `e6bfff547` and dirty=1; source hashes and independent binaries identify what was actually tested. Topic-level records retain all exploratory and rejected candidates. Platform evaluation is still required.
