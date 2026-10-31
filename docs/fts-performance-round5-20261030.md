# FTS performance round 5 (2026-10-30)

This candidate builds on `55378155e0fa04fe33d77e68482eec548a73eae3`, whose reported platform raw score was 86.7533. Fulltext queries still return the original matches, ranking and floating-point scores; this change reduces work in storage decoding, ID merging, dictionary lookup and bulk index construction.

## Changes and invariants

- Fixed-width 8/16-byte document IDs use constant-size copies and comparisons. Batch skip state is checked once, and the unique minimum posting head is consumed without comparing it again. Other widths, NULL/ext datums and scored paths retain their existing behavior.
- Small string-difference decoder batches build a zero-initialized common-byte template once per batch. Per-row varying bytes and hex decoding remain unchanged; large strings and single-row batches use the original path.
- Completed dictionaries compact their character maps according to distinct characters rather than trie node count. Character codes and DAT entries remain identical; offsets move before publication. Optional scratch allocation failure retains the original dictionary. Probe loops are bounded.
- Loader cache hits use the existing shared lock and acquire the handle reference under that lock. A miss releases it before taking the exclusive lock and rechecking initialization and the map. There is no lock upgrade or unprotected pointer handoff.
- Eligible four-column FTS DDL scans output a batch of words from one document at a time. They never advance the document or reuse its word arena while that batch is live. Sampling and the underlying DAS document/lookup scans keep their scalar contract; filters and unsupported projections keep the original path. Rescan resets the word cache.
- Sort comparison caches immutable metadata and the first eight comparator classifications without allocating. Classification matches the original function pointer, retaining LOB/type/collation handling, NULL/ext fallback, additional keys and encoded-sort fallback.

## Correctness and implementation failure caught locally

Clang 17 Release compilation/linking with two workers passed. Final binary SHA256: `ab55406d96e2316662a4d52ca1b06fdb0d1ace557fc1cbf4421acd8d09142782`. All 26 modified source/test files match the build overlay, including 14 implementation files.

378 differential SQL statements across 14 groups and 17 native mysqltest suites passed. Coverage includes binary IDs, batch boundaries, NULL/empty/long documents, multiple columns, heaps, partitions, DML/rebuild, Unicode sorting and more than eight ordering keys. There are also 27,665 unrounded score rows; 88 independent BMW comparisons (155,947 rows and four unchanged unsupported errors); 3,120 concurrent TOKENIZE calls; and 720 exact token/error outputs. Candidate2's isolated cold loader check passed 328 outcomes across eight sessions; those loader/dictionary sources are identical in the final candidate.

Candidate1 initially passed small-data tests, but a later diagnostic hung on an SSTable source. Its positive storage batch size was incompatible with the scalar `get_next_row()` caller: `OB_PUSHDOWN_STATUS_CHANGED` repeatedly requested a batch-mode switch without progress. It was rejected. The final implementation keeps DAS scan, lookup and attached-scan pushdown batch sizes zero while batching only expanded output. A new CTAS plus primary-key rebuild regression reproduces the storage path and passes with a bounded timeout.

A separate post-measurement GDB diagnostic observed an actual returned batch of **16 rows**, detached cleanly, and verified 64 matching documents. It completed in approximately 7.1 seconds and did not modify the binary. Earlier DEBUG-log attempts could not observe batches because Release excludes those logging calls; their lack of evidence was not treated as proof. Diagnostic timings are not performance measurements.

The regression suite does not establish every physical sort mode or force external spilling. Score snapshots use the same physical corpus on both binaries; the old binary recorded a new exact oracle after the corpus was rebuilt. No tolerance or rounding was added to the full-corpus comparison.

## Measurement

The final serial old/new/new/old sequence used the unchanged official benchmark: 20,000 rows, batch 500, 3,000 tokenize rounds, 200 query rounds, three samples, warmup 30 and fresh load each run. The same dedicated 2 GiB runtime was restarted between binaries. No compilation, debugger or other test load ran concurrently. All samples are retained; SELECT 1 overhead is not subtracted. Build units are seconds; other units are milliseconds.

| Metric | Previous version mean | Final candidate mean | Time improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 6.03050 | 5.36650 | 11.01% |
| build_ik_content_sec | 4.90750 | 4.09600 | 16.54% |
| build_beng_en_sec | 2.53450 | 2.28750 | 9.75% |
| tokenize_ik_avg_ms | 0.15085 | 0.14700 | 2.55% |
| tokenize_beng_avg_ms | 0.11180 | 0.11025 | 1.39% |
| query_cn_avg_ms | 0.66520 | 0.53460 | 19.63% |
| query_beng_avg_ms | 0.86630 | 0.70240 | 18.92% |
| query_mixed_avg_ms | 0.70955 | 0.56775 | 19.98% |
| query_limit_avg_ms | 0.25955 | 0.23950 | 7.72% |

Mean category improvements versus the previous local version: **build 12.43%, tokenize 1.97%, query 16.57%**. The equal-category mean is 10.32%. This is a local version comparison, not a predicted platform score or confirmation of full marks. Tokenize improvements are small relative to observed run variability.

The intermediate candidate2 sequence is also retained. Its English query sample means were 0.7146/4.9816/0.7013 ms in new1, producing a negative query-category mean; the available logs do not establish the cause. No sample was removed. The final candidate has a subsequent code change to sort metadata caching and its own complete sequence, not a replacement of an individual slow observation.

The official SQL, corpus, timer, scoring, baseline and build flags are unchanged. Reports retain the reused Linux cache's older Git metadata (`e6bfff547`, dirty=1); the independent binary hashes and normalized source manifest identify the tested candidate. Raw final and intermediate reports, source manifest and validation summary are in `docs/performance/fts-20261030-round5/`. All exploratory logs remain in the desktop topic's `validation-round5/`. Platform evaluation remains necessary.
