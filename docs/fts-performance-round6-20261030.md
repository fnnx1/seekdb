# FTS performance round 6 (2026-10-30)

This change builds on `53f903e0ba78c6bbf15434df7e5834c8b77edc67`. It removes a temporary document-ID pointer array and the second projection pass in presence-only fulltext queries, inlines hot comparison/copy operations, and specializes binary sort comparisons for 8/16-byte IDs. It does not alter tokenization, scoring, the benchmark or the scoring baseline.

## Final implementation

Presence-only output descriptors now directly reference each child iterator's owned ID cache. No child can refill after an output ID is retained; the shared scan frame is restored before the first refill of the next call. OFFSET does not retain aliases. The original linear sorted-run merge, equality deduplication and LIMIT semantics remain in place. Scored/unsupported paths use their previous implementation.

`ObDocIdExt::from_datum_fast` inlines the same non-NULL 8/16-byte owned `memmove`, including datum metadata and the pointer to owned storage. Only dense posting batches call it. Other sizes, NULLs and representations call the existing exported `from_datum`; object layout and the old function remain unchanged. Binary sort specialization retains byte ordering and unaligned input support; other lengths and collations use the existing code.

## Experiments rejected before release

The first candidate also shared a DAT character lookup within one character-processing cursor and used boundary probes/binary search for disjoint posting runs. Correctness checks passed, but two complete old/new/new/old sequences produced only +0.5716% combined equal-category improvement, with tokenize -1.7869%. These experiments were withdrawn. All eight reports and their combined comparison remain in the evidence directory, and their patch is retained in the desktop topic folder.

The Linux source overlay explicitly includes reversions, so omitted files cannot silently leave experimental code in the compile cache. Final source changes affect five implementation files; the 19-file overlay includes eight reverted implementation files and six new regression files. Perf stack inspection also identified folded-symbol aliases in apparent array-access hotspots; no speculative change was made based on the misleading symbol names.

## Validation

Clang17 Release compile/link with two workers passed. Binary SHA256: `2cb69287e019f821a892613c59ec72dcb989c4a1116730bb9a41963320a67744`. The final Windows and Linux build-overlay hashes match.

440 differential SQL statements in 16 groups and 20 native mysqltest suites passed. Coverage includes scalar and 1/2/7/32/128-row batches, fixed/variable binary IDs, no-primary-key tables, overlaps, OFFSET/LIMIT, prepared-statement reuse, DML/rollback, Unicode sorting, long documents and SSTable-source index builds. Further checks passed 27,665 unrounded score rows, 88 independent BMW comparisons (155,947 rows and four unchanged unsupported errors), 3,120 concurrent TOKENIZE calls, and 720 exact token/error outcomes.

The new tokenizer regression retains invalid-parameter recovery checks even though the experimental lookup was withdrawn. Its expected error is validated by native mysqltest; the simple Python SQL replay cannot interpret `--error` and was not used to suppress that failure. An initial full-score comparison used a pre-restart reference. Restarting the original binary on the same physical corpus reproduced the candidate's exact values; fresh original snapshots were then used without tolerance or rounding. Initial failures and the independent verification are retained.

## Final candidate measurement

The narrowed candidate first ran an independent old/new/new/old sequence in the reused regression runtime. One new-version run showed elevated IK/BENG samples and some query samples, producing an overall -4.8486% comparison. The tokenizer implementation had been restored byte-for-byte to the reference; that fact does not rule out indirect effects. The logs establish variability, not its cause. All four reports are retained.

A further fixed old/new/new/old sequence used four newly created runtimes, one for each run, to avoid carrying accumulated DDL/test history between versions. The unchanged official workload uses 20,000 rows, batch500, 3,000 tokenize rounds, 200 query rounds, three samples, warmup30 and fresh load on every run. Each runtime uses the same 2GiB server configuration; compilation and diagnostics are not run concurrently. A lightweight vmstat record is retained for each run. All observations are retained and SELECT1 overhead is not subtracted. The following table is specifically this fresh-runtime sequence; the previous sequence is not silently replaced. Build units are seconds; other units are milliseconds.

| Metric | Previous mean | Final candidate mean | Time improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 5.92450 | 6.05350 | -2.18% |
| build_ik_content_sec | 4.31300 | 4.03650 | 6.41% |
| build_beng_en_sec | 2.28300 | 2.23300 | 2.19% |
| tokenize_ik_avg_ms | 0.16520 | 0.14115 | 14.56% |
| tokenize_beng_avg_ms | 0.11015 | 0.10890 | 1.13% |
| query_cn_avg_ms | 0.52590 | 0.47855 | 9.00% |
| query_beng_avg_ms | 0.70155 | 0.63150 | 9.99% |
| query_mixed_avg_ms | 0.56045 | 0.51790 | 7.59% |
| query_limit_avg_ms | 0.23120 | 0.23615 | -2.14% |

In fresh runtimes, query-category improvement is 6.11%, with the three main queries improving 7.59% to9.99%; LIMIT regresses2.14%. Build-category improvement is2.14%, including IK-all regressing2.18%. Tokenize's measured7.85% improvement is not attributed to a tokenization code change: that implementation is unchanged and the samples vary. The measured equal-category mean is5.37%, but it is not a stable code-only estimate or a platform score prediction. The reused-runtime negative result remains relevant uncertainty. The public scoring baseline is unchanged.

## Platform-score discrepancy

The supplied preceding platform report had raw score86.7533; the next had75.7903. Yet all nine scoring times fell: build category10.64%, tokenize5.60%, query23.09% relative to the preceding report. With any identical positive baseline and the documented monotonic formula, all nine times falling cannot lower the score. The public checked-in baseline yields a capped100 for both reports and therefore cannot reproduce the platform scores. This does not establish which platform baseline, implementation or metric-to-score association differed, and is not proof of a full platform score.

No attempt is made to alter timing, data, SQL, score or baseline files to reconcile this discrepancy. Full marks still require actual platform evaluation. Raw reports retain the reused Linux cache's older Git metadata; the source manifest and independent binary hashes identify the binaries tested here. Complete source, tests, measurements and failed-experiment records remain in the corresponding desktop topic directory.
