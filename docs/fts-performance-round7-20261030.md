# FTS performance round 7 (2026-10-30)

Baseline: `7fe1edb7bf6a1ce209facae9770e1fd182fd538d`. Final candidate: candidate2. The supplied platform report has mean improvement 0.4037451663, raw score 80.7490333, or 40.3745166 on this task's 50-point scale. The user did not provide its SHA; association with the last delivered commit is provisional.

## Implementation

Two- and three-stream presence-only, vectorized natural-language MATCH now use a fixed-fan-in merge. It compares each head once, advances equal heads with a mask and refills only at batch boundaries. It preserves ordered deduplication, OFFSET and LIMIT, and the rule that no child cache can refill while the returned output borrows its IDs. The shared scan frame is restored before the first refill. Other fan-ins, scored queries and unsupported ID types keep their existing paths. A new inline read-only datum accessor preserves the old exported accessor and object layout.

Short fixed-width, non-hex StringDiff cells prepare common bytes and patch source/destination offsets once per batch. Each non-null row is initialized and patched in one pass, using constant-size copies for short runs. Original buffer stride, allocation owner and NULL behavior are preserved. Hex-packed, long and variable-width cells use their existing paths. There is no storage-format change or result cache.

Only four implementation files and two regression files differ from the baseline. The eight-entry build overlay includes two explicit dictionary reversions. Windows sources and the actual Linux build sources match the candidate2 source manifest. Official SQL, corpus, scoring, baseline, compiler options and server resource settings are unchanged.

## Rejected experiment

Candidate1 also removed temporary DAT character copying. It passed correctness checks, but its fresh-instance old/new/new/old comparison showed tokenize-category -0.2539%, with no stable tokenizer improvement. Both dictionary files were restored byte-for-byte. Candidate1 reports and the full patch remain in the desktop topic folder; the four reports are also preserved here.

## Validation

Clang17 Release compilation/linking with two workers passed. Final binary SHA256: `9aeee65e28953cb05b4827d021eaf46ff3560609175400d6f91a5b6be114b024`.

445 differential SQL statements in 16 groups and 20 native mysqltest suites passed. New cases cover three-stream merges with batch sizes 1/2/7, term reordering, an empty stream, and OFFSET/LIMIT beyond or near the end. Existing suites cover binary and implicit IDs, NULLs, multiple short/long cell widths, hex-packed and general decoding, DML/rollback, prepared reuse, SSTable sources and scalar fallback.

18 full-result comparisons matched 27,851 rows exactly, including 27,665 rows in the three complete scored result sets. Independent BMW comparisons matched 88 queries, 155,947 rows and four pre-existing unsupported errors. 3,120 concurrent TOKENIZE calls and all 720 original JSON/error outcomes matched. No rounding, score tolerance or result reordering was introduced. Timings are excluded only from correctness equality; every timing sample is retained in performance reports.

A separate bounded GDB diagnostic hit `ob_string_diff_decoder.cpp:193` in the new branch with `row_cap=256`; its read-only query returned 11,000 hits and the debugger explicitly detached. The first diagnostic hit the same branch but could not inspect an additional optimized/debug-type header field, so its summary remains unsuccessful. The second confirms only the actual branch and batch size, not those unavailable extra fields. Neither diagnostic is part of benchmark timing.

## Measurement and uncertainty

Each sequence uses the unchanged full workload: 20,000 rows, batch500, 3,000 tokenize rounds, 200 query rounds, three samples, warmup30, fresh load. Each run starts its own empty runtime with identical 2GiB server settings. Order is fixed old/new/new/old; no compilation or debugger runs concurrently. All raw samples and vmstat records are included; no sample is removed and SELECT1 overhead is not subtracted.

The first final-candidate sequence (`round7-c2-fresh`) produced overall -3.2302%. Its new1 English query samples were 0.5545, 0.5599 and 5.2211ms. The slow third sample is retained; the available evidence does not establish its cause. CN/mixed improved 13.43%/12.29%, but English's mean regressed 103.36% in that sequence.

A further complete, predetermined four-run sequence (`round7-c2-confirm`) produced the following. Build units are seconds; tokenize/query units are milliseconds.

| Metric | Previous mean | Candidate mean | Time improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 5.86800 | 6.34600 | -8.15% |
| build_ik_content_sec | 4.11550 | 4.08650 | 0.70% |
| build_beng_en_sec | 2.82550 | 2.28200 | 19.24% |
| tokenize_ik_avg_ms | 0.14790 | 0.14750 | 0.27% |
| tokenize_beng_avg_ms | 0.11180 | 0.10990 | 1.70% |
| query_cn_avg_ms | 0.49850 | 0.43375 | 12.99% |
| query_beng_avg_ms | 0.64560 | 0.56945 | 11.80% |
| query_mixed_avg_ms | 0.52255 | 0.48045 | 8.06% |
| query_limit_avg_ms | 0.22965 | 0.24250 | -5.60% |

That confirmation sequence measures build-category +3.9314%, tokenize +0.9850%, query +6.8113%, equal-category mean +3.9092%. Tokenization code is unchanged, so its small movement is not attributed to a tokenizer optimization. LIMIT regresses 5.60% in this sequence.

### All eight final-candidate runs, without filtering

| Metric | Previous mean | Candidate mean | Time improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 6.05925 | 6.00625 | 0.87% |
| build_ik_content_sec | 4.36650 | 4.03725 | 7.54% |
| build_beng_en_sec | 2.55550 | 2.25700 | 11.68% |
| tokenize_ik_avg_ms | 0.14355 | 0.14438 | -0.57% |
| tokenize_beng_avg_ms | 0.10990 | 0.11015 | -0.23% |
| query_cn_avg_ms | 0.49575 | 0.43028 | 13.21% |
| query_beng_avg_ms | 0.64970 | 0.94950 | -46.14% |
| query_mixed_avg_ms | 0.52545 | 0.47193 | 10.19% |
| query_limit_avg_ms | 0.24575 | 0.23580 | 4.05% |

Across all eight, build-category +6.6986%, tokenize -0.4011%, query -4.6754%, equal-category mean +0.5407%. The English outlier still produces a 46.14% aggregate regression. CN and mixed improve 13.21% and 10.19% across both sequences, while aggregate LIMIT improves 4.05%; build timing varies substantially. This is a correctness-validated candidate for platform evaluation, not evidence of a stable overall speedup, a guaranteed score increase or full marks.

The preceding platform reports could not be reconciled with one fixed positive baseline; the public checked-in baseline does not reproduce platform scores. No scoring input is modified to address that discrepancy. Raw local reports retain the reused Linux cache's older Git metadata; independent binary hashes and source manifests identify what was tested.

All FTS validation/build/debug/cold services are inactive. Only the twelve temporary runtimes created for these three sequences were removed after path and process checks; reports, resource samples and log tails remain. Other databases and user services were untouched.
