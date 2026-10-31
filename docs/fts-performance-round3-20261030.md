# FTS performance round 3 (2026-10-30)

Counting a limited fulltext view previously calculated BM25/top-k ordering and fetched unused base-table columns. A narrowly guarded rewrite now omits implicit ordering and prunes the view projection when its only consumer is a plain scalar COUNT(*). It retains LIMIT/OFFSET and exact matching. It does not rewrite explicit ordering, COUNT(column), outer filters/joins/grouping, complex projections, boolean/weighted MATCH or score predicates. Direct ranked result queries retain the original scores and tie order.

The remaining changes reduce repeated work: ASCII utf8mb4_general_ci comparison with the original case/padding rules and Unicode fallback; stable sort/merge array access; bounded adaptive BMW sequential windows. A rising top-k threshold immediately returns traversal to BMW. No result cache, benchmark-specific tokens, corpus rules, estimated counts or score adjustments were introduced.

## Validation

Release compilation/linking passed (Clang 17, two workers). Final binary SHA256: `21c0ad220e3c054a4ab3438549b39b19a64c3fa4137e1b4409dd0e322795ae03`.

- 165 differential SQL statements and seven native mysqltest cases passed.
- 27,665 full-corpus scores, exact floating-point values, ranked IDs and LIMIT/OFFSET results matched the original binary on the same physical corpus.
- An independent 18,000-row fixture passed 88 comparisons: 84 successful cases / 155,947 rows and four pre-existing unsupported ES errors. This covers sparse/dense regions, ties, threshold transitions, 8/9 dimensions and weighted fallbacks.
- Eight sessions completed 3,120 concurrent TOKENIZE comparisons; dynamic row parameters, nested calls, error recovery and Unicode inputs also passed.
- EXPLAIN confirms COUNT/LIMIT has `calc_relevance=false` and `is_index_back=false`; direct LIMIT and COUNT(column) retain ranking.

The first score snapshot changed after a server restart in BOTH the original and candidate binaries. Exact comparisons were repeated against the restarted original; no tolerance or rounding was used. The ASCII expected file only needed CRLF-to-LF normalization, verified byte-for-byte. An initial rewrite-only candidate regressed LIMIT because it retained unused ID lookup; it was rejected and corrected with the shared projection-pruning utility before this final candidate.

## Paired measurement

Reference commit: `635af5e17e1c66d9b20f3b59d3a695913c13a14e`. Original binary SHA256: `253af5d2e7f83c54765f0d7684583e6ed01744ce6551f7546d440f62076bce6e`.

One old/new/new/old sequence ran the UNMODIFIED official script, each with ROWS=20000, BATCH=500, ROUNDS=3000, QUERY_ROUNDS=200, SAMPLES=3, WARMUP=30, SKIP_LOAD=0. No compilation ran concurrently. All four reports and their binary hashes are retained. Earlier candidates had high-variance tokenize samples and no established tokenization gain. Their tokenizer and dictionary modifications were reverted, and the final source was rebuilt and revalidated before this sequence. All four runs contribute to the means below. Build units are seconds; tokenize/query units are milliseconds.

The last original-binary run slowed down across almost all metrics (IK build 10.130s versus 6.369s in the first original run). Therefore the overall and per-category means below are descriptive measurements, not isolated estimates of code speedup. In particular, tokenizer source is unchanged and its apparent gain cannot be credited to this patch. The COUNT/LIMIT reduction is much larger than this variation and consistently reproduced across the candidate runs. Platform measurement is needed to establish the remaining benefits.

| Metric | Original mean | Candidate mean | Improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 8.24950 | 6.06650 | 26.46% |
| build_ik_content_sec | 5.09650 | 4.59350 | 9.87% |
| build_beng_en_sec | 3.40300 | 2.48350 | 27.02% |
| tokenize_ik_avg_ms | 0.17555 | 0.15560 | 11.36% |
| tokenize_beng_avg_ms | 0.12520 | 0.10845 | 13.38% |
| query_cn_avg_ms | 0.84220 | 0.72920 | 13.42% |
| query_beng_avg_ms | 1.06160 | 0.95225 | 10.30% |
| query_mixed_avg_ms | 0.90930 | 0.78650 | 13.50% |
| query_limit_avg_ms | 1.86330 | 0.23995 | 87.12% |

Category mean improvements versus the previous local candidate: build 21.12%, tokenize 12.37%, query 31.09%. Their mean is 21.53%. These are NOT a prediction of the competition score. Timing variation and individual regressions remain visible; tokenization gains are not established. The public baseline JSON differs from the baseline implied by the platform reports, so this document does not claim a full score.

The raw benchmark reports show the reused Linux build directory's older Git HEAD `e6bfff547` and dirty=1. Actual source files were verified using SHA256 manifests and the separately retained binary for each run. Those raw report fields were preserved rather than rewritten to impersonate a clean checkout.

All supplied benchmark SQL, corpus construction, timing, scoring and baseline files remain unchanged. Local topic-level progress records retain the exploratory and rejected runs as well as the final evidence. Platform evaluation of this branch is still required.
