# Fulltext performance: second iteration, 2026-10-30

The latest user-reported platform score is 17.225/50. This iteration starts
from the preceding submitted commit `6f3dc40b57c08a908f1bdece60e7e58c3f309ca2`; the user
did not repeat the evaluated SHA with the metrics. No full-score claim is made.

## Implementation

- Binary document-ID comparators select the equivalent byte comparison by the
  original comparison-function identity. NULL, extension sentinels and other
  comparison functions retain their original behavior. Eight-byte comparisons
  use a constant size to permit efficient compiler-generated loads.
- Stored-row sorting bypasses UTF-8 decoding only for byte-identical strings
  under the known UTF8MB4_GENERAL_CI comparator. Nonidentical strings keep the
  collation rules. Known binary comparators use byte order; LOB comparators,
  other types, NULLs and extended values stay on their original path.
- Small unweighted BMW queries can temporarily use the existing sequential
  merge after 64 pivot evaluations without additional range pruning. It runs
  at most 256 rows before reconsidering BMW. All iterators must have actual
  positions, not shallow upper-bound positions. The same score collector,
  top-k heap and tie behavior are used. Nonunit weights retain normal BMW.
- Document-local token-map nodes use a local no-lock pool, and insertion or
  frequency updates use one hash probe. Dictionary collision probes retain
  their exact order using conditional wraparound instead of modulo by the
  dictionary capacity (minimum 101). Shared dictionary synchronization stays.

No corpus, benchmark, scoring rule, parser output or cross-query result cache
was changed. Temporary diagnostic counters were used only in the Linux copy
and removed before the final build and all final measurements.

## Correctness and build

The full Linux Release target compiled and linked successfully with Clang 17,
GCC 12's C++ library and Rust 1.98.1, using two build jobs. Four native mysqltest
cases passed: predicate_only_match, document_local_tokens, retrieval_binary_ids,
and retrieval_sort_keys. Their expected outputs were recorded with the previous
binary. The old predicate-only expected file needed SQL indentation correction
to match native mysqltest output; result values were unchanged.

The independent SQL runner compared all 112 statements with the previous
binary. Coverage includes NULL, non-ASCII collation, case/accent/trailing spaces,
binary IDs, empty/sparse/dense matches, scoring, Boolean mode, LIMIT/OFFSET,
large offsets, prepared queries, scalar/vectorized execution and DML.
On the same physical 20,000-document corpus, all 27,665 full unrounded scores,
explicit and implicit ranking, score-only projections and derived LIMIT counts
matched exactly. Eight concurrent sessions passed 3,120 TOKENIZE comparisons.
This does not claim the whole seekdb test suite was executed.

## Performance evidence

The official script and parameters are unchanged: 20,000 rows, batch 500,
3,000 tokenize rounds, 200 query rounds, three samples, 30 warmups, SKIP_LOAD=0.
The WSL server used a 2 GiB memory limit. No build or profiler ran concurrently.
Final measurement order was old/candidate/candidate/old; both complete runs of
each version are averaged below. This avoids selecting only the fastest run.
Query hits remain 8,001 / 11,000 / 7,332 / 20.

| Metric | Previous mean | Candidate mean | Time reduction |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 6.56600 | 6.16700 | 6.08% |
| build_ik_content_sec | 4.99150 | 5.12500 | -2.67% |
| build_beng_en_sec | 2.88500 | 2.73700 | 5.13% |
| tokenize_ik_avg_ms | 0.15530 | 0.14990 | 3.48% |
| tokenize_beng_avg_ms | 0.10935 | 0.11280 | -3.16% |
| query_cn_avg_ms | 0.71475 | 0.73035 | -2.18% |
| query_beng_avg_ms | 0.94730 | 0.95265 | -0.56% |
| query_mixed_avg_ms | 0.79515 | 0.77830 | 2.12% |
| query_limit_avg_ms | 1.80670 | 1.60580 | 11.12% |


Category reductions: build 2.84%,
tokenize 0.16%,
query 2.62%; equal-weight mean
1.88%. LIMIT improves more clearly
than other queries. Some metrics regress or vary within host noise; no stable
tokenize improvement or proportional platform score increase is established.
This remains far short of proving the platform's full-mark requirement.

All four raw reports, the comparison JSON, platform input, and validation logs
are in [the evidence directory](performance/fts-20261030-round2/). Earlier
candidate attempts and CPU profiles remain in the topic's local validation
folders; they were not pooled with the final candidate measurements.

The raw reports show the Linux checkout's old `e6bfff547` Git metadata and
`git_dirty=1`: the separately saved old binary and newly compiled candidate
were run against that checkout's unchanged benchmark script. They do not imply
both binaries have the same source. The Windows commit is the submission;
source-file SHA-256 comparison verifies the copied Linux build inputs.
