# Fulltext retrieval and document-local token maps

## Change

Natural-language MATCH used only as a direct Boolean WHERE predicate does not
need a BM25 score. The optimizer now recognizes that narrow case, and the
composition, statistics and token iterators support it consistently. Scores
remain enabled for score consumers, thresholds, residual predicates, Boolean
queries and implicit relevance ordering with LIMIT.

Small natural-language queries merge posting-list heads directly instead of
using the generic merge heap. Unscored unions allow up to eight streams. Scored
queries are limited to the existing simple-merger threshold of three streams:
equal document IDs contribute in the same iterator-index order as before.
They retain BM25 estimation, weights, the relevance collector and filtering.
This preserves floating-point accumulation order and minimum-match behavior.
Other modes, document-ID types and larger scored queries retain the original
implementation.

Document token maps use the no-lock hash-map policy. All five production
construction sites own a stack-local map and synchronously consume it:
TOKENIZE, fulltext index-row generation, optimizer query tokenization, and the
two DAS query-tokenization implementations. Shared dictionary containers are
unchanged.

## Profiling evidence

Linux CPU sampling of repeated English MATCH queries identified generic
merge-heap operations, document-ID comparison and per-row virtual calls as
significant costs. TOKENIZE sampling identified repeated read/write latches
in document-local token processing. These changes remove work from those
paths without caching query results, changing input data or changing parser
rules. The competition benchmark and scoring inputs are unchanged.

## Verification

The regression SQL is in the existing `fts_index` suite:

- `predicate_only_match`: scored/unscored predicates, scalar/vectorized
  execution, residual filtering, limits/offsets, Boolean mode, prepared
  statement reuse, NULL/empty inputs, DML and a 300-term TAAT query.
- `document_local_tokens`: five parsers, repeated-token frequencies, IK
  modes, parser properties, hidden primary keys and indexed updates/deletes.

The full Linux Release build completed successfully. The 44 and 26 SQL
statements in these two files were executed through a PyMySQL comparison
runner and exactly matched the saved baseline executable. This is not a claim
that the complete mysqltest suite was run. Eight concurrent sessions also
matched the baseline in 3,120 TOKENIZE calls.

On the same physical 20,000-document corpus, all 27,665 unrounded scores for
the English, mixed and content queries, explicit ranking, and implicit LIMIT
IDs matched exactly. Four query families also matched complete document-ID
sets, threshold results and their top-30 scores. Comparing an older statistics
snapshot produced small score differences; switching back to the baseline
executable on the current data reproduced the candidate results exactly.

## Local performance

Both versions used the unchanged `tools/benchmark/fts_large_bench.sh`:
20,000 rows, batches of 500, 3,000 TOKENIZE rounds, 200 query rounds, three
samples and 30 warmups, with index rebuilding enabled. The Linux server ran
in WSL Ubuntu with a 2 GiB memory limit. No compilation ran during measurement.
The baseline is the previous candidate `e6bfff547d9ca65ffd2ae59098c2dc5dc6c06fad`,
whose tree equals the previously published `f1a7995a` candidate. It is **not**
the platform's fixed scoring baseline.

| Metric | Baseline | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| IK all-fields build (s) | 6.565 | 6.771 | -3.14% |
| IK content build (s) | 5.348 | 5.248 | 1.87% |
| BENG build (s) | 3.038 | 2.938 | 3.29% |
| IK TOKENIZE (ms) | 0.1668 | 0.1570 | 5.88% |
| BENG TOKENIZE (ms) | 0.1204 | 0.1133 | 5.90% |
| Chinese MATCH (ms) | 1.3327 | 0.7350 | 44.85% |
| English MATCH (ms) | 1.9151 | 0.9783 | 48.92% |
| Mixed MATCH (ms) | 1.5139 | 0.8050 | 46.83% |
| LIMIT query (ms) | 1.8861 | 1.8506 | 1.88% |

Query hits remained 8,001 / 11,000 / 7,332 / 20. Query-category mean reduction
was 35.62%; the equally weighted build/tokenize/query mean reduction was
14.06%. Index-build changes are within a few percent and no substantial build
improvement is claimed. These measurements do not establish full marks;
the platform must evaluate the pushed commit against its own baseline.

Raw measurements are in [the baseline report](performance/fts-20261030/baseline.txt)
and [the candidate report](performance/fts-20261030/candidate.txt). The candidate
was built before committing, so its raw report correctly shows the baseline
Git HEAD with `git_dirty=1`.
