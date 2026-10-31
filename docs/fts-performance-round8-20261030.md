# FTS performance round 8 (2026-10-30)

Baseline: `89af6b8041c4e81f941f2edd48962f7a89ada89a`. Final evaluation candidate: candidate3.
The user supplied mean improvement 0.3966546536, raw score 79.3309307160, approximately
39.6655/50. No SHA accompanied that report; association with the last delivered commit
is provisional. This round does not establish full marks or a stable overall speedup.

## Implementation

The IK DAT character map now hashes one-to-four-byte character keys by safe constant-size
copies and a length-mixed 32-bit avalanche. It uses the same hash for insertion, lookup
and compaction. Full length/byte equality and bounded collision probing are retained.
Other key lengths use the original hash. Maps are constructed and read within a process;
dictionary codes, dictionary content, persisted indexes and stopword semantics are unchanged.

The lookup accepts the existing character view directly instead of copying a temporary
ObFTSingleWord. The old entry point remains available. No mutable shared query state,
result cache, altered workload, data-dependent benchmark shortcut or resource increase
is introduced. Only two implementation files differ from the baseline.

The supplementary test is in `unittest/storage/fts/test_dictionary_tokenize.py`.
Run it once against the baseline with `--output baseline.json`, then against the candidate
with `--reference baseline.json --output candidate.json`; host, port and database are options.
It requires PyMySQL and an existing test database. It creates and removes only its own
uniquely named fixture. It compares 2,400 deterministic texts in IK smart, IK max_word and
BENG modes, alternating UTF-8 collations, with exact JSON order/frequencies and errors.
Its corpus samples dictionary ranges, long prefixes, parser batch boundaries, embedded NUL,
empty strings, supplementary Unicode, mixed scripts and punctuation.

## Validation

Release compilation/linking passed with two workers. Binary SHA256:
`d467d863faad372f86debbd9df2fbfad427bc869569a05c6f81321cadd7d9e7e`.
The complete 6,669-file source manifest matches both the Windows candidate and actual
Linux compilation sources. The five-entry compilation overlay also includes the explicit
sort reversion and two experimental sorting tests, archived in the desktop topic folder.
The new Python test was added after compilation and ran directly from the Windows checkout;
it is not part of the compiled server. Official benchmark/scoring files and compiler options
were not changed.

Passed: 17 groups/463 differential SQL statements, 21 native mysqltest suites,
18 complete result sets/27,851 rows (including 27,665 unrounded scored rows),
88 BMW queries/155,947 rows/four pre-existing errors, 3,120 concurrent TOKENIZE calls,
720 existing exact TOKENIZE outcomes and 7,200 expanded outcomes. An empty runtime with
eight concurrent cold-loader sessions passed 328 checks against the previous reference.
No result sorting, score tolerance or error suppression was added. Independent audit
summaries and source/binary hashes are included alongside the reports.

## Rejected experiments

Candidate1 grouped repeated leading sort keys before sorting subsequent keys. Correctness
passed, but its complete fresh old/new/new/old sequence measured overall -26.4568%, including
a slow LIMIT sample. Candidate2 forced inlining of the shared comparator and measured only
+0.1994% overall, with build-category -1.9536%. Both were reverted; all eight reports and
the candidate2 patch are retained. The archived extra sorting test still ran against the
final candidate; it is not added to the final functional test suite.

Candidate2's four timed runs and plan captures finished successfully. Its cleanup trap then
tried to capture a plan after stopping the service and got a refused connection. The trap
was fixed. This was a post-measurement harness failure, not a failed timed run. SQL audit
was unavailable; EXPLAIN is retained, but no unavailable audit data is claimed as evidence.

## All eight final-candidate measurements

Two predetermined old/new/new/old sequences each used four empty runtimes with identical
`memory_budget=2G` settings. The unchanged official workload used 20,000 rows, batch500,
3,000 tokenization rounds, 200 query rounds, three samples and warmup30. Compilation and
profiling did not run during timing. Every sample, raw report and vmstat record is retained;
no SELECT1 subtraction or outlier removal is applied. These are local comparisons against
the previous candidate, not the competition's scoring baseline.

The first sequence measured +1.6262% overall; the confirmation measured -1.1591%.
Across all eight, build-category +3.2570%, tokenize +0.1966%, query -2.2586%,
equal-category mean +0.3983%. The individual means follow (build: seconds; others: milliseconds).

| Metric | Baseline | Candidate | Time improvement |
| --- | ---: | ---: | ---: |
| build_ik_all_sec | 5.915250 | 5.883750 | 0.53% |
| build_ik_content_sec | 4.275500 | 4.200500 | 1.75% |
| build_beng_en_sec | 2.441750 | 2.259000 | 7.48% |
| tokenize_ik_avg_ms | 0.143400 | 0.142350 | 0.73% |
| tokenize_beng_avg_ms | 0.110600 | 0.110975 | -0.34% |
| query_cn_avg_ms | 0.416175 | 0.423250 | -1.70% |
| query_beng_avg_ms | 0.545750 | 0.562350 | -3.04% |
| query_mixed_avg_ms | 0.465500 | 0.484925 | -4.17% |
| query_limit_avg_ms | 0.229825 | 0.230100 | -0.12% |

IK single-query tokenization improved 0.46% and 1.00% in the two sequences, 0.73% pooled.
CN/EN/mixed queries regressed 1.70%/3.04%/4.17% pooled. BENG index construction varies
despite its tokenizer code being unchanged. These movements cannot all be attributed to
the changed character map, and the available evidence does not establish their cause.

Separate 15-second CPU samples showed map lookup plus DAT matching at 10.58% of sampled
CPU for the baseline and 5.66% for the candidate. Sample percentages are not elapsed-time
improvements. A separate batch experiment, after each confirmation run's timed benchmark,
processed all 2,400 corpus texts with three warmups and nine measurements per mode. All
output-byte totals matched, while exact output equality was checked separately above.
Batch IK smart measured +0.3494%, IK max_word -0.4408%, BENG -8.9857%. This does not show
a stable broad tokenizer speedup. All 108 measured batch samples are preserved.

The result is a correctness-validated, narrowly scoped candidate for platform evaluation.
It is not evidence of full marks or a guaranteed increase over the previous submission.
Local raw reports retain the reused compilation cache's older Git metadata; source manifests
and binary hashes identify the actual binaries. Earlier `memory_limit` compatibility flags
must not be interpreted as proof of an effective 2GiB budget; this round used `memory_budget`
equally for both binaries.

Only trailing display padding in the copied CPU and validation transcripts is normalized;
the original logs remain in the topic folder. Benchmark raw reports are copied unchanged.

All test services were stopped. Only the 17 explicitly verified temporary runtimes from
this round were removed; reports, binaries, source, and log tails remain in the desktop
topic folder. No other user database or service was removed.
