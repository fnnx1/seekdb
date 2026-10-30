# AI column integration

This branch builds on `b4d90d3a2d37abfb6b70653882c2a50282b8dd4c`, which contains
the previously evaluated AI function and File SQL implementations.

## Source attribution

The initial AI-column implementation was selectively ported from the public
Apache-2.0 contribution by GitHub user `r1ck9-2q`:

- https://github.com/oceanbase/seekdb/pull/2249
- Reference commit: `aed94823356b320ac2012d3db4464a98c2ac8f23`

Original copyright and license notices are retained. This is an adaptation of
that contribution, not a claim that its original code was authored here. The
reference's AI-function, File SQL, performance and unrelated changes were not
imported. Its reported test results are not evidence for this branch.

## Integration changes

- Merge parser tokens and system variables with the existing File SQL feature.
- Register durable AI-column metadata, hidden columns and the task catalog.
- Add DML hooks, asynchronous model evaluation, retries and strict reads.
- Propagate task write errors to the business transaction instead of losing work.
- Use timestamp values for task eligibility and retain microsecond backoff.
- Resolve each column's model endpoint independently and decode completion text
  without adding an extra JSON string layer.
- Keep publication and task cleanup in one transaction, invalidate superseded
  requests, and recheck request ownership during publication.
- Put shared SQL hook declarations in the neutral share module.
- Preserve byte-safe SQL literals and release per-operator task allocations.

## Validation (2026-10-30)

The integrated source was compiled and linked in WSL Ubuntu with Clang 17 and
the project's Release-compatible CMake configuration, using two build jobs.
All five unmodified AI-column mysqltests passed:

- `ai_column_ddl`
- `ai_column_t1_dml`
- `ai_column_worker`
- `ai_column_vector_index`
- `ai_column_retry_strict`

`unittest/sql/engine/basic/test_ai_column_runtime.py` also passed: mixed
endpoints, atomic group publication, cached successful responses during retries,
special-character output, direct result-write rejection, transaction rollback,
and bounded retries for invalid typed responses.

Regression checks passed for `ai_function_enhancement`, `file_sql_basic`, and the
existing CSV/JSONL/Parquet/COPY runtime coverage. The File SQL runner substitutes
local fixture setup for its remote SSH setup; its SQL assertions and expected
results are unchanged. AI-column and AI-function test inputs and expected outputs
are unchanged.

One discovered RETRY defect was fixed after the first test run: retrying a row
again left its older request alive. Manual retry now atomically deletes the old
generation, invalidates the result, and inserts the new task through a dedicated
transactional inner session.

Full logs and the final submission fields are retained in the task folder.

## Follow-up on platform execution failure (2026-10-30)

The platform reported an execution failure for `503c36b2293ac1f3682ed29243ab96f7a4017ea7`
without build or runtime diagnostics. Anonymous remote lookup confirmed the
submitted branch and commit. A fresh database started successfully and passed
all five original AI-column mysqltests. Clean CMake configuration with parser
caching disabled produced the same parser and inner-table sources as the
previous build. These checks did not reproduce the platform failure.

The audit found and repaired two independent defects:

- The new Share headers were listed in the public inventory but had no Bazel
  semantic owner. The package's ownership validator rejected the submitted
  revision. `ai_column_interface` now owns both headers; Observer and the actual
  SQL consumers declare the dependency. The original validator passes after
  the repair. A complete Bazel build was not run.
- Constant AI prompts caused the worker to issue an empty SQL query. The worker
  now checks that the business row still exists using `SELECT 1` before sending
  the constant prompt. Literal braces are preserved. The added runtime
  regression fails against the previous binary with a completion timeout and
  covers row deletion while a model response is in flight.

The platform's build path is unknown, so the Bazel error is not confirmed as the
cause of the reported execution failure. Local checks do not imply a platform
score or guarantee that submission infrastructure will succeed.

The repaired source compiled and linked successfully with Linux Clang 17. On a
new empty database, all five original AI-column mysqltests, the added runtime
regressions (including the constant-prompt case), AI-function enhancement, and
File SQL checks passed. The original competition test inputs and expected
results were not changed. The local service was stopped after verification.

## Queue isolation after recycled-table cleanup (2026-10-30)

The platform also rejected `f51e2fd47f54fc61789041c555385b3e44c206e8`
without diagnostics. Its anonymous remote reference was correct. All 9,197
tracked files were compared with the Linux source cache: the only differences
were an older documentation file and one extra trailing newline in three
headers. Actual compile flags matched the independent clean configuration.
Clean configuration of the exact f51 source, with parser caching disabled,
also succeeded. This is not an uncached full-project compilation claim.

A runtime defect was reproduced after using the repository CI initialization
(`init.sql` and `init_user.sql`, including `recyclebin=on`) and the `admin/test`
test identity. The five original AI-column tests passed, but their final
RETRY/DROP sequence left a task whose table had moved to the recyclebin and
whose endpoint had been removed. The next healthy request group never ran.
The separate `--case recyclebin-orphan` regression also fails on the f51
binary with the old tasks still present and the healthy result still NULL.

The worker now treats a successfully resolved absent or recycled table as
obsolete. Schema, allocation, and metadata read errors retain durable tasks;
metadata row-iteration errors are propagated instead of turning a partial
result into evidence of a dropped column. A dropped column only invalidates
its own task, preserving other columns in the group.

Errors are scoped to one request group. Endpoint resolution occurs for each
nonterminal task and follows bounded retry handling; cached successful and
terminal failed tasks can be published without an endpoint lookup. A missing
endpoint therefore cannot prevent unrelated groups or stale-task cleanup from
making progress. The worker also checks its stop flag between work items and
before new HTTP/write batches, and cannot be restarted during server shutdown.

Focused local-only regressions can be run against a disposable instance:

```bash
python unittest/sql/engine/basic/test_ai_column_runtime.py --port PORT --case recyclebin-orphan
python unittest/sql/engine/basic/test_ai_column_runtime.py --port PORT --case missing-endpoint
```

They hold the first HTTP response while another task is pending, remove the
table or endpoint, then check bounded cleanup and independent healthy work.
Observed task states are recorded as diagnostics. NULL plus strict-read
rejection is not presented as proof of a separate hidden FAILED state.
The competition SQL test inputs and expected results remain unchanged.

Validation completed with Linux Clang 17 and `make -j2` including the default
`observer` target. Both focused lifecycle cases fail on the retained f51
binary and pass on the repaired binary. The missing-endpoint case observes
task states `(1,0)`, `(3,1)`, and `(4,1)` before cleanup; its unrelated healthy
request completes with exactly one model call. A new database using CI
initialization passed all five unchanged AI-column mysqltests, all existing
and added runtime regressions, AI-function enhancement, and File SQL checks.
File SQL fixture preparation remains local; SQL and expected results are
unchanged. The final runtime binary SHA256 is
`841fa712348b35333543c7e5d7953af1bddc11aaebf19f88550c1ec1d3805db0`.

This demonstrates a real cross-test queue-stall defect. Without the competition
runner logs, it does not uniquely establish the platform execution-failure
cause or guarantee the next platform score.

## Independent compilation audit (2026-10-30)

The earlier runtime results used an existing incremental build tree. They did
not establish that the submitted revision would work when built from scratch.
The clean-build audit below supersedes that assumption.

`ob_system_variable_factory.cpp` used `OV` and `OX` without including their
definition. Its Unity group supplied the macros indirectly, but compiling the
file as its own translation unit failed with undeclared-identifier errors.
The file now includes `lib/ob_check_macros.h` explicitly. With otherwise
identical compiler arguments, the original file fails and the repaired file
passes. This issue also existed in the pre-AI-column baseline; it is not proof
of the competition runner's failure cause.

The cached binary for `5286a23de217559dafabdcf544f874692b856733` also passed
all five original AI-column tests through the repository's CI runner helpers
and default `sdb.py` startup, rather than only through a custom startup command.
A local blocked HTTP request survived stopping and restarting the database:
the persisted task completed after restart and its task row was removed.
The original SQL inputs and expected results were unchanged.

## Clean-build bootstrap registration failure (2026-10-30)

A separate Linux source tree was extracted from the submitted revision. Its
release build reused only the third-party toolchain, with no old C++/Rust
objects, parser cache or ccache. With the explicit factory include above, it
built successfully but every MySQL handshake failed with error 1193, before
any competition SQL could run. The server log identifies `sys_var_id=10755`,
`i=720`, `ret=-5044` in `ObBasicSessionInfo::load_all_sys_vars`.

The new `ai_column_strict_read` variable increased the metadata count to 721,
but `ob_system_variable_init.cpp` still initialized `var_amount` to 720.
`ObSysVariableSchema::load_default_system_variable` therefore omitted the last
variable, while login required all 721. This is a reproduced startup/login
defect; the platform itself does not expose logs confirming its failure stage.

The initialization count now derives from `ObSysVarMeta::ALL_SYS_VARS_COUNT`.
The generator emits the same definition and the factory's explicit macro
include, so regeneration preserves both repairs. Isolated regeneration and
comparison with the JSON source confirm that both generated and checked-in
metadata contain all 721 registered variables.

The runtime regression's `--case registration` checks a fresh connection,
default strict-read value, inheritance by a new session after `SET GLOBAL`,
and isolation of a session override. It fails at the handshake on the retained
clean-build binary (SHA256
`34740a474aaa1eff537de3299ba2d05138e55fbf7b4b091f392eca52742dec63`).
Run this test on a newly initialized disposable instance built from clean
source; a stale binary or an already populated schema can hide the defect.

After applying the registration fix in that same clean build tree, the release
build and link succeeded. The repaired binary SHA256 is
`38dfd507ad37eacc7315ab38be497b942b8811a77965f0091c8bd34cd32cd422`.
A separate new database accepted login in 2.05 seconds and passed all five
original AI-column mysqltests with CI initialization and admin/test identity,
the registration and existing runtime regressions, AI-function enhancement,
and File SQL mysqltest plus supplemental runtime checks. Competition SQL and
expected results were not changed; File SQL used local fixture preparation.

The user also reported a 10-second model-drop timeout on the earlier baseline
without AI columns. A local probe kept that same query timeout and exercised
80 model drops across two concurrent admin/test sessions (existing and missing
models); all returned the expected result, with a maximum latency of 0.027
seconds. The timeout was not reproduced and no speculative timeout workaround
was added. These are local results; the repaired revision still needs platform
evaluation.
