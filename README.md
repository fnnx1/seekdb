# MiniVectorDB

A small, single-process vector database for the OceanBase 2026 database
competition playoff question **MiniVectorDB** (`questionId=600081`).

The database speaks the official **JSONL stdin/stdout protocol** documented in
[`STDIO_PROTOCOL.md`](STDIO_PROTOCOL.md): at startup it prints exactly
`{"type":"ready"}`; it then reads one JSON request per line
(`{"id":...,"op":"execute","sql":"...","search_mode":"exact"}`) and writes one
JSON response per line (queries return `columns`/`rows`, DML returns
`affected_rows`, errors return `ok:false` and a non-empty `error.code`).
`{"id":...,"op":"shutdown"}` rolls back uncommitted work, persists, replies and
exits 0. Diagnostics go to **stderr**; all durable state is under **`/data`**.
It does not open a network listener.

## Build

Requirements: Linux x86_64, a C++17 compiler, GNU make.

```bash
make
```

The resulting binary is `./minivectordb`.

Docker:

```bash
docker build -t minivectordb .
mkdir -p data
docker run -i --rm -v "$PWD/data:/data" minivectordb
```

The evaluation driver starts the container with `docker run -i` (no `-d`, no
`-t`), waits for `{"type":"ready"}`, and then exchanges JSON lines.

Run the official protocol checker against the built image:

```bash
python3 protocol_check/run.py --image minivectordb --handshake-only \
  --output "protocol-results/handshake-$(date +%Y%m%d-%H%M%S)"
python3 protocol_check/run.py --image minivectordb \
  --output "protocol-results/public-$(date +%Y%m%d-%H%M%S)"
```

## Run

```bash
./minivectordb /path/to/data
# or
VECTORDB_DATA_DIR=/path/to/data ./minivectordb
```

If no argument is given, the data directory defaults to `/data`. The directory
is created if needed. All persistent files (`snapshot.bin`, `wal.bin`) live
there.

Example JSONL session:

```bash
./minivectordb /tmp/mvdb-data <<'EOF'
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, title VARCHAR(100), emb VECTOR(4));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1, 'a', '[1,0,0,0]'), (2, 'b', '[0,1,0,0]');","search_mode":"exact"}
{"id":3,"op":"execute","sql":"SELECT id, title FROM docs ORDER BY L2_DISTANCE(emb, '[1,0,0,0]') LIMIT 2;","search_mode":"exact"}
{"id":4,"op":"shutdown"}
EOF
```

The response for the query is
`{"id":3,"ok":true,"columns":["id","title"],"rows":[[1,"a"],[2,"b"]]}`.

### Plain-SQL regression mode

The original plain-SQL REPL is still available for the local legacy tests by
setting `MVDB_LEGACY_STDIO=1`; it is **not** the submission protocol:

```bash
MVDB_LEGACY_STDIO=1 ./minivectordb /path/to/data
```

In that mode `SELECT` output is tab-separated and DML/DDL return
`Query OK, N rows affected`; errors are `ERROR <code> (<sqlstate>): <message>`
on stdout.

## SQL subset

- `CREATE TABLE name (col type, ...)`, `DROP TABLE [IF EXISTS] name`
- Types: `BIGINT` primary key, `VARCHAR(n)`, `VECTOR(n)` (one vector column per table)
- `INSERT INTO t [(cols)] VALUES (...), (...)` (multi-row)
- `SELECT * | exprs FROM t [WHERE expr] [ORDER BY ...] [LIMIT n]`
  - expressions: columns, `L2_DISTANCE(vector_col, '[v1,...]')`, aliases
  - predicates: `=`, `<>`/`!=`, `<`, `<=`, `>`, `>=`, `AND`, `OR`, parentheses
  - filtered TopK is computed inside the filtered set
  - ties in `ORDER BY` are broken by primary key ascending
- `UPDATE t SET col = expr, ... [WHERE ...]` (primary key updates are rejected)
- `DELETE FROM t [WHERE ...]`
- `BEGIN` / `START TRANSACTION`, `COMMIT`, `ROLLBACK`
- `CREATE VECTOR INDEX name ON t(v)` (accepted; exact search is always used)

Not implemented (as allowed by the problem): NULL, JOIN, aggregates,
subqueries, ALTER TABLE, foreign keys, nested transactions, primary-key
updates, and approximate-index-only semantics. Exact search is always
available, so exact results are returned even after `CREATE VECTOR INDEX`.

## Design

- **Storage layout:** row metadata (primary key, strings and vector
  references) is kept in memory; vector payloads live in a per-table
  file-backed store. The first 64 MiB of a vector store is `mmap`-ed for fast
  scans; larger stores switch to `pread`/`pwrite`, so the vector payload does
  not inflate anonymous RSS. A binary snapshot (`/data/snapshot.bin`) stores
  schemas and live rows, and a write-ahead log (`/data/wal.bin`) records
  committed SQL statements with length, sequence number and CRC32. The vector
  file is a derived cache: the snapshot/WAL are the source of truth and the
  cache is rebuilt on startup.
- **Vector search:** exact L2 scan over the filtered row set. TopK is produced
  by sorting on distance and then primary key. `CREATE VECTOR INDEX` is
  accepted as a no-op because exact search is available; this guarantees
  Recall@10 = 1.0.
- **Transactions:** at most one explicit transaction. Changes are applied
  immediately and an undo log restores the previous state on `ROLLBACK`.
  Statements are validated before any change, so a failed multi-row statement
  leaves no partial effect. `COMMIT` appends all statements of the transaction
  to the WAL and `fsync`s before reporting success.
- **Recovery:** on startup the snapshot is loaded and complete WAL
  transaction groups with sequence greater than the snapshot's sequence are
  replayed. A trailing incomplete record is ignored. Auto-commit statements are
  logged before being applied. Normal close rolls back an open transaction,
  writes a snapshot and truncates the WAL. `SIGTERM`/`SIGINT` trigger the same
  graceful close; `SIGKILL` is recovered from the WAL.
- **Resource use:** the process uses only the C++ standard library, does not
  spawn worker processes or threads, and does not listen on any port. On this
  machine, 1,000,000 rows x 128 dimensions peaks at about 370 MiB RSS
  (insert and restart) instead of keeping 512 MiB of vector payloads on the
  heap; the snapshot is about 526 MiB on disk.

## Limitations

- Row metadata and the snapshot as a whole are still in memory/page cache;
  the vector payload is off-heap. A dataset whose *row metadata* exceeds the
  memory limit would still need a fully paged row store.
- The vector file is append-only between snapshots; repeated updates can grow
  it until the next restart/compaction.
- The snapshot duplicates the vector payload once on disk; a very large
  dataset needs disk for both the snapshot and the transient vector file.
- Approximate indexing is not implemented; exact scans satisfy the recall
  requirement but may rank lower on the large ANN benchmark.
- The SQL parser covers the documented subset; vendor-specific syntax is not
  supported.

## Tests

```bash
make
CMD='["'"$PWD"'/minivectordb","{data_dir}"]'
python3 protocol_check/run.py --command "$CMD" --handshake-only \
  --output "protocol-results/handshake-$(date +%Y%m%d-%H%M%S)"
python3 protocol_check/run.py --command "$CMD" \
  --output "protocol-results/public-$(date +%Y%m%d-%H%M%S)"
```

The official checker runs the 20 public cases over the JSONL protocol,
including `--restart`, `--mode ann`, error responses and transaction/shutdown
rollback.

The legacy plain-SQL regression suite is still run with
`MVDB_LEGACY_STDIO=1`:

```bash
export MVDB_LEGACY_STDIO=1
python3 tests/test_minivector.py
python3 tests/protocol_test.py
python3 tests/fuzz_smoke.py
python3 tests/large_data_test.py
```

`./scripts/check.sh` runs the clean build, the legacy suite, ASan/UBSan, the
official protocol handshake and all 20 public cases on the host; with
`RUN_DOCKER=1` it repeats the handshake and public cases against the built
image. The dynamic suite covers creation, insert/select/update/delete,
filtered vector TopK, tie-breaking, transactions and rollback, restart
persistence, `SIGKILL` recovery, graceful-close rollback, error handling,
`IF EXISTS`, `CREATE VECTOR INDEX`, and a 10k-row performance smoke test. The
separate `tests/large_data_test.py` loads 100k x 128D, checks exact TopK,
restarts and records peak RSS.
