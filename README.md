# MiniVectorDB

A small, single-process vector database for the OceanBase 2026 database
competition playoff question **MiniVectorDB** (`questionId=600081`).

The database reads SQL from **stdin**, writes query results and execution
status to **stdout**, writes diagnostics to **stderr**, and stores all durable
state under **`/data`**. It does not open a network listener.

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

The evaluation driver connects to the container's stdin/stdout and uses
`docker run -i` (no `-d`, no `-t`).

## Run

```bash
./minivectordb /path/to/data
# or
DATA_DIR=/data ./minivectordb
```

If no argument is given, the data directory defaults to `/data`. The directory
is created if needed. All persistent files (`snapshot.bin`, `wal.bin`) live
there.

Example session (stdin):

```sql
CREATE TABLE docs (id BIGINT PRIMARY KEY, title VARCHAR(100), emb VECTOR(4));
INSERT INTO docs VALUES (1, 'a', '[1,0,0,0]'), (2, 'b', '[0,1,0,0]');
SELECT id, title FROM docs ORDER BY L2_DISTANCE(emb, '[1,0,0,0]') LIMIT 2;
BEGIN;
INSERT INTO docs VALUES (3, 'c', '[0,0,1,0]');
ROLLBACK;
SELECT id FROM docs ORDER BY id;
```

`SELECT` output is tab-separated: one header line followed by rows. DML/DDL
statements return `Query OK, N rows affected`. Errors are returned as
`ERROR <code> (<sqlstate>): <message>` on stdout; logs go to stderr.

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

- **Storage layout:** rows are kept in memory; primary key is indexed by a
  hash map. A binary snapshot (`/data/snapshot.bin`) stores table schemas and
  live rows. A write-ahead log (`/data/wal.bin`) records committed SQL
  statements with length, sequence number and CRC32.
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
  spawn worker processes or threads, and does not listen on any port.

## Limitations

- The row store is currently in-memory. It is sized for the published small
  benchmark; a disk-backed row store is future work if the hidden tests
  exceed the memory limit.
- Approximate indexing is not implemented; exact scans satisfy the recall
  requirement but may rank lower on the large ANN benchmark.
- The SQL parser covers the documented subset; vendor-specific syntax is not
  supported.

## Tests

```bash
make
python3 tests/test_minivector.py
```

The dynamic suite covers creation, insert/select/update/delete, filtered
vector TopK, tie-breaking, transactions and rollback, restart persistence,
`SIGKILL` recovery, graceful-close rollback, error handling, `IF EXISTS`,
`CREATE VECTOR INDEX`, and a 10k-row performance smoke test.
