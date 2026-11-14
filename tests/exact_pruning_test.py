#!/usr/bin/env python3
"""Check exact rejection bounds against a full scan and integer-distance oracle.

The full-sort query has no bounded heap and cannot reject a candidate.  This
also checks float32 ordering for scales where a double Python oracle differs.
"""
import json
import os
import random
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("MINIVECTORDB", os.path.join(ROOT, "minivectordb"))


def literal(v):
    return "'[" + ",".join(format(x, ".9g") for x in v) + "]'"


class Database:
    def __init__(self):
        self.path = tempfile.mkdtemp(prefix="mvdb-exact-pruning-")
        env = dict(os.environ)
        env.pop("MVDB_LEGACY_STDIO", None)
        self.proc = subprocess.Popen([BIN, self.path], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)
        assert json.loads(self.proc.stdout.readline()) == {"type": "ready"}
        self.ident = 0

    def sql(self, sql):
        self.ident += 1
        request = dict(id=self.ident, op="execute", search_mode="exact", sql=sql)
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        response = json.loads(self.proc.stdout.readline())
        assert response.get("ok"), (sql[:200], response)
        return response.get("rows", [])

    def close(self):
        self.proc.stdin.write('{"id":0,"op":"shutdown"}\n')
        self.proc.stdin.flush()
        assert json.loads(self.proc.stdout.readline()).get("ok")
        self.proc.wait(timeout=10)
        assert self.proc.returncode == 0
        shutil.rmtree(self.path)


def insert(db, rows):
    for b in range(0, len(rows), 100):
        db.sql("INSERT INTO t VALUES " + ",".join(
            "(%d,%d,%s)" % (pk, group, literal(v))
            for pk, group, v in rows[b:b + 100]))


def check(db, query, count, group=None, k=10, offset=0, desc=False, oracle=None):
    where = " WHERE grp = %d" % group if group is not None else ""
    order = " ORDER BY L2_DISTANCE(v,%s)%s" % (literal(query), " DESC" if desc else "")
    prefix = "SELECT id FROM t" + where + order
    full = db.sql(prefix + " LIMIT %d" % count)
    bounded = db.sql(prefix + " LIMIT %d,%d" % (offset, k))
    assert bounded == full[offset:offset + k], (group, k, offset, desc, bounded, full[:80])
    if oracle is not None:
        candidates = [r for r in oracle if group is None or r[1] == group]
        expected = sorted(candidates, key=lambda r: (
            (-1 if desc else 1) * sum((a - b) ** 2 for a, b in zip(r[2], query)), r[0]))
        expected = [[r[0]] for r in expected[offset:offset + k]]
        assert bounded == expected, (bounded, expected)


def integer_and_mutation_cases():
    rng = random.Random(7119)
    dim, count = 128, 2048  # Both OpenMP and the quantized scan are active.
    ids = list(range(-count // 2, count // 2))
    rng.shuffle(ids)
    rows = [(pk, i % 10, [0] * dim if i % 17 == 0 else
             [rng.randint(-8, 8) for _ in range(dim)]) for i, pk in enumerate(ids)]
    db = Database()
    try:
        db.sql("CREATE TABLE t(id BIGINT PRIMARY KEY, grp BIGINT, v VECTOR(128))")
        insert(db, rows)
        for query in [[0] * dim, [3] * dim, [rng.randint(-8, 8) for _ in range(dim)]]:
            for group in [None, 3]:
                for k, offset, desc in [(1, 0, False), (10, 0, False),
                                         (17, 9, False), (64, 0, False),
                                         (65, 0, False), (10, 3, True)]:
                    check(db, query, count, group, k, offset, desc, rows)
        # Build both caches, mutate inside a transaction, then undo it.
        pk = rows[1][0]
        db.sql("BEGIN")
        db.sql("UPDATE t SET v=%s, grp=3 WHERE id=%d" % (literal([0] * dim), pk))
        check(db, [0] * dim, count, 3)
        db.sql("DELETE FROM t WHERE grp=4")
        check(db, [0] * dim, count)
        db.sql("ROLLBACK")
        check(db, [0] * dim, count, oracle=rows)
        check(db, [0] * dim, count, 4, oracle=rows)
        db.sql("DELETE FROM t WHERE grp=4")
        remaining = [r for r in rows if r[1] != 4]
        check(db, [0] * dim, count, oracle=remaining)
        db.sql("TRUNCATE TABLE t")
        assert db.sql("SELECT id FROM t ORDER BY L2_DISTANCE(v,%s) LIMIT 10" % literal([0] * dim)) == []
        replacement = [(10000 + i, i % 10, [i % 9] * dim) for i in range(512)]
        insert(db, replacement)
        check(db, [2] * dim, 512, oracle=replacement)
        check(db, [2] * dim, 512, 3, oracle=replacement)
    finally:
        db.close()


def scale_cases():
    rng = random.Random(309)
    # Non-multiples of 32 cover the scalar tail of the integer dot product.
    for dim, scale in [(16, 1.0), (17, 2.0 ** -70), (31, 1e30),
                       (32, 2.0 ** -20), (127, 1.0), (129, 1.0)]:
        rows = [(10000 - i, i % 10,
                 [scale * rng.randint(-8, 8) for _ in range(dim)]) for i in range(512)]
        if dim == 127:
            # One large coordinate weakens the bound but must not change results.
            rows[0][2][0] = 1e35
        db = Database()
        try:
            db.sql("CREATE TABLE t(id BIGINT PRIMARY KEY, grp BIGINT, v VECTOR(%d))" % dim)
            insert(db, rows)
            for query in [[0.0] * dim, [scale * 3] * dim,
                          [scale * 100] * dim]:  # Out-of-range queries fall back.
                for group in [None, 7]:
                    check(db, query, len(rows), group)
            if dim == 129:
                # Near-equal vectors around a large common center exercise
                # cancellation and distance ties after float32 conversion.
                db.sql("TRUNCATE TABLE t")
                near = [(i, i % 10, [1 + (i % 7) * 2.0 ** -22] * dim) for i in range(512)]
                insert(db, near)
                check(db, [1.0] * dim, 512)
                check(db, [1 + 2.0 ** -22] * dim, 512, 3)
        finally:
            db.close()


if __name__ == "__main__":
    integer_and_mutation_cases()
    scale_cases()
    print("PASS exact_pruning_test: exact TopK, ties, float scales, cache invalidation")
