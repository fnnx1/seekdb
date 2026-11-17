#!/usr/bin/env python3
"""Independent exact regressions for VARCHAR equality indexes and grouped codes.

Set MINIVECTORDB to the candidate. Optionally set MVDB_BASELINE to also run
identical operations against an older binary and compare all responses.
Data uses small integer coordinates; squared distances are exactly represented
in float32. The independent Python oracle does not read engine internals.
"""
import json
import math
import os
from pathlib import Path
import random
import selectors
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get('MINIVECTORDB', str(ROOT / 'minivectordb'))
BASELINE = os.environ.get('MVDB_BASELINE')
D = 128
N = 2057
GROUPS = ['', 'a', 'aa', 'a\0x', 'a\0y', '汉字', 'é', 'e',
          "quote's", 'slash\\group', '\nline\tbreak', 'long_' + 'z' * 80,
          '5', '-2', '0', '🙂']
QUERIES = []
for seed in (1, 17, 91):
    rng = random.Random(seed)
    QUERIES.append(tuple(rng.randrange(-4, 5) for _ in range(D)))
QUERIES += [tuple([0] * D)]


def quote(value):
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'").replace('\0', '\\0').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r') + "'"


def vector(values):
    return "'[" + ','.join(map(str, values)) + "]'"


class Engine:
    def __init__(self, binary, directory):
        self.binary = str(Path(binary).resolve())
        self.directory = directory
        self.ident = 0
        self.start()

    def start(self):
        env = dict(os.environ)
        env.pop('MVDB_LEGACY_STDIO', None)
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([self.binary, str(self.directory)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.err, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        assert self.read() == {'type': 'ready'}

    def read(self):
        assert self.selector.select(30), 'response timeout'
        line = self.proc.stdout.readline()
        if not line:
            self.err.seek(0)
            raise AssertionError(self.err.read().decode(errors='replace'))
        return json.loads(line)

    def request(self, sql=None, op='execute', ok=True):
        self.ident += 1
        payload = {'id': self.ident, 'op': op}
        if sql is not None:
            payload.update(sql=sql, search_mode='exact')
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode() + b'\n')
        self.proc.stdin.flush()
        response = self.read()
        assert response.get('id') == self.ident, response
        assert response.get('ok') is ok, (sql and sql[:150], response)
        if not ok:
            assert response.get('error', {}).get('code'), response
        return response

    def close_handles(self):
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.err.close()

    def restart(self, kill=False):
        if kill:
            self.proc.kill()
        else:
            self.request(op='shutdown')
        self.proc.wait(timeout=30)
        self.close_handles()
        self.start()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.close_handles()


class Harness:
    def __init__(self, directory):
        self.engines = [Engine(BIN, directory / 'candidate')]
        if BASELINE:
            self.engines.append(Engine(BASELINE, directory / 'baseline'))
        self.checked = 0

    def request(self, sql=None, op='execute', ok=True):
        responses = [engine.request(sql, op=op, ok=ok) for engine in self.engines]
        for other in responses[1:]:
            assert other == responses[0], ('baseline mismatch', sql and sql[:150], responses)
        return responses[0]

    def check(self, model, table, query, condition='', predicate=lambda r: True,
              offset=0, limit=10, descending=False, alias=False):
        distances = {}
        candidates = []
        for key, row in model.items():
            if not predicate(row):
                continue
            values = row[2]
            if values not in distances:
                distances[values] = sum((a - b) ** 2 for a, b in zip(values, query))
            candidates.append((distances[values], key, row))
        candidates.sort(key=lambda item: ((-item[0] if descending else item[0]), item[1]))
        wanted = candidates[offset:offset + limit]
        expression = 'L2_DISTANCE(v,%s)' % vector(query)
        projection = 'id,s,s2' + (',' + expression + ' AS distance' if alias else '')
        order = 'distance' if alias else expression
        sql = ('SELECT %s FROM %s%s ORDER BY %s %s LIMIT %d OFFSET %d' %
               (projection, table, ' WHERE ' + condition if condition else '',
                order, 'DESC' if descending else 'ASC', limit, offset))
        got = self.request(sql)['rows']
        expected = [[key, row[0], row[1]] for _, key, row in wanted]
        assert len(got) == len(expected), ('count mismatch', sql[:180], got, expected)
        for actual, want, candidate in zip(got, expected, wanted):
            assert actual[:3] == want, ('exact/tie/filter mismatch', sql[:180], got, expected)
            if alias:
                assert math.isclose(actual[3], math.sqrt(candidate[0]), rel_tol=1e-5, abs_tol=1e-5), (actual, candidate)
        self.checked += 1

    def suite(self, model, table, label, full=False):
        for q in QUERIES:
            self.check(model, table, q)
        for i, value in enumerate(GROUPS):
            q = QUERIES[i % len(QUERIES)]
            for col, ri in [('s', 0), ('s2', 1)]:
                predicate = lambda row, ri=ri, value=value: row[ri] == value
                condition = '%s=%s' % (col, quote(value))
                self.check(model, table, q, condition, predicate, offset=i % 3,
                           limit=[1, 7, 8, 9, 17][i % 5], alias=i % 4 == 0)
                if full:
                    self.check(model, table, q, '%s=%s' % (quote(value), col), predicate,
                               descending=i % 2 == 0)
        # The engine's cross-type comparison formats values as strings.  These
        # integer literals must use scalar comparison, not a mismatched group.
        for literal in (5, -2, 0):
            self.check(model, table, QUERIES[0], 's=%d' % literal,
                       lambda row, literal=literal: row[0] == str(literal))
            self.check(model, table, QUERIES[1], '%d=s' % literal,
                       lambda row, literal=literal: row[0] == str(literal))
        self.check(model, table, QUERIES[2], "s='not-present'", lambda row: False)
        self.check(model, table, QUERIES[3], "s='a' AND s2<>'not-present'",
                   lambda row: row[0] == 'a' and row[1] != 'not-present', limit=19)
        self.check(model, table, QUERIES[2], "s='a'", lambda row: row[0] == 'a', limit=0)
        self.check(model, table, QUERIES[2], "s='a'", lambda row: row[0] == 'a', limit=500)
        self.check(model, table, QUERIES[1], "s='a'", lambda row: row[0] == 'a', descending=True)
        print('PASS', label, flush=True)

    def close(self):
        for engine in self.engines:
            engine.close()


def main():
    with tempfile.TemporaryDirectory(prefix='mvdb-varchar-test-') as tmp:
        harness = Harness(Path(tmp))
        try:
            harness.request('CREATE TABLE t(id BIGINT PRIMARY KEY,s VARCHAR(180),s2 VARCHAR(180),v VECTOR(128))')
            rng = random.Random(771)
            prototypes = [tuple(rng.randrange(-4, 5) for _ in range(D)) for _ in range(37)]
            prototypes[0] = tuple([0] * D)
            model = {}
            batch = []
            for i in range(N):
                key = N - i
                row = (GROUPS[(i * 7) % len(GROUPS)], GROUPS[(i * 11 + 3) % len(GROUPS)], prototypes[i % len(prototypes)])
                model[key] = row
                batch.append('(%d,%s,%s,%s)' % (key, quote(row[0]), quote(row[1]), vector(row[2])))
                if len(batch) == 100 or i + 1 == N:
                    harness.request('INSERT INTO t VALUES ' + ','.join(batch))
                    batch.clear()
            harness.suite(model, 't', 'initial Unicode, escapes, string groups, cross-type and ties', full=True)
            # Scalar queries also use the new index; check exact string values.
            for value in GROUPS:
                got = harness.request('SELECT id,s FROM t WHERE s=%s ORDER BY id' % quote(value))['rows']
                assert got == [[key, row[0]] for key, row in sorted(model.items()) if row[0] == value]
            for key in (1, 129, 257, 2057):
                old = model[key]
                value = GROUPS[(key * 3) % len(GROUPS)]
                newvec = tuple([key % 4] * D)
                harness.request('UPDATE t SET s=%s,s2=%s,v=%s WHERE id=%d' %
                                (quote(value), quote(GROUPS[(key + 1) % len(GROUPS)]), vector(newvec), key))
                model[key] = (value, GROUPS[(key + 1) % len(GROUPS)], newvec)
            harness.request('DELETE FROM t WHERE id=12 OR id=2048')
            del model[12], model[2048]
            harness.suite(model, 't', 'index and grouped-code invalidation after updates/deletes')

            before = dict(model)
            harness.request('BEGIN')
            harness.request("UPDATE t SET s='changed',s2='a',v=%s WHERE id=129" % vector(prototypes[0]))
            model[129] = ('changed', 'a', prototypes[0])
            harness.request('DELETE FROM t WHERE id=257')
            del model[257]
            harness.request('INSERT INTO t VALUES (9001,%s,%s,%s)' % (quote('a\0x'), quote('🙂'), vector(prototypes[3])))
            model[9001] = ('a\0x', '🙂', prototypes[3])
            harness.request("INSERT INTO t VALUES (9002,'new','a',%s),(1,'duplicate','a',%s)" %
                            (vector(prototypes[4]), vector(prototypes[4])), ok=False)
            harness.suite(model, 't', 'transaction visibility and failed statement atomicity')
            harness.request('ROLLBACK')
            model = before
            harness.suite(model, 't', 'rollback invalidation')
            harness.request("INSERT INTO t VALUES (9002,'new','a',%s),(1,'duplicate','a',%s)" %
                            (vector(prototypes[4]), vector(prototypes[4])), ok=False)
            harness.suite(model, 't', 'autocommit failed insertion atomicity')

            harness.request('BEGIN')
            harness.request("UPDATE t SET s='5',s2='a',v=%s WHERE id=129" % vector(prototypes[2]))
            model[129] = ('5', 'a', prototypes[2])
            harness.request('DELETE FROM t WHERE id=257')
            del model[257]
            harness.request('COMMIT')
            harness.suite(model, 't', 'committed updates and deletes')
            for engine in harness.engines:
                engine.restart(kill=True)
            harness.suite(model, 't', 'SIGKILL recovery')
            for engine in harness.engines:
                engine.restart()
            harness.suite(model, 't', 'snapshot restart')

            # Keep the previous BIGINT preference with a VARCHAR before it.
            harness.request('CREATE TABLE u(id BIGINT PRIMARY KEY,s VARCHAR(180),s2 VARCHAR(180),g BIGINT,v VECTOR(128))')
            smaller = dict(list(model.items())[:769])
            rows = []
            for key, row in smaller.items():
                rows.append('(%d,%s,%s,%d,%s)' % (key, quote(row[0]), quote(row[1]), key % 7, vector(row[2])))
            for i in range(0, len(rows), 100):
                harness.request('INSERT INTO u VALUES ' + ','.join(rows[i:i + 100]))
            harness.suite(smaller, 'u', 'VARCHAR filters with BIGINT-preferred quantized layout', full=True)
            for group in range(7):
                subset = {key: row for key, row in smaller.items() if key % 7 == group}
                harness.check(subset, 'u', QUERIES[group % 4], 'g=%d' % group)
            print('PASS varchar_filter_test: %d independent exact queries%s' %
                  (harness.checked, ' plus baseline response cross-checks' if BASELINE else ''), flush=True)
        finally:
            harness.close()


if __name__ == '__main__':
    main()
