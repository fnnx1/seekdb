#!/usr/bin/env python3
"""Ordinary-decimal variant of the local Docker/cgroup benchmark.

Default: 10k x 128D, 100 commits, 1000 exact Top10 queries (half filtered).
Coordinates are random.uniform(-1, 1), formatted to eight decimal places.
Set MVDB_REFERENCE_BINARY to the native reference executable. Expected
answers come from that baseline, on
CPUs 14,15 during preparation, outside measured T/B/M. This is baseline
differential checking, NOT an independent exact integer/float oracle.
The inherited --ties option is unused: this workload injects no tied vectors.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(os.environ.get('MVDB_REFERENCE_BINARY', '/nonexistent/set-MVDB_REFERENCE_BINARY'))
BASELINE_CPUS = {14, 15}
spec = importlib.util.spec_from_file_location(
    'mvdb_cgroup_decimal_runner', REPO / 'tests/cgroup_performance_bench.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
native = runner.bench
REFERENCE = {}


def prepare_decimal(args, directory):
    if not BASELINE.is_file() or not os.access(BASELINE, os.X_OK):
        raise RuntimeError('required native differential baseline is not executable: %s' % BASELINE)
    started = time.perf_counter()
    rng = random.Random(args.seed)

    def vector():
        return '[' + ','.join('%.8f' % rng.uniform(-1, 1) for _ in range(args.dims)) + ']'

    def group_literal(group):
        return "'group_%d'" % group if args.filter_type == 'varchar' else str(group)

    group_type = 'VARCHAR(32)' if args.filter_type == 'varchar' else 'BIGINT'
    ident = 1
    schema = native.encode(ident, 'CREATE TABLE t (id BIGINT PRIMARY KEY, grp %s, v VECTOR(%d));'
                           % (group_type, args.dims))
    rows = ["(%d,%s,'%s')" % (i, group_literal(i % args.groups), vector())
            for i in range(args.rows)]
    writes = []
    for start in range(0, args.rows, args.batch):
        for sql in ['BEGIN;', 'INSERT INTO t VALUES ' + ','.join(rows[start:start + args.batch]) + ';',
                    'COMMIT;']:
            ident += 1
            writes.append(native.encode(ident, sql))
    queries = []
    for i in range(args.queries):
        condition = ' WHERE grp = ' + group_literal(i % args.groups) if i < args.queries // 2 else ''
        ident += 1
        queries.append(native.encode(ident, "SELECT id FROM t%s ORDER BY L2_DISTANCE(v,'%s') LIMIT %d;"
                                     % (condition, vector(), args.k)))
    preparation_s = time.perf_counter() - started
    REFERENCE.update(kind='baseline differential; NOT independent oracle',
                     binary=str(BASELINE), sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                     engine_cpus=sorted(BASELINE_CPUS), mode='exact',
                     preparation_outside_measured_T_B_M=True,
                     requested_MVDB_SYNC_COMMIT=os.environ.get('MVDB_SYNC_COMMIT', '0'))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='mvdb-decimal-reference-', dir=args.data_parent) as temporary:
        engine = native.Engine(str(BASELINE), temporary, BASELINE_CPUS, args.timeout)
        try:
            engine.request(schema)
            for payload in writes:
                engine.request(payload)
            answers = [engine.request(payload) for payload in queries]
            expected = [[row[0] for row in response.get('rows', [])] for response in answers]
            engine.request(native.encode(99999999, op='shutdown'))
            if engine.process.wait(timeout=args.timeout) != 0:
                raise AssertionError('native differential baseline failed at shutdown')
        finally:
            engine.kill()
    REFERENCE['queries'] = len(expected)
    REFERENCE['result_sha256'] = hashlib.sha256(
        json.dumps(expected, separators=(',', ':')).encode()).hexdigest()
    return schema, writes, queries, expected, preparation_s, time.perf_counter() - started


original_phase = runner.phase


def differential_phase(*args, **kwargs):
    result = original_phase(*args, **kwargs)
    result['baseline_differential_queries_verified'] = result.pop('oracle_queries_verified')
    result['validation'] = 'baseline differential; NOT independent oracle'
    return result


class ReportJSON:
    """Relabel every intermediate/final report without replacing runner logic."""

    loads = staticmethod(json.loads)

    @staticmethod
    def dumps(value, *args, **kwargs):
        if isinstance(value, dict) and {'kind', 'workload', 'binaries', 'limits'} <= value.keys():
            value = dict(value)
            value['kind'] = 'local ordinary-decimal Docker/cgroup baseline differential; NOT official E'
            value['workload'] = dict(value['workload'],
                                     dataset='random.uniform(-1,1), eight decimal digits, engine float32',
                                     row_order='sequential primary keys',
                                     ties='none injected; inherited --ties option unused',
                                     mode='exact', requests_preencoded=True)
            value['correctness_reference'] = dict(REFERENCE)
            if 'oracle_s' in value:
                value['baseline_reference_s'] = value.pop('oracle_s')
            value['limits'] = [line.replace('local dyadic vectors', 'local eight-decimal uniform random vectors')
                               for line in value['limits']]
            value['limits'].append('Answers use the named native baseline, not an independent oracle; shared bugs may pass.')
            value['limits'].append('Native reference preparation is outside the measured containers; it does not drop host caches.')
        return json.dumps(value, *args, **kwargs)


runner.bench.prepare = prepare_decimal
runner.phase = differential_phase
runner.json = ReportJSON
runner.__doc__ = __doc__ + '\n\n' + runner.__doc__.replace(
    "using MiniVectorDB's independent exact oracle", 'using baseline differential answers').replace(
    'Workload generation and exact answers are shared with performance_bench.py;',
    'The wrapper supplies ordinary-decimal requests and baseline differential answers;')


if __name__ == '__main__':
    runner.main()
