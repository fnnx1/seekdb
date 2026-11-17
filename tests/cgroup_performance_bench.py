#!/usr/bin/env python3
"""Local Docker/cgroup comparison using MiniVectorDB's independent exact oracle.

This is NOT the official benchmark. It uses the documented 10k/128D shape,
separate write/query containers, and records whole-cgroup memory.peak and raw
per-device io.stat. The precise frozen generator, baseline values, dedicated
storage-device isolation and cache initialization are unavailable. Docker's
attach forwarding adds a fixed round-trip cost; compare binaries under the
same harness, never splice these T values into native-pipe measurements.

No device totals are combined by default. Select one physical accounting
level using --io-device MAJOR:MINOR (repeat only for independent devices).
The chosen device can also host image/binary pages; such reads cannot be
attributed exclusively to /data by this harness.

Containers run as the host UID:GID so the host driver can fsync /data files
without launching a helper inside the measured cgroup. The fsync interval is
outside T, included in B/M. Data directories start fresh, but host file caches
are not globally dropped. All warmup and request generation policy is recorded.
Workload generation and exact answers are shared with performance_bench.py;
this harness defaults to one full query warmup pass, while that harness uses 0.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('mvdb_native_bench', ROOT/'tests/performance_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE).strip()


def read_pairs(path):
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


def io_stat(path):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        result[fields[0]] = {key: int(value) for key, value in (s.split('=', 1) for s in fields[1:])}
    return result


def sample(cgroup):
    result = {
        'memory_peak_bytes': int((cgroup/'memory.peak').read_text()),
        'memory_current_bytes': int((cgroup/'memory.current').read_text()),
        'memory_stat': read_pairs(cgroup/'memory.stat'),
        'io_stat': io_stat(cgroup/'io.stat'),
        'memory_events': read_pairs(cgroup/'memory.events'),
        'cpu_stat': read_pairs(cgroup/'cpu.stat'),
    }
    swap = cgroup/'memory.swap.current'
    if swap.exists():
        result['memory_swap_current_bytes'] = int(swap.read_text())
    return result


def fsync_data(directory):
    started = time.perf_counter()
    files = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        files.append({'name': str(path.relative_to(directory)), 'bytes': path.stat().st_size})
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return time.perf_counter() - started, files


def settled_sample(cgroup):
    # fsync already waits for data I/O. Allow delayed cgroup accounting to settle.
    previous = None
    equal = 0
    deadline = time.monotonic() + 2.0
    while True:
        result = sample(cgroup)
        if result['io_stat'] == previous:
            equal += 1
        else:
            previous, equal = result['io_stat'], 0
        if equal >= 2 or time.monotonic() >= deadline:
            return result
        time.sleep(0.02)


class Engine:
    def __init__(self, binary, directory, args):
        self.args = args
        self.name = 'mvdb-cgroup-' + uuid.uuid4().hex
        self.stderr = tempfile.TemporaryFile()
        self.process = None
        self.selector = selectors.DefaultSelector()
        self.closed = False
        cmd = ['docker', 'run', '-i', '--name', self.name,
               '--cpuset-cpus', args.cpus, '--cpus', str(args.cpu_quota),
               '--memory', args.memory, '--memory-swap', args.memory,
               '--network', 'none', '--read-only', '--cap-drop', 'ALL',
               '--user', '%d:%d' % (os.getuid(), os.getgid()),
               '--mount', 'type=bind,src=%s,dst=/app/candidate,readonly' % binary,
               '--mount', 'type=bind,src=%s,dst=/data' % directory,
               '-e', 'MVDB_SYNC_COMMIT=' + os.environ.get('MVDB_SYNC_COMMIT', '0'),
               '--entrypoint', '/app/candidate', args.image, '/data']
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.stderr)
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            ready = self.receive()
            if json.loads(ready) != {'type': 'ready'}:
                raise AssertionError('invalid ready handshake: %r' % ready)
            self.startup_s = time.perf_counter() - started
            self.pid = int(command(['docker', 'inspect', '--format', '{{.State.Pid}}', self.name]))
            groups = Path('/proc/%d/cgroup' % self.pid).read_text().splitlines()
            group = next(line[3:] for line in groups if line.startswith('0::'))
            self.cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
            self.at_ready = sample(self.cgroup)
        except BaseException:
            self.cleanup()
            raise

    def receive(self):
        if not self.selector.select(self.args.timeout):
            raise TimeoutError('engine response exceeded %ss' % self.args.timeout)
        line = self.process.stdout.readline()
        if not line or not line.endswith(b'\n'):
            self.stderr.seek(0)
            raise RuntimeError('engine response incomplete: %r; %s' %
                               (line, self.stderr.read().decode(errors='replace')[-4000:]))
        return line

    def request_raw(self, payload):
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        return self.receive()

    def requests_raw(self, payloads):
        return [self.request_raw(payload) for payload in payloads]

    def shutdown(self):
        response = json.loads(self.request_raw(bench.encode(99999999, op='shutdown')))
        if response.get('ok') is not True:
            raise AssertionError(response)
        if self.process.wait(timeout=self.args.timeout) != 0:
            raise AssertionError('nonzero docker/engine shutdown')

    def cleanup(self):
        if self.closed:
            return
        self.closed = True
        subprocess.run(['docker', 'rm', '-f', self.name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process.stdin.close()
            self.process.stdout.close()
        self.selector.close()
        self.stderr.close()


def decode(lines, payloads):
    result = []
    for raw, payload in zip(lines, payloads):
        response = json.loads(raw)
        request = json.loads(payload)
        if response.get('id') != request.get('id') or response.get('ok') is not True:
            raise AssertionError(response)
        result.append(response)
    if len(lines) != len(payloads):
        raise AssertionError('incomplete response sequence')
    return result


def selected_io(stats, devices):
    if not devices:
        return None
    missing = set(devices) - set(stats)
    if missing:
        raise RuntimeError('selected I/O devices absent from accounting: %s; got %s' % (sorted(missing), sorted(stats)))
    return sum(stats[dev].get('rbytes', 0) + stats[dev].get('wbytes', 0) for dev in devices)


def phase(binary, args, workload, kind):
    schema, writes, searches, expected, _, _ = workload
    with tempfile.TemporaryDirectory(prefix='mvdb-cgroup-data-', dir=args.data_parent) as temporary:
        directory = Path(temporary)
        engine = Engine(binary, directory, args)
        try:
            decode([engine.request_raw(schema)], [schema])
            started = time.perf_counter()
            loaded_raw = engine.requests_raw(writes)
            load_s = time.perf_counter() - started
            if kind == 'write':
                measured_s = load_s
            decode(loaded_raw, writes)
            warmed_s = 0.0
            if kind == 'query':
                for _ in range(args.warmup):
                    started = time.perf_counter()
                    warmed_raw = engine.requests_raw(searches)
                    warmed_s += time.perf_counter() - started
                    bench.check_answers(decode(warmed_raw, searches), expected)
                before_measurement = sample(engine.cgroup)
                started = time.perf_counter()
                query_raw = engine.requests_raw(searches)
                measured_s = time.perf_counter() - started
            else:
                before_measurement = engine.at_ready
            before_sync = sample(engine.cgroup)
            sync_s, files = fsync_data(directory)
            metrics = settled_sample(engine.cgroup)
            if metrics['memory_events'].get('oom', 0) or metrics['memory_events'].get('oom_kill', 0):
                raise AssertionError('cgroup OOM occurred')
            if metrics.get('memory_swap_current_bytes', 0):
                raise AssertionError('swap is disabled but nonzero usage observed')
            # Validation is after the measured resource snapshot for the write phase.
            if kind == 'write':
                query_raw = engine.requests_raw(searches)
            digest = bench.check_answers(decode(query_raw, searches), expected)
            result = {'phase': kind, 'T_s': measured_s,
                      'M_bytes': metrics['memory_peak_bytes'],
                      'B_bytes_selected_devices': selected_io(metrics['io_stat'], args.io_device),
                      'startup_s': engine.startup_s, 'load_s': load_s,
                      'warmup_s': warmed_s, 'post_measurement_fsync_s': sync_s,
                      'at_ready': engine.at_ready, 'before_measurement': before_measurement,
                      'before_fsync': before_sync, 'after_fsync': metrics,
                      'files': files, 'oracle_queries_verified': len(expected),
                      'result_sha256': digest}
            engine.shutdown()
            return result
        finally:
            engine.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', action='append', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--dims', type=int, default=128)
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--groups', type=int, default=10)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--ties', choices=['copies', 'zeros'], default='copies')
    parser.add_argument('--filter-type', choices=['bigint', 'varchar'], default='bigint')
    parser.add_argument('--warmup', type=int, default=1, help='full query-sequence warmup passes (default 1)')
    parser.add_argument('--phase', choices=['both', 'write', 'query'], default='both')
    parser.add_argument('--cpus', default='0,1')
    parser.add_argument('--cpu-quota', type=float, default=2.0)
    parser.add_argument('--driver-cpu', type=int, default=2)
    parser.add_argument('--memory', default='2g')
    parser.add_argument('--data-parent', type=Path, default=ROOT)
    parser.add_argument('--image', default='minivectordb:check')
    parser.add_argument('--io-device', action='append', default=[])
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    for name in ['rounds', 'rows', 'dims', 'queries', 'batch', 'groups', 'k']:
        if getattr(args, name) <= 0:
            parser.error('--%s must be positive' % name)
    if args.queries % 2 or args.dims > 1024 or args.warmup < 0:
        parser.error('queries must be even, dims<=1024, warmup>=0')
    args.data_parent = args.data_parent.resolve()
    binaries = [str(Path(p).resolve()) for p in args.binary]
    if any(not os.access(p, os.X_OK) for p in binaries):
        parser.error('all --binary paths must be executable Linux binaries for the image')
    if not args.data_parent.is_dir():
        parser.error('--data-parent must exist')
    if len(set(args.io_device)) != len(args.io_device):
        parser.error('duplicate --io-device')
    os.sched_setaffinity(0, {args.driver_cpu})
    report = {
        'timestamp_Asia_Shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'kind': 'local Docker/cgroup comparison; NOT official E',
        'workload': {name: getattr(args, name) for name in ['rows', 'dims', 'queries', 'batch', 'groups', 'k', 'seed', 'ties', 'filter_type', 'warmup', 'phase', 'cpus', 'cpu_quota', 'driver_cpu', 'memory', 'image', 'io_device']},
        'data_parent': str(args.data_parent),
        'data_filesystem': command(['stat', '-f', '-c', '%T', str(args.data_parent)]),
        'data_mount': command(['findmnt', '-T', str(args.data_parent), '-no', 'SOURCE,FSTYPE,MAJ:MIN']),
        'container_uid_gid': '%d:%d' % (os.getuid(), os.getgid()),
        'requested_MVDB_SYNC_COMMIT': os.environ.get('MVDB_SYNC_COMMIT', '0'),
        'binaries': {p: {'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest(), 'rounds': []} for p in binaries},
        'limits': [
            'Frozen official generator/baselines/cache procedure unavailable; local dyadic vectors.',
            'T includes docker attach forwarding, unlike native-pipe measurements.',
            'M is whole measured container memory.peak including page cache; driver and compiler are outside.',
            'B is raw per-device io.stat through final host fsync; devices are not summed unless selected.',
            'Image and /data may share a physical device, so B may include image/binary-page reads.',
            'Fresh directories but no global host cache drop; warmup sequence is local policy.',
            'Resource snapshot precedes post-phase validation and normal shutdown.'
        ]}
    with tempfile.TemporaryDirectory(prefix='mvdb-cgroup-driver-') as temporary:
        workload = bench.prepare(args, Path(temporary))
        report['preparation_s'], report['oracle_s'] = workload[4:]
        kinds = ['write', 'query'] if args.phase == 'both' else [args.phase]
        for iteration in range(args.rounds):
            offset = iteration % len(binaries)
            for binary in binaries[offset:] + binaries[:offset]:
                result = {}
                for kind in kinds:
                    metrics = phase(binary, args, workload, kind)
                    result[kind] = metrics
                    print('round %d %s %s: T %.3f ms M %.3f MiB B_selected %s io_devices %s' %
                          (iteration+1, Path(binary).name, kind, metrics['T_s']*1000,
                           metrics['M_bytes']/2**20, metrics['B_bytes_selected_devices'],
                           list(metrics['after_fsync']['io_stat'])), file=sys.stderr, flush=True)
                report['binaries'][binary]['rounds'].append(result)
                args.json.write_text(json.dumps(report, indent=2)+'\n')
    for binary, details in report['binaries'].items():
        details['median'] = {}
        for kind in kinds:
            values = [r[kind] for r in details['rounds']]
            keys = ['T_s', 'M_bytes', 'B_bytes_selected_devices', 'load_s', 'warmup_s', 'post_measurement_fsync_s']
            details['median'][kind] = {key: statistics.median(r[key] for r in values)
                                     if all(r[key] is not None for r in values) else None for key in keys}
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(str(args.json.resolve()))


if __name__ == '__main__':
    main()
