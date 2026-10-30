"""Additional AI-column runtime regressions; run against a disposable local DB.

Usage: python test_ai_column_runtime.py --port 2895 [--case lifecycle]
Requires PyMySQL. Model calls use local HTTP mocks only.
"""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pymysql


def run_registration_case(port):
    """Run on a fresh instance built without objects from before the new sysvar."""
    def connect():
        return pymysql.connect(host='127.0.0.1', port=port, user='root',
                               autocommit=True, connect_timeout=5, read_timeout=15)

    # Missing bootstrap registration fails in the handshake, before any SQL.
    db = connect()
    try:
        with db.cursor() as cur:
            cur.execute('SELECT @@global.ai_column_strict_read, @@session.ai_column_strict_read')
            assert cur.fetchone() == (0, 0), 'fresh instance must default to non-strict reads'
            try:
                cur.execute('SET GLOBAL ai_column_strict_read=ON')
                fresh = connect()
                try:
                    with fresh.cursor() as new_cur:
                        new_cur.execute('SELECT @@session.ai_column_strict_read')
                        assert new_cur.fetchone() == (1,), 'new connection must inherit the global value'
                        new_cur.execute('SET SESSION ai_column_strict_read=OFF')
                        new_cur.execute('SELECT @@global.ai_column_strict_read, @@session.ai_column_strict_read')
                        assert new_cur.fetchone() == (1, 0), 'session override must not change the global value'
                finally:
                    fresh.close()
            finally:
                cur.execute('SET GLOBAL ai_column_strict_read=OFF')
    finally:
        db.close()
    print('PASS: fresh login, strict-read default, global inheritance and session override', flush=True)


def run_endpoint_lifecycle_case(port, recycle):
    """Keep one HTTP call in flight while another durable task loses its endpoint."""
    case = 'recyclebin-orphan' if recycle else 'missing-endpoint'
    suffix = 'orphan' if recycle else 'missing_ep'
    database = 'ai_column_lifecycle_' + suffix
    broken_model = 'ai_lifecycle_' + suffix + '_broken'
    healthy_model = 'ai_lifecycle_' + suffix + '_healthy'
    started = threading.Event()
    released = threading.Event()
    lock = threading.Lock()
    calls = Counter()
    first_prompt = []
    blocked_prompts = (case + '-one', case + '-two')
    healthy_prompt = case + '-healthy'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            prompt = next(m['content'] for m in body['messages'] if m['role'] == 'user')
            with lock:
                calls[prompt] += 1
                should_block = prompt in blocked_prompts and not first_prompt
                if should_block:
                    first_prompt.append(prompt)
            if should_block:
                started.set()
                if not released.wait(30):
                    raise TimeoutError('endpoint lifecycle request was not released')
            payload = json.dumps({'choices': [{'message': {'content': 'mock:' + prompt}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    db = pymysql.connect(host='127.0.0.1', port=port, user='root', database='oceanbase',
                         autocommit=True, read_timeout=30, charset='utf8mb4')
    cur = db.cursor()

    def drop_model_objects(model):
        for method, obj in [('DROP_AI_MODEL_ENDPOINT', model + '_ep'), ('DROP_AI_MODEL', model)]:
            try:
                cur.execute(f'CALL DBMS_AI_SERVICE.{method}(%s)', (obj,))
            except pymysql.MySQLError as error:
                if error.args[0] not in (11112, 11118):
                    raise

    def create_table(name, model):
        cur.execute(f'''CREATE TABLE `{database}`.`{name}`(
            id BIGINT PRIMARY KEY, content VARCHAR(200),
            answer VARCHAR(400) AI COLUMN (AI_COMPLETE(%s,content))
                WITH(sync_mode='async',max_retries=1))''', (model,))

    try:
        # Cleanup must not manufacture another recycled table before the case.
        cur.execute('SET SESSION recyclebin=OFF')
        cur.execute(f'DROP DATABASE IF EXISTS `{database}`')
        cur.execute(f'CREATE DATABASE `{database}`')
        for model in (broken_model, healthy_model):
            drop_model_objects(model)
            cur.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)',
                        (model, json.dumps({'type': 'completion', 'model_name': model})))
            cur.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s,%s)',
                        (model + '_ep', json.dumps(dict(ai_model_name=model, provider='openai',
                         access_key='local-test-only', request_model_name=model,
                         url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))))
        create_table('broken', broken_model)
        create_table('healthy', healthy_model)
        cur.execute('''SELECT t.table_id FROM oceanbase.__all_table t
            JOIN oceanbase.__all_database d ON d.database_id=t.database_id
            WHERE d.database_name=%s AND t.table_name='broken' ''', (database,))
        table_id = cur.fetchone()[0]
        cur.execute(f'INSERT INTO `{database}`.broken(id,content) VALUES(1,%s),(2,%s)',
                    blocked_prompts)
        try:
            assert started.wait(10), (case, 'first model request did not reach mock')
            # Do not assume request-id order or that both rows were prefetched
            # in one scan: whichever row reached HTTP first is our barrier.
            with lock:
                inflight_prompt = first_prompt[0]
                pending_prompt = next(p for p in blocked_prompts if p != inflight_prompt)
                assert calls[pending_prompt] == 0, (case, dict(calls))
            cur.execute('SELECT count(*) FROM oceanbase.__all_ai_column_task WHERE table_id=%s',
                        (table_id,))
            assert cur.fetchone()[0] == 2, (case, 'both durable tasks must still exist')
            if recycle:
                cur.execute('SET SESSION recyclebin=ON')
                cur.execute(f'DROP TABLE `{database}`.broken')
                cur.execute('SELECT count(*) FROM oceanbase.__all_recyclebin WHERE table_id=%s',
                            (table_id,))
                assert cur.fetchone()[0] == 1, 'DROP did not put the table into the recyclebin'
            cur.execute('CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT(%s)', (broken_model + '_ep',))
            if recycle:
                cur.execute('CALL DBMS_AI_SERVICE.DROP_AI_MODEL(%s)', (broken_model,))
            # Enqueue unrelated work before releasing the first HTTP response.
            # Its completion proves that the lost endpoint cannot starve it.
            cur.execute(f'INSERT INTO `{database}`.healthy(id,content) VALUES(1,%s)',
                        (healthy_prompt,))
        finally:
            released.set()

        observed_states = set()
        deadline = time.monotonic() + 20
        while True:
            cur.execute('''SELECT state,retry_count FROM oceanbase.__all_ai_column_task
                WHERE table_id=%s''', (table_id,))
            tasks = cur.fetchall()
            observed_states.update(tasks)
            assert all(0 <= retry <= 1 for _, retry in tasks), (case, 'retry limit exceeded', tasks)
            cur.execute(f'SELECT answer FROM `{database}`.healthy WHERE id=1')
            healthy_answer = cur.fetchone()[0]
            with lock:
                assert calls[pending_prompt] == 0, (case, 'pending row called a deleted endpoint', dict(calls))
            if not tasks and healthy_answer is not None:
                assert healthy_answer == 'mock:' + healthy_prompt, healthy_answer
                break
            assert time.monotonic() < deadline, (
                case, 'old tasks did not terminate or healthy work was starved', tasks, healthy_answer)
            time.sleep(0.03)
        with lock:
            assert calls[inflight_prompt] == 1 and calls[healthy_prompt] == 1, (case, dict(calls))
        if not recycle:
            cur.execute(f'SELECT content,answer FROM `{database}`.broken ORDER BY id')
            assert dict(cur.fetchall()) == {
                inflight_prompt: 'mock:' + inflight_prompt, pending_prompt: None}, case
            cur.execute('SET SESSION ai_column_strict_read=ON')
            try:
                cur.execute(f'SELECT answer FROM `{database}`.broken')
            except pymysql.MySQLError as error:
                assert error.args[0] == 11120, error
            else:
                raise AssertionError('missing endpoint result was incorrectly considered ready')
            finally:
                cur.execute('SET SESSION ai_column_strict_read=OFF')
        # State 3/4 samples are diagnostic only: a poll can miss a short state.
        # NULL plus strict-read rejection proves not-ready, not a hidden FAILED
        # value; task removal and the deadline prove bounded terminal cleanup.
        print('LIFECYCLE_EVIDENCE ' + json.dumps(dict(
            case=case, observed_task_states=sorted(observed_states), calls=dict(calls)), sort_keys=True),
            flush=True)
        print('PASS: ' + ('recyclebin orphan cleanup and healthy progress' if recycle else
              'missing endpoint bounded retries, terminal cleanup and healthy progress'), flush=True)
    finally:
        released.set()
        try:
            cur.execute('SET SESSION ai_column_strict_read=OFF')
            cur.execute('SET SESSION recyclebin=OFF')
            cur.execute(f'DROP DATABASE IF EXISTS `{database}`')
            for model in (broken_model, healthy_model):
                drop_model_objects(model)
        finally:
            db.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2895)
    parser.add_argument('--case', choices=('all', 'existing', 'registration', 'lifecycle',
                                          'recyclebin-orphan', 'missing-endpoint'), default='all')
    args = parser.parse_args()
    if args.case in ('all', 'registration'):
        run_registration_case(args.port)
    if args.case == 'registration':
        return
    if args.case not in ('all', 'existing'):
        if args.case in ('lifecycle', 'recyclebin-orphan'):
            run_endpoint_lifecycle_case(args.port, recycle=True)
        if args.case in ('lifecycle', 'missing-endpoint'):
            run_endpoint_lifecycle_case(args.port, recycle=False)
        return
    calls = Counter()
    lock = threading.Lock()
    servers = []
    threads = []
    constant_prompt = "literal {0}, quote ', backslash \\ and 中文"
    block_constant_request = threading.Event()
    constant_request_started = threading.Event()
    release_constant_request = threading.Event()

    def handler(name):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prompt = next(m['content'] for m in body['messages'] if m['role'] == 'user')
                with lock:
                    calls[(name, prompt)] += 1
                    attempt = calls[(name, prompt)]
                if prompt == constant_prompt and block_constant_request.is_set():
                    constant_request_started.set()
                    if not release_constant_request.wait(15):
                        raise TimeoutError('constant request was not released by the test')
                time.sleep(0.12)
                if name == 'b' and attempt == 1:
                    status, response = 400, {'error': {'message': 'transient test failure'}}
                else:
                    status = 200
                    content = 'invalid-number' if prompt == 'invalid-number' else name + ':' + prompt
                    response = {'choices': [{'message': {'content': content}}]}
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        return Handler

    db = pymysql.connect(host='127.0.0.1', port=args.port, user='root', database='oceanbase',
                         autocommit=True, read_timeout=30, charset='utf8mb4')
    cur = db.cursor()
    try:
        cur.execute('DROP DATABASE IF EXISTS ai_column_extra_regression')
        cur.execute('CREATE DATABASE ai_column_extra_regression')
        for name in ('a', 'b'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler(name))
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
            model, endpoint = 'ai_extra_' + name, 'ai_extra_' + name + '_ep'
            for statement in ('DROP_AI_MODEL_ENDPOINT', 'DROP_AI_MODEL'):
                try:
                    cur.execute(f'CALL DBMS_AI_SERVICE.{statement}(%s)',
                                (endpoint if statement.endswith('ENDPOINT') else model,))
                except pymysql.MySQLError as e:
                    if e.args[0] not in (11112, 11118):
                        raise
            cur.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s,%s)',
                        (model, json.dumps({'type': 'completion', 'model_name': model})))
            cur.execute('CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s,%s)',
                        (endpoint, json.dumps(dict(ai_model_name=model, provider='openai',
                         access_key='local-test-only', request_model_name=model,
                         url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'))))
        cur.execute('''CREATE TABLE ai_column_extra_regression.docs(
            id BIGINT PRIMARY KEY, content VARCHAR(200),
            a VARCHAR(400) AI COLUMN (AI_COMPLETE('ai_extra_a', content))
                WITH(sync_mode='async',max_retries=2),
            b VARCHAR(400) AI COLUMN (AI_COMPLETE('ai_extra_b', content))
                WITH(sync_mode='async',max_retries=2))''')
        prompt = "backslash \\ and quote ' and newline\n中文"
        cur.execute('INSERT INTO ai_column_extra_regression.docs(id,content) VALUES(1,%s)', (prompt,))
        deadline = time.monotonic() + 20
        while True:
            cur.execute('SELECT a,b FROM ai_column_extra_regression.docs WHERE id=1')
            a, b = cur.fetchone()
            assert (a is None) == (b is None), ('partial group publication', a, b)
            if a is not None:
                assert a == 'a:' + prompt and b == 'b:' + prompt, (a, b)
                break
            assert time.monotonic() < deadline, 'mixed endpoint group did not finish'
            time.sleep(0.03)
        assert calls[('a', prompt)] == 1, dict(calls)
        assert calls[('b', prompt)] == 2, dict(calls)
        print('PASS: mixed endpoints, atomic publication, cached success, byte-preserving result', flush=True)

        for sql in ["UPDATE ai_column_extra_regression.docs SET a='manual' WHERE id=1",
                    "INSERT INTO ai_column_extra_regression.docs(id,content,a) VALUES(2,'x','manual')"]:
            try:
                cur.execute(sql)
            except pymysql.MySQLError as e:
                assert e.args[0] == 1235, e
            else:
                raise AssertionError('manual AI-result assignment was accepted')
        print('PASS: direct AI-result assignments rejected', flush=True)

        cur.execute('START TRANSACTION')
        cur.execute("INSERT INTO ai_column_extra_regression.docs(id,content) VALUES(3,'rolled-back')")
        cur.execute('ROLLBACK')
        time.sleep(0.8)
        assert calls[('a', 'rolled-back')] == 0 and calls[('b', 'rolled-back')] == 0
        print('PASS: rolled-back writes do not call models', flush=True)

        cur.execute('''CREATE TABLE ai_column_extra_regression.constant_prompts(
            id BIGINT PRIMARY KEY,
            answer VARCHAR(400) AI COLUMN (AI_COMPLETE('ai_extra_a',%s))
                WITH(sync_mode='async',max_retries=1))''', (constant_prompt,))
        cur.execute('INSERT INTO ai_column_extra_regression.constant_prompts(id) VALUES(1)')
        deadline = time.monotonic() + 20
        while True:
            cur.execute('SELECT answer FROM ai_column_extra_regression.constant_prompts WHERE id=1')
            answer = cur.fetchone()[0]
            if answer is not None:
                assert answer == 'a:' + constant_prompt, answer
                break
            assert time.monotonic() < deadline, 'constant prompt did not finish'
            time.sleep(0.1)
        assert calls[('a', constant_prompt)] == 1, dict(calls)
        # While the first model request is blocked, delete both rows. Any
        # already-picked sibling must be dropped after checking row existence,
        # and the first response cannot publish into either deleted row.
        block_constant_request.set()
        cur.execute('INSERT INTO ai_column_extra_regression.constant_prompts(id) VALUES(2),(3)')
        try:
            assert constant_request_started.wait(10), 'constant request did not reach mock'
            cur.execute('DELETE FROM ai_column_extra_regression.constant_prompts WHERE id IN (2,3)')
        finally:
            release_constant_request.set()
        # A new task is a completion barrier for the single worker: observing
        # its result proves the earlier in-flight response has been handled.
        cur.execute('INSERT INTO ai_column_extra_regression.constant_prompts(id) VALUES(4)')
        deadline = time.monotonic() + 20
        while True:
            cur.execute('SELECT answer FROM ai_column_extra_regression.constant_prompts WHERE id=4')
            answer = cur.fetchone()[0]
            with lock:
                request_count = calls[('a', constant_prompt)]
            assert request_count <= 3, ('deleted row caused a model request', dict(calls))
            if answer is not None:
                assert answer == 'a:' + constant_prompt, answer
                break
            assert time.monotonic() < deadline, 'constant worker barrier did not finish'
            time.sleep(0.1)
        assert calls[('a', constant_prompt)] == 3, dict(calls)
        cur.execute('SELECT id,answer FROM ai_column_extra_regression.constant_prompts ORDER BY id')
        assert cur.fetchall() == ((1, 'a:' + constant_prompt), (4, 'a:' + constant_prompt))
        print('PASS: literal constant prompt, stale deleted task and late response', flush=True)

        cur.execute('''CREATE TABLE ai_column_extra_regression.bad_types(
            id BIGINT PRIMARY KEY, content VARCHAR(100),
            score BIGINT AI COLUMN (AI_COMPLETE('ai_extra_a',content))
                WITH(sync_mode='async',max_retries=1))''')
        cur.execute("INSERT INTO ai_column_extra_regression.bad_types(id,content) VALUES(1,'invalid-number')")
        deadline = time.monotonic() + 20
        while True:
            cur.execute('''SELECT count(*) FROM oceanbase.__all_ai_column_task task
                JOIN oceanbase.__all_table t ON task.table_id=t.table_id
                JOIN oceanbase.__all_database d ON t.database_id=d.database_id
                WHERE d.database_name='ai_column_extra_regression' AND t.table_name='bad_types' ''')
            if cur.fetchone()[0] == 0:
                break
            assert time.monotonic() < deadline, 'conversion failure was not bounded'
            time.sleep(0.1)
        cur.execute('SELECT score FROM ai_column_extra_regression.bad_types')
        assert cur.fetchone() == (None,)
        assert calls[('a', 'invalid-number')] == 2, dict(calls)
        print('PASS: invalid typed output exhausts bounded retries and stays NULL', flush=True)
    finally:
        release_constant_request.set()
        cur.execute('DROP DATABASE IF EXISTS ai_column_extra_regression')
        for name in ('a', 'b'):
            for method, obj in [('DROP_AI_MODEL_ENDPOINT', 'ai_extra_' + name + '_ep'),
                                ('DROP_AI_MODEL', 'ai_extra_' + name)]:
                try:
                    cur.execute(f'CALL DBMS_AI_SERVICE.{method}(%s)', (obj,))
                except pymysql.MySQLError:
                    pass
        db.close()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)

    if args.case == 'all':
        run_endpoint_lifecycle_case(args.port, recycle=True)
        run_endpoint_lifecycle_case(args.port, recycle=False)


if __name__ == '__main__':
    main()
