"""Record/replay BMW results on one physical corpus, preserving every score bit.

--prepare creates ONLY fts_bmw_round3.docs (and fails if that table exists).
Record a baseline once, then restart the same database with the candidate and
run without --prepare plus --reference. Never recreate the fixture between
these runs: implicit LIMIT tie order is compared on the same physical IDs.
The competition's fts_large_bench database is never accessed or changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import pymysql


def encoded(value):
    if isinstance(value, float):
        return {"double_hex": struct.pack(">d", value).hex(), "text": repr(value)}
    if value is None:
        return None
    return str(value)


def fixture_rows():
    """Dense/tied -> rare high scores -> low sparse scores -> dense again.

    Changes happen after several 256/512/1024/2048 sequential windows, allowing
    a high-scoring row to raise the threshold inside a long window. Additional
    rare terms and nine dimensions exercise pruning and the >8 fallback.
    """
    for ident in range(18000):
        zone = ident // 4500
        if zone == 0:
            text = "alpha beta " + "walnut cedar maple " * 9
        elif zone == 1:
            text = ("alpha gamma " * 12 if ident % 97 == 0
                    else "alpha " + "walnut cedar maple " * 30)
        elif zone == 2:
            text = ("beta gamma " * 16 if ident % 193 == 0
                    else "beta " + "walnut cedar maple " * 20)
        else:
            text = "alpha beta gamma delta epsilon zeta theta lambda omega "
        if ident % 137 == 0:
            text += "quasar nebula " * (1 + ident % 5)
        yield ident, zone, text


def queries():
    expressions = {
        "ties_dense": "MATCH(content) AGAINST('alpha beta')",
        "threshold_transition": "MATCH(content) AGAINST('alpha gamma')",
        "sparse": "MATCH(content) AGAINST('quasar nebula')",
        "eight_dimensions": "MATCH(content) AGAINST('alpha beta gamma delta epsilon zeta theta lambda')",
        "nine_dimensions": "MATCH(content) AGAINST('alpha beta gamma delta epsilon zeta theta lambda omega')",
        "boolean": "MATCH(content) AGAINST('+alpha -beta gamma' IN BOOLEAN MODE)",
        "duplicate_boolean": "MATCH(content) AGAINST('alpha alpha beta' IN BOOLEAN MODE)",
        "weighted_terms": "MATCH(content^1, 'alpha^2 beta^0.5 gamma^1.25', '')",
        "weighted_field": "MATCH(content^2, 'alpha beta gamma', '')",
        "unit_weight": "MATCH(content^1, 'alpha beta gamma', '')",
        "minimum_match": "MATCH(content^1, 'alpha beta gamma', 'minimum_should_match=2')",
    }
    for name, match in expressions.items():
        score = "SCORE()" if "content^" in match else match
        yield name + "/all", f"SELECT id, {score} AS score FROM docs WHERE {match} ORDER BY id"
        for limit, offset in [(1, 0), (20, 0), (35, 13), (100, 4000), (7, 18001)]:
            suffix = f"LIMIT {limit} OFFSET {offset}"
            # Implicit relevance order and equal-score order are externally visible.
            yield f"{name}/implicit/{limit}/{offset}", f"SELECT id, {score} AS score FROM docs WHERE {match} {suffix}"
        yield name + "/explicit", f"SELECT id, {score} AS score FROM docs WHERE {match} ORDER BY score DESC, id LIMIT 100"
        yield name + "/count_offset", f"SELECT COUNT(*) FROM (SELECT id FROM docs WHERE {match} LIMIT 100 OFFSET 4000) t"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2893)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    connection = pymysql.connect(host=args.host, port=args.port, user="root",
        password=os.environ.get("SEEKDB_TEST_PASSWORD", ""), charset="utf8mb4",
        autocommit=True, connect_timeout=10, read_timeout=180)
    results, timings, plans = {}, {}, {}
    with connection.cursor() as cursor:
        cursor.execute("SET ob_query_timeout=180000000")
        if args.prepare:
            cursor.execute("CREATE DATABASE IF NOT EXISTS fts_bmw_round3")
            cursor.execute("USE fts_bmw_round3")
            cursor.execute("CREATE TABLE docs(id BIGINT PRIMARY KEY, zone INT, content TEXT)")
            rows = list(fixture_rows())
            for begin in range(0, len(rows), 300):
                cursor.executemany("INSERT INTO docs VALUES(%s,%s,%s)", rows[begin:begin+300])
            cursor.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft(content) WITH PARSER beng")
        else:
            cursor.execute("USE fts_bmw_round3")
        cursor.execute("SELECT COUNT(*), MIN(id), MAX(id), SUM(zone) FROM docs")
        if cursor.fetchone() != (18000, 0, 17999, 27000):
            raise SystemExit("Unexpected fixture: use --prepare once on the baseline")
        for name, sql in queries():
            started = time.perf_counter()
            try:
                cursor.execute(sql)
                results[name] = [[encoded(v) for v in row] for row in cursor.fetchall()]
            except pymysql.MySQLError as error:
                # Unsupported query shapes are part of the baseline contract.
                # Keep the exact errno/message so a candidate cannot silently
                # introduce a new error or replace an existing one unnoticed.
                results[name] = {"error": {"errno": error.args[0], "message": error.args[1]}}
            timings[name] = (time.perf_counter() - started) * 1000
            print(name, results[name] if isinstance(results[name], dict) else len(results[name]),
                  round(timings[name], 3), flush=True)
            if "/implicit/20/0" in name:
                try:
                    cursor.execute("EXPLAIN " + sql)
                    plans[name] = [[str(v) for v in row] for row in cursor.fetchall()]
                except pymysql.MySQLError as error:
                    plans[name] = {"error": {"errno": error.args[0], "message": error.args[1]}}
    connection.close()
    for name in results:
        if name.endswith("/count_offset") and isinstance(results[name], list):
            all_rows = results[name.replace("/count_offset", "/all")]
            if not isinstance(all_rows, list):
                continue
            full_count = len(all_rows)
            assert results[name] == [[str(min(100, max(0, full_count-4000)))]], name
    report = {"results": results, "elapsed_ms": timings, "plans": plans}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))["results"]
        changed = [name for name in sorted(set(results) | set(reference))
                   if results.get(name) != reference.get(name)]
        if changed:
            raise SystemExit("FAIL exact score/order mismatch: " + ", ".join(changed))
        print("PASS: all exact float bits, IDs and tie ordering match", flush=True)
    digest = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
    print("queries", len(results), "sha256", digest, flush=True)


if __name__ == "__main__":
    main()
