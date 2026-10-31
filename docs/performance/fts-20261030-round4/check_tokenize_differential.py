"""Record/replay exact TOKENIZE outputs using a deterministic, independent corpus.

Run serially on the baseline and candidate binaries. Each run uses a uniquely
named VARCHAR fixture table and removes it afterward; existing benchmark tables
are never read or changed. --reference compares input, collation, options, error
outcomes and the original JSON text, including token-array order and frequencies.
TOKENIZE(output=all) exposes tokens/doc_len, but no offsets; none are synthesized.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

import pymysql


ROOT = Path(__file__).resolve().parents[3]
VARIANTS = (
    ("ik_smart", "ik", '[{"additional_args":[{"ik_mode":"smart"}]},{"output":"all"}]'),
    ("ik_max_word", "ik", '[{"additional_args":[{"ik_mode":"max_word"}]},{"output":"all"}]'),
    ("beng", "beng", '[{"additional_args":[{"min_token_size":1},{"max_token_size":84}]},{"output":"all"}]'),
)


def dictionary_words(path):
    source = path.read_text(encoding="utf-8")
    marker = "static const char *main_dic[] = {"
    block = source.split(marker, 1)[1].split("};", 1)[0]
    words = re.findall(r'^\s*"([^"\\]*)",?\s*$', block, flags=re.MULTILINE)
    if len(words) < 100_000:
        raise ValueError("Unexpected IK dictionary source: insufficient words")
    # Draw from throughout the dictionary, including either side of the normal
    # 50,000-word range divisions. These are test inputs, never expected output.
    boundary_words = []
    for boundary in range(50_000, len(words), 50_000):
        boundary_words.extend(words[max(0, boundary - 4):boundary + 5])
    return words, boundary_words


def make_cases(seed, count, dictionary):
    rng = random.Random(seed)
    words, boundary_words = dictionary_words(dictionary)
    english = ["alpha", "BETA", "Gamma", "database", "Search", "INDEX", "context",
               "tokenizer", "latency", "a", "ABOUT", "the", "www", "willow",
               "ab", "abouta", "abcdefghi", "abcdefghijkl", "abcdefghijklm"]
    prefixes = ["中华人民共和国", "中华人民共和国全国人民代表大会常务委员会",
                "全国人民代表大会", "人民大会堂", "中华人民", "一一对应", "数据库索引"]
    unicode_parts = ["café", "résumé", "THE", "thé", "İS", "ıS", "Straße",
                     "STRASSE", "日本語", "カタカナ", "半角ｶﾀｶﾅ", "한국어",
                     "русский", "αβγ", "ＡＢＣ", "１２３", "𠮷", "😀", "e\u0301"]
    separators = ["", " ", "  ", "\t", "\n", "\r\n", ",", ".", ";", "。",
                  "，", "；", "!", "?", "-", "_", "@", "#", "\u3000", "\u00a0"]
    texts = [
        "", " ", "\t\r\n", "。；，!?", "a", "A", "the ABOUT and",
        "\x00", "alpha\x00beta\x00数据库", "'quoted' \\\"escaped\\\"",
        "中华人民共和国全国人民代表大会常务委员会人民大会堂",
        "中华人民共和国中华人民共和中华人民中华中",
        "a中华人民共和国b one数据库two 3.14公斤 １２３平方公里",
        "café CAFÉ cafe\u0301 THE thé thè thê thë résumé RESUME",
        "ＡＢＣ１２３ ｶﾀｶﾅ カタカナ 한국어 😀 𠮷\u200d😀",
        "a about an are as at be by com de en for from how i in is it la of on or that the this to was what when where who will with und www",
        "|".join(boundary_words),
        " ".join(english * 3),
        # More than sixteen disjoint chains, and >1,000-character parser batches.
        "中国 北京 上海 数据库。" * 45,
        "中华人民共和国数据库索引。" * 90,
        "中华人民共和国全国人民代表大会常务委员会" * 65,
        "alpha BETA gamma。中文分词 数据库索引\t" * 70,
    ]
    for index in range(count - len(texts)):
        flavor = index % 7
        parts = []
        for _ in range(rng.randint(2, 30)):
            if flavor == 0:
                # Cross dictionary ranges within the same active-state stream.
                part = rng.choice(boundary_words) if boundary_words else rng.choice(words)
            elif flavor == 1:
                prefix = rng.choice(prefixes)
                part = prefix[:rng.randint(1, len(prefix))]
            elif flavor == 2:
                part = rng.choice(english)
                part = part.upper() if rng.randrange(2) else part.lower()
            elif flavor == 3:
                part = rng.choice(unicode_parts)
            elif flavor == 4:
                part = rng.choice(words)
            elif flavor == 5:
                part = rng.choice(["3.14159", "123kg", "１２．３公斤", "三十六万",
                                   "123abc", "foo-bar", "email@example", "A_B"])
            else:
                part = rng.choice(prefixes + english + unicode_parts)
            if rng.randrange(7) == 0:
                part *= rng.randint(2, 5)
            parts.append(part)
            parts.append(rng.choice(separators))
        texts.append("".join(parts))
    if len(texts) != count or any(len(text) > 8192 for text in texts):
        raise ValueError("Fixture exceeds requested case count or VARCHAR capacity")
    return [{"id": index, "text": text,
             "collation": "utf8mb4_bin" if index % 2 else "utf8mb4_general_ci"}
            for index, text in enumerate(texts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2893)
    parser.add_argument("--user", default="root")
    parser.add_argument("--database", default="fts_predicate_regression")
    parser.add_argument("--seed", type=int, default=202610304)
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--dictionary", type=Path,
                        default=ROOT / "src/storage/fts/dict/ob_ik_dic.cpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if not 200 <= args.cases <= 300:
        parser.error("--cases must be between 200 and 300")
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    cases = make_cases(args.seed, args.cases, args.dictionary)
    corpus_json = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    report = {"schema_version": 1, "seed": args.seed, "case_count": len(cases),
              "corpus_sha256": hashlib.sha256(corpus_json.encode("utf-8")).hexdigest(),
              "comparison": "Exact original JSON text and error outcomes; no sorting or rounding",
              "offsets": "Not exposed by TOKENIZE(output=all); no synthesized offsets",
              "cases": []}
    if reference is not None and any(reference.get(key) != report[key]
                                     for key in ("schema_version", "seed", "case_count", "corpus_sha256")):
        raise SystemExit("Reference corpus differs; use the same seed, case count and dictionary source")
    table = "fts_tokenize_diff_" + uuid.uuid4().hex[:16]
    connection = pymysql.connect(host=args.host, port=args.port, user=args.user,
        password=os.environ.get("SEEKDB_TEST_PASSWORD", ""), database=args.database,
        charset="utf8mb4", autocommit=True, connect_timeout=10, read_timeout=180)
    created = False
    started = time.perf_counter()
    successful = 0
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET ob_query_timeout=180000000")
            cursor.execute(f"CREATE TABLE `{table}` (id INT PRIMARY KEY, content VARCHAR(8192)) "
                           "DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci")
            created = True
            cursor.executemany(f"INSERT INTO `{table}` VALUES (%s,%s)",
                               [(case["id"], case["text"]) for case in cases])
            for case in cases:
                result = {**case, "variants": {}}
                for name, ft_parser, properties in VARIANTS:
                    # A physical VARCHAR avoids the old baseline's LOB/REPEAT
                    # argument limitation and tests actual parser behavior.
                    sql = (f"SELECT TOKENIZE(content COLLATE {case['collation']}, %s, %s) "
                           f"AS tokens FROM `{table}` WHERE id=%s")
                    try:
                        cursor.execute(sql, (ft_parser, properties, case["id"]))
                        rows = cursor.fetchall()
                        if len(rows) != 1 or len(rows[0]) != 1:
                            raise AssertionError("Expected one TOKENIZE result row")
                        raw = rows[0][0]
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        if not isinstance(raw, str):
                            raise AssertionError("TOKENIZE did not return JSON text")
                        parsed = json.loads(raw)
                        if not isinstance(parsed, dict) or "tokens" not in parsed or "doc_len" not in parsed:
                            raise AssertionError("Unexpected TOKENIZE(output=all) shape")
                        result["variants"][name] = {"parser": ft_parser, "properties": properties,
                                                    "json_text": raw}
                        successful += 1
                    except pymysql.err.DatabaseError as exc:
                        result["variants"][name] = {"parser": ft_parser, "properties": properties,
                                                    "error": list(exc.args)}
                report["cases"].append(result)
                if (case["id"] + 1) % 40 == 0:
                    print(f"Recorded {case['id'] + 1}/{len(cases)} strings", flush=True)
    finally:
        if created:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP TABLE `{table}`")
            except pymysql.err.DatabaseError as exc:
                print(f"Fixture cleanup failed for {table}: {exc}", file=sys.stderr)
        connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = len(cases) * len(VARIANTS)
    if successful < total // 2:
        raise SystemExit(f"Only {successful}/{total} evaluations succeeded; inspect the output before comparing")
    if reference is not None:
        expected = reference.get("cases", [])
        mismatches = [case["id"] for case, old in zip(report["cases"], expected) if case != old]
        if len(expected) != len(report["cases"]):
            raise SystemExit("Reference result count differs")
        if mismatches:
            raise SystemExit(f"TOKENIZE differs in {len(mismatches)} strings; first IDs: {mismatches[:10]}")
        print(f"PASS: all {total} original JSON/error outcomes match the reference exactly")
    else:
        print(f"Recorded {total} outcomes: {successful} successful, {total - successful} SQL errors")
    print(f"Elapsed {time.perf_counter() - started:.3f}s; saved {args.output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
