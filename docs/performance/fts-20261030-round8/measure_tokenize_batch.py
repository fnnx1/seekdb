"""Supplementary end-to-end server batch workload; never used as official score."""
import json
from pathlib import Path
import statistics
import sys
import time

import pymysql

root = Path(__file__).resolve().parent
label = sys.argv[1]
corpus = json.loads((root / 'baseline-expanded-tokenize.json').read_text())
connection = pymysql.connect(host='127.0.0.1', port=2893, user='root',
                             charset='utf8mb4', autocommit=True, read_timeout=180)
table = 'fts_round8_batch_probe.docs'
variants = [
    ('ik_smart', 'ik', '[{"additional_args":[{"ik_mode":"smart"}]},{"output":"all"}]'),
    ('ik_max_word', 'ik', '[{"additional_args":[{"ik_mode":"max_word"}]},{"output":"all"}]'),
    ('beng', 'beng', '[{"output":"all"}]'),
]
output = {'kind': 'supplementary batch workload; not official benchmark or score',
          'corpus_sha256': corpus['corpus_sha256'], 'rows': len(corpus['cases']),
          'warmups': 3, 'samples': 9, 'metrics': {}}
try:
    with connection.cursor() as cursor:
        cursor.execute('CREATE DATABASE fts_round8_batch_probe')
        cursor.execute(f'CREATE TABLE {table}(id INT PRIMARY KEY, content VARCHAR(8192)) '
                       'DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci')
        cursor.executemany(f'INSERT INTO {table} VALUES (%s, %s)',
                           [(item['id'], item['text']) for item in corpus['cases']])
        cursor.execute('SET ob_query_timeout=180000000')
        for name, parser, properties in variants:
            sql = f'SELECT SUM(LENGTH(TOKENIZE(content, %s, %s))) FROM {table}'
            samples = []
            checksums = []
            for index in range(output['warmups'] + output['samples']):
                start = time.perf_counter()
                cursor.execute(sql, (parser, properties))
                result = int(cursor.fetchone()[0])
                elapsed = time.perf_counter() - start
                checksums.append(result)
                if index >= output['warmups']:
                    samples.append(elapsed)
            assert len(set(checksums)) == 1, checksums
            output['metrics'][name] = {'samples_sec': samples,
                                       'mean_sec': statistics.mean(samples),
                                       'stdev_sec': statistics.stdev(samples),
                                       'output_bytes': checksums[0]}
        cursor.execute('DROP DATABASE fts_round8_batch_probe')
finally:
    connection.close()
path = root / f'{label}-tokenize-batch.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output))
