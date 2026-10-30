"""Run native SQL regressions against a local seekdb instance.

python test_file_sql_runtime.py --directory <server-visible-test-directory>
Requires pymysql and pyarrow (fixture generator only). Creates a unique child
directory and removes only that directory at the end.
"""
import argparse
from datetime import datetime
from pathlib import Path
import tempfile
import uuid
import pymysql
import pyarrow as pa
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('--directory', required=True)
p.add_argument('--port', type=int, default=2881)
args = p.parse_args()
parent = Path(args.directory).resolve()
parent.mkdir(parents=True, exist_ok=True)
db = pymysql.connect(host='127.0.0.1', port=args.port, user='root', password='',
                     database='oceanbase', charset='utf8mb4', autocommit=True)
cur = db.cursor()
def query(sql):
    cur.execute(sql)
    return cur.fetchall()
def error(sql, code=None):
    try:
        query(sql)
    except pymysql.MySQLError as e:
        assert code is None or e.args[0] == code, (sql, e)
    else:
        raise AssertionError('expected failure: ' + sql)
with tempfile.TemporaryDirectory(prefix='seekdb_sql_', dir=parent) as tmp:
    root = Path(tmp).resolve()
    assert root.parent == parent
    def path(name):
        return (root / name).as_posix()
    (root / 'data.csv').write_text('id,s,n\n1,"a""b",1\n2,"",\n3,"NULL",3\n', encoding='utf-8')
    (root / 'data.jsonl').write_text('{"id":1,"label":"one"}\n{"id":3,"label":"three"}\n', encoding='utf-8')
    pq.write_table(pa.table({'id': [1, 2, 3], 'n': [5, None, 7]}), root / 'data.parquet',
                   data_page_version='1.0', row_group_size=2, compression='snappy')
    assert query('select @@session.enable_file_sql') == ((0,),)
    error(f"copy (select 1) to '{path('off.csv')}'", 1235)
    query('set session enable_file_sql=ON')
    (root / 'seconds.csv').write_text('t\n2026-08-01T10:11:12\n', encoding='utf-8')
    assert query(f"select t from read_csv('{path('seconds.csv')}')") == ((datetime(2026,8,1,10,11,12),),)
    assert cur.description[0][5] == 0, cur.description
    (root / 'fractional.csv').write_text(
        'id,t\n1,2026-08-01 10:11:12\n2,2026-08-01T10:11:12.1234\n3,2026-08-01 10:11:12.001234\n', encoding='utf-8')
    assert query(f"select t from read_csv('{path('fractional.csv')}') order by id") == (
        (datetime(2026,8,1,10,11,12),), (datetime(2026,8,1,10,11,12,123400),),
        (datetime(2026,8,1,10,11,12,1234),))
    assert cur.description[0][5] == 6, cur.description
    (root / 'fractional.jsonl').write_text(
        '{"id":1,"t":"2026-08-01"}\n{"id":2,"t":"2026-08-01T10:11:12.1234"}\n{"id":3}\n', encoding='utf-8')
    assert query(f"select t from read_json('{path('fractional.jsonl')}') order by id") == (
        (datetime(2026,8,1),), (datetime(2026,8,1,10,11,12,123400),), (None,))
    assert cur.description[0][5] == 4, cur.description
    assert query(f"select id,n from read_parquet('{path('data.parquet')}') order by id") == ((1,5),(2,None),(3,7))
    assert query(f"select c.id,j.label from read_csv('{path('data.csv')}') c left join read_json('{path('data.jsonl')}') j on c.id=j.id order by c.id") == ((1,'one'),(2,None),(3,'three'))
    assert query(f"select count(*) from read_csv('{path('data.csv')}') c join read_parquet('{path('data.parquet')}') p on c.id=p.id") == ((3,),)
    rows = query(f"describe select id as renamed,n from read_parquet('{path('data.parquet')}')")
    assert [r[:2] for r in rows] == [('renamed','BIGINT'),('n','BIGINT')], rows
    error(f"describe select missing from read_csv('{path('data.csv')}')", 1054)
    query(f"explain select * from read_csv('{path('data.csv')}')")
    assert cur.description[0][0] != 'column_name', cur.description
    query('explain select 1')
    query(f"copy (select id,s,n from read_csv('{path('data.csv')}') order by id) to '{path('copy.csv')}' (format csv, header, nullstr 'NULL')")
    assert query(f"select * from read_csv('{path('copy.csv')}',nullstr='NULL') order by id") == ((1,'a"b',1),(2,'',None),(3,'NULL',3))
    saved = (root/'copy.csv').read_bytes()
    error(f"copy (select 999) to '{path('copy.csv')}'")
    assert (root/'copy.csv').read_bytes() == saved
    query(f"copy (select 1 as id where 0) to '{path('empty.csv')}' (header)")
    assert (root/'empty.csv').read_text() == '"id"\n'
    query(f"copy (select 1 as id,'a;b' as s,NULL as n) to '{path('custom.csv')}' (header, delimiter ';',quote '|',escape '|',nullstr 'NA')")
    assert query(f"select * from read_csv('{path('custom.csv')}',delimiter=';',quote='|',escape='|',nullstr='NA')") == ((1,'a;b',None),)
    query(f"copy (select 1 as id) to '{path('noheader.csv')}' (header false)")
    assert (root/'noheader.csv').read_text() == '1\n'
    table_name = 'copy_regression_' + uuid.uuid4().hex
    query(f'create table {table_name}(id int, s varchar(30))')
    try:
        query(f"insert into {table_name} values(1,'test')")
        query(f"copy {table_name} to '{path('table.csv')}' (header)")
        assert query(f"select * from read_csv('{path('table.csv')}')") == ((1,'test'),)
    finally:
        query(f'drop table {table_name}')
    # Two scalar-subquery rows force an execution error, after the sink opens.
    error(f"copy (select (select id from read_csv('{path('data.csv')}'))) to '{path('failed.csv')}'")
    assert not (root/'failed.csv').exists()
    assert not list(root.glob('*.seekdb-copy-*.tmp'))
    print('PASS native CSV/JSONL/Parquet, DATETIME precision, JOIN, DESCRIBE, EXPLAIN, COPY options/empty/table/rollback/no-overwrite')
db.close()
