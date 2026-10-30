"""Cross-check the standalone native reader using Arrow-generated test inputs.

Run: python test_file_parquet_reader.py <compiled-reader-executable>
Requires pyarrow only for test fixture generation, never for seekdb execution.
"""
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="seekdb_parquet_test_", dir=Path(exe).parent) as tmp:
    root = Path(tmp).resolve()
    assert root.parent == Path(exe).parent
    arrays = [
        pa.array([None if i % 7 == 0 else i - 1024 for i in range(6000)], type=pa.int32()),
        pa.array([None if i % 9 == 0 else 2**40 + i for i in range(6000)], type=pa.int64()),
        pa.array([None if i % 11 == 0 else i / 4 for i in range(6000)], type=pa.float64()),
        pa.array([None if i % 13 == 0 else i % 2 == 0 for i in range(6000)]),
        pa.array([None if i % 17 == 0 else "label" + str(i % 19) for i in range(6000)]),
    ]
    table = pa.table(arrays, names=["i", "big", "d", "b", "s"])
    values = [a.to_pylist() for a in arrays]
    for version, compression, dictionary in itertools.product(["1.0", "2.0"], ["NONE", "snappy"], [False, True]):
        path = root / "fixture.parquet"
        pq.write_table(table, path, data_page_version=version, compression=compression,
                       use_dictionary=dictionary, row_group_size=2049, data_page_size=1024,
                       write_batch_size=128)
        result = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (version, compression, dictionary, result.stderr)
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        assert len(rows) == table.num_rows
        for n, row in enumerate(rows):
            for col, text in enumerate(row):
                expected = values[col][n]
                if expected is None:
                    assert text == r"\N", (n, col, text)
                elif col in (0, 1, 3):
                    assert int(text) == expected, (n, col, text, expected)
                elif col == 2:
                    assert float(text) == expected
                else:
                    assert text == expected
        print("PASS", version, compression, "dictionary=" + str(dictionary))
    # Empty row groups/files and required columns.
    for table in [pa.table({"x": pa.array([], type=pa.int32())}),
                  pa.Table.from_arrays([pa.array([1, 2, 3])], schema=pa.schema([pa.field("x", pa.int64(), nullable=False)]))]:
        pq.write_table(table, root / "edge.parquet")
        result = subprocess.run([exe, str(root / "edge.parquet")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    # Invalid footer and nested types must produce errors, never partial rows.
    good = (root / "fixture.parquet").read_bytes()
    for bad in [good[:24], good[:-8] + b"\xff\xff\xff\x7fPAR1"]:
        (root / "bad.parquet").write_bytes(bad)
        assert subprocess.run([exe, str(root / "bad.parquet")], capture_output=True).returncode == 1
    pq.write_table(pa.table({"nested": [[1, 2], [3]]}), root / "bad.parquet")
    assert subprocess.run([exe, str(root / "bad.parquet")], capture_output=True).returncode == 1
    print("PASS empty/required/malformed/nested")
