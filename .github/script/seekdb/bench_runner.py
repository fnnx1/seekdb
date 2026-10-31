#!/usr/bin/env python3
"""Start a seekdb instance for the FTS benchmark and leave it running.

``fts_large_bench.sh`` needs a live server, so this reuses the instance bring-up
routine from ``mysqltest_for_seekdb.py`` (create dirs, start, wait for readiness,
run deploy init SQL) without running any test case afterwards.

Usage:
    python3 .github/script/seekdb/bench_runner.py \
        --seekdb build_release/src/observer/seekdb \
        --obclient deps/3rd/u01/obclient/bin/obclient \
        --base-dir "$PWD/bench_out/instance"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mysqltest_for_seekdb as runner  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seekdb", required=True)
    parser.add_argument("--obclient", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--destroy", action="store_true")
    args = parser.parse_args(argv)
    # The bring-up routine treats these as Path objects (it reads .parent).
    args.seekdb = Path(args.seekdb).resolve()
    args.obclient = Path(args.obclient).resolve()
    args.base_dir = Path(args.base_dir).resolve()

    repo_root = HERE.parents[2]
    deploy_dir = repo_root / "tools" / "deploy"
    sdb_script = HERE / "sdb.py"

    runner.prepare_instance(args, repo_root, sdb_script, deploy_dir)
    print("[bench] seekdb ready at {}:{}".format(args.host, args.port))

    if args.destroy:
        runner.destroy_instance(sdb_script, args.base_dir, repo_root, check=False)
        print("[bench] destroyed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
