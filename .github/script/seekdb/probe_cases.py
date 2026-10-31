#!/usr/bin/env python3
"""Run a subset of the seekdb mysqltest suite and print the expected/actual diff.

The official runner (``mysqltest_for_seekdb.py``) shards every case in the repo,
which is far too slow when only one suite is under investigation.  This wrapper
reuses its instance bootstrap and per-case execution, keeps the same environment,
and then prints a unified diff between the expected ``.result`` file and the
``.reject`` file mysqltest produced.

Usage:
    python3 probe_cases.py --prefix ai_function. --seekdb <bin> \
        --obclient <bin> --mysqltest <bin> --work-dir probe_out
"""

from __future__ import print_function

import argparse
import difflib
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mysqltest_for_seekdb as runner  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True,
                        help="comma separated case-name prefixes, e.g. ai_function.")
    parser.add_argument("--seekdb", required=True)
    parser.add_argument("--obclient", required=True)
    parser.add_argument("--mysqltest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--base-dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    args = parser.parse_args(argv)
    if not args.base_dir:
        args.base_dir = str(Path(args.work_dir).resolve() / "instance")
    for name in ("seekdb", "obclient", "mysqltest", "work_dir", "base_dir"):
        setattr(args, name, runner.absolute_path(getattr(args, name)))
    return args


def print_diff(expected_path, reject_path):
    try:
        expected = expected_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print("[DIFF] cannot read expected result: {}".format(exc))
        return
    try:
        actual = reject_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print("[DIFF] cannot read reject file: {}".format(exc))
        return
    lines = list(difflib.unified_diff(expected, actual, "expected", "actual", n=4, lineterm=""))
    if not lines:
        print("[DIFF] expected and actual are identical (whitespace only?)")
        return
    limit = 400
    for line in lines[:limit]:
        print(line)
    if len(lines) > limit:
        print("... {} more diff lines truncated".format(len(lines) - limit))


def print_observer_log(case_name, base_dir, destination):
    """Copy the instance logs before they are destroyed and expose the error chain.

    The mysqltest output only carries the user facing error code, so the server
    side log is the only place that shows which statement produced it.
    """
    try:
        runner.copy_instance_diagnostics(Path(base_dir), destination)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never mask the real failure
        print("[OBSERVER] failed to copy instance logs: {}".format(exc))
        return
    print("[OBSERVER] logs copied to {}".format(destination))
    keywords = ("4152", "null value", "ai_complete", "ai_embed", "ai_model_endpoint",
                "ERROR", "ret=-")
    for log_file in sorted(destination.rglob("*.log*")):
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        hits = [line for line in lines if any(key in line for key in keywords)]
        if not hits:
            continue
        print("--- {} ({} matching lines)".format(log_file.name, len(hits)))
        for line in hits[-60:]:
            print(line)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = HERE.parents[2]
    deploy_dir = repo_root / "tools" / "deploy"
    sdb_script = HERE / "sdb.py"

    prefixes = tuple(p.strip() for p in args.prefix.split(",") if p.strip())
    all_cases = runner.discover_cases(repo_root)
    selected = [case for case in all_cases if case.name.startswith(prefixes)]
    if not selected:
        print("[probe] no case matches {}".format(prefixes))
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.work_dir / "tmp"
    log_dir = args.work_dir / "log"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] {} case(s): {}".format(len(selected), ", ".join(c.name for c in selected)))
    failures = []
    error = None
    try:
        runner.prepare_instance(args, repo_root, sdb_script, deploy_dir)
        for case in selected:
            code, output = runner.run_case(args, deploy_dir, case, tmp_dir, log_dir)
            print("=" * 78)
            print("[CASE] {} -> exit {}".format(case.name, code))
            print(output.rstrip())
            reject = log_dir / (case.result_file.stem + ".reject")
            if code != 0:
                failures.append(case.name)
                print("-" * 78)
                if reject.exists():
                    print_diff(case.result_file, reject)
                else:
                    print("[DIFF] no reject file at {}".format(reject))
                print_observer_log(case.name,
                                   args.base_dir,
                                   args.work_dir / "diagnostics" / case.name)
    except Exception as exc:  # noqa: BLE001 - infrastructure errors must not be silent
        error = str(exc)
        print("[probe][ERROR] {}".format(error), file=sys.stderr)
        traceback.print_exc()
    finally:
        runner.destroy_instance(sdb_script, args.base_dir, repo_root, check=False)

    print("=" * 78)
    print("[probe] failed {}/{}: {}".format(len(failures), len(selected), ", ".join(failures) or "-"))
    if error:
        print("[probe] infrastructure error: {}".format(error))
        return 3
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
