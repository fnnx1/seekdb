#!/usr/bin/env bash
# Pre-submission local checks.  Run from the repository root.
#   ./scripts/check.sh
#   RUN_DOCKER=1 ./scripts/check.sh   # also build and protocol-test the image
set -euo pipefail
cd "$(dirname "$0")/.."

# The original plain-SQL harness is kept as a fast regression suite; the
# formal submission path is the JSONL protocol exercised below.
export MVDB_LEGACY_STDIO=1

echo "== clean build =="
make clean
make

echo "== legacy dynamic tests =="
python3 tests/test_minivector.py

echo "== exact pruning regression =="
python3 tests/exact_pruning_test.py

echo "== SELECT template regression =="
python3 tests/select_plan_test.py

echo "== VARCHAR equality indexes and grouped scans =="
python3 tests/varchar_filter_test.py

echo "== durable WAL alignment and recovery =="
python3 tests/wal_alignment_test.py

echo "== immediate acknowledged-commit crash recovery =="
MVDB_SYNC_COMMIT=0 python3 tests/commit_crash_test.py
MVDB_SYNC_COMMIT=1 python3 tests/commit_crash_test.py

echo "== commit synchronization policy and interrupted checkpoints =="
python3 tests/commit_sync_policy_test.py

echo "== WAL cache advice and acknowledged-commit recovery =="
python3 tests/wal_cache_advice_test.py

echo "== legacy protocol tests =="
python3 tests/protocol_test.py

echo "== fuzz smoke =="
python3 tests/fuzz_smoke.py

echo "== large-data smoke =="
python3 tests/large_data_test.py

echo "== file-backed scan transactions and crash recovery =="
python3 tests/spill_scan_test.py

echo "== ASan/UBSan legacy tests =="
SAN_BIN="$(mktemp -u /tmp/minivectordb-san.XXXXXX)"
trap 'rm -f "$SAN_BIN"' EXIT
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" src/minivectordb.cpp
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/test_minivector.py
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/exact_pruning_test.py
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/select_plan_test.py
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/varchar_filter_test.py
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/wal_alignment_test.py

echo "== ASan/UBSan quantized dot product tests =="
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" tests/quant_dot_test.cpp
ASAN_OPTIONS=detect_leaks=1 "$SAN_BIN"

echo "== ASan/UBSan SIMD lanes and decimal parser =="
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" tests/search_kernel_test.cpp
ASAN_OPTIONS=detect_leaks=1 "$SAN_BIN"

echo "== ASan/UBSan compact cell ownership and snapshot tests =="
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" tests/stored_value_test.cpp
ASAN_OPTIONS=detect_leaks=1 "$SAN_BIN"

echo "== ASan/UBSan file-backed read windows =="
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter -pthread \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" tests/vector_store_read_test.cpp
ASAN_OPTIONS=detect_leaks=1 "$SAN_BIN"

echo "== optional clang-tidy =="
if [[ "${RUN_TIDY:-0}" == "1" ]]; then
  if command -v clang-tidy >/dev/null 2>&1; then
    TIDY_OUT="$(mktemp /tmp/minivectordb-tidy.XXXXXX)"
    clang-tidy src/minivectordb.cpp --checks='-*,clang-analyzer-*,bugprone-*' -- -std=c++17 2>&1 | tee "$TIDY_OUT"
    if grep -q 'error:' "$TIDY_OUT"; then rm -f "$TIDY_OUT"; exit 1; fi
    rm -f "$TIDY_OUT"
  else
    echo "clang-tidy not found; skipped"
  fi
fi

unset MVDB_LEGACY_STDIO

echo "== official JSONL handshake (host) =="
rm -rf protocol-results/check-host-handshake protocol-results/check-host-public
mkdir -p protocol-results
CMD="[\"$PWD/minivectordb\",\"{data_dir}\"]"
python3 protocol_check/run.py --command "$CMD" --handshake-only \
  --output protocol-results/check-host-handshake

echo "== official JSONL public cases (host) =="
python3 protocol_check/run.py --command "$CMD" \
  --output protocol-results/check-host-public

echo "== whitespace/static check =="
git diff --check
git diff --cached --check

if [[ "${RUN_DOCKER:-0}" == "1" ]]; then
  echo "== docker build =="
  docker build -t minivectordb:check .
  rm -rf protocol-results/check-docker-handshake protocol-results/check-docker-public
  echo "== official JSONL handshake (Docker) =="
  python3 protocol_check/run.py --image minivectordb:check --handshake-only \
    --output protocol-results/check-docker-handshake
  echo "== official JSONL public cases (Docker) =="
  python3 protocol_check/run.py --image minivectordb:check \
    --output protocol-results/check-docker-public
fi

echo "ALL CHECKS PASSED"
