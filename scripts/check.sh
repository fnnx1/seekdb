#!/usr/bin/env bash
# Pre-submission local checks.  Run from the repository root.
#   ./scripts/check.sh
#   RUN_DOCKER=1 ./scripts/check.sh   # also build and smoke-test the image
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== clean build =="
make clean
make

echo "== dynamic tests =="
python3 tests/test_minivector.py

echo "== protocol tests =="
python3 tests/protocol_test.py

echo "== ASan/UBSan tests =="
SAN_BIN="$(mktemp -u /tmp/minivectordb-san.XXXXXX)"
trap 'rm -f "$SAN_BIN"' EXIT
g++ -O1 -g -std=c++17 -Wall -Wextra -Wno-unused-parameter \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    -o "$SAN_BIN" src/minivectordb.cpp
MINIVECTORDB="$SAN_BIN" ASAN_OPTIONS=detect_leaks=1 python3 tests/test_minivector.py

echo "== whitespace/static check =="
git diff --check
git diff --cached --check

if [[ "${RUN_DOCKER:-0}" == "1" ]]; then
  echo "== docker build =="
  docker build -t minivectordb:check .
  D="$(mktemp -d /tmp/minivectordb-docker.XXXXXX)"
  trap 'rm -rf "$D"' EXIT
  echo "== docker run/restart =="
  printf "%s\n" \
    "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(2));" \
    "INSERT INTO t VALUES (1,'[1,2]'),(2,'[3,4]');" \
    "SELECT id FROM t ORDER BY L2_DISTANCE(v,'[1,2]');" \
    | docker run --rm -i -v "$D:/data" minivectordb:check > "$D/out1.txt"
  grep -q '^1$' "$D/out1.txt"
  grep -q '^2$' "$D/out1.txt"
  printf "SELECT id FROM t ORDER BY id;\n" \
    | docker run --rm -i -v "$D:/data" minivectordb:check > "$D/out2.txt"
  grep -q '^1$' "$D/out2.txt"
  grep -q '^2$' "$D/out2.txt"
  echo "docker smoke OK"
fi

echo "ALL CHECKS PASSED"
