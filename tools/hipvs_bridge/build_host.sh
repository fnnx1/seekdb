#!/usr/bin/env bash
# Host-only build of the seekdb cuVS bridge.
#
# Deliberately does NOT pass `-x hip` and does NOT pass `--offload-arch`:
# seekdb_cuvs_bridge_host.c contains no __global__ kernels, so no device code
# object is produced and the platform build does not pay fat/device-code
# compile cost.  (Measured on the eval GPU host: ~0.98 s host-only vs ~4.47 s
# for the fat gfx1100 device build of seekdb_cuvs_bridge.c.)
#
# Usage: ./build_host.sh [output.so]
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HOSTCXX=${HOSTCXX:-$ROCM_ROOT/lib/llvm/bin/clang++}
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}
SRC=${SRC:-$SCRIPT_DIR/seekdb_cuvs_bridge_host.c}

[[ -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]] || {
  echo "hipVS headers are missing under $HIPVS_ROOT/include" >&2
  exit 2
}
[[ -f "$HIPVS_ROOT/lib/libcuvs_c.so" ]] || {
  echo "libcuvs_c.so is missing under $HIPVS_ROOT/lib" >&2
  exit 2
}

# Plain host compiler first; fall back to hipcc forced into C++ mode (hipcc
# would otherwise add `-x hip --offload-arch=<gfx>` on its own).
EXTRA_FLAGS=()
if [[ -x "$HOSTCXX" ]]; then
  :
elif [[ -x "$ROCM_ROOT/bin/hipcc" ]]; then
  HOSTCXX="$ROCM_ROOT/bin/hipcc"
  EXTRA_FLAGS+=(-x c++)
else
  echo "no host C++ compiler found (HOSTCXX=$HOSTCXX)" >&2
  exit 2
fi
echo "host-only build with $HOSTCXX"

mkdir -p "$(dirname "$OUTPUT")"

"$HOSTCXX" \
  -shared \
  -fPIC \
  -O2 \
  -std=c++17 \
  -pthread \
  -D__HIP_PLATFORM_AMD__ \
  -I"$SCRIPT_DIR" \
  -I"$HIPVS_ROOT/include" \
  -I"$ROCM_ROOT/include" \
  "${EXTRA_FLAGS[@]}" \
  "$SRC" \
  -L"$HIPVS_ROOT/lib" \
  -L"$ROCM_ROOT/lib" \
  -Wl,-soname,libseekdb_cuvs_bridge.so \
  -Wl,-rpath,"$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -lcuvs_c \
  -lcuvs \
  -lamdhip64 \
  -o "$OUTPUT"

echo "built $OUTPUT (host-only)"
