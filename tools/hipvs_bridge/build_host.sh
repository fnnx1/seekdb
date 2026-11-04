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

# Locate a plain host C++ compiler.  hipcc on ROCm 7.x injects
# `-x hip --offload-arch=<gfx>` by itself, so it is only the last resort and is
# then forced into C++ mode.
EXTRA_FLAGS=()
pick_cxx() {
  local c
  if [[ -n "${HOSTCXX:-}" ]]; then
    [[ -x "$HOSTCXX" ]] && { echo "$HOSTCXX"; return 0; }
    echo "HOSTCXX is set but not executable: $HOSTCXX" >&2
  fi
  for c in \
    "$ROCM_ROOT/lib/llvm/bin/clang++" \
    /opt/rocm/lib/llvm/bin/clang++ \
    /opt/rocm-*/lib/llvm/bin/clang++ \
    "$(command -v clang++ 2>/dev/null || true)" \
    "$(command -v g++ 2>/dev/null || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then echo "$c"; return 0; fi
  done
  return 1
}
if ! HOSTCXX=$(pick_cxx); then
  if [[ -x "$ROCM_ROOT/bin/hipcc" ]]; then
    HOSTCXX="$ROCM_ROOT/bin/hipcc"
    EXTRA_FLAGS+=(-x c++)
    echo "no host clang++ found; falling back to hipcc -x c++" >&2
  else
    echo "no host C++ compiler found" >&2
    exit 2
  fi
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

# Hard guarantee: no device code object may end up in the bridge.
if command -v roc-obj-ls >/dev/null 2>&1; then
  if roc-obj-ls "$OUTPUT" 2>&1 | grep -qiE 'gfx[0-9]|amdgcn'; then
    echo "ERROR: device code object found in $OUTPUT" >&2
    roc-obj-ls "$OUTPUT" >&2 || true
    exit 3
  fi
fi

echo "built $OUTPUT (host-only)"
