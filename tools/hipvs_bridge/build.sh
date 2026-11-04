#!/usr/bin/env bash
# Platform build entry: build the host-only cuVS brute-force bridge.
# The implementation lives in build_host.sh (no -x hip, no --offload-arch,
# no device code object).  Keeping this wrapper means the platform's
# tools/hipvs_bridge/build.sh invocation automatically gets the host build.
set -Eeuo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/build_host.sh" "$@"
