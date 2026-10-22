#!/usr/bin/env bash
# Compatibility entry point: starts services only; benchmarks are separate.
set -euo pipefail
exec "$(dirname -- "$0")/serve.sh" run "$@"
