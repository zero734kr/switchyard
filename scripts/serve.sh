#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "$root" python "$root/scripts/switchyard.py" "$@"
