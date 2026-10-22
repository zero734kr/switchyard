#!/usr/bin/env bash
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config.yaml}"
die() { echo "error: $*" >&2; exit 1; }
load_profile() {
    local resolved
    local config_python=${CONFIG_PYTHON:-$HOME/.venvs/vllm/bin/python}
    local args=("$1" --config "$CONFIG_FILE")
    [[ -z "${SWITCHYARD_MODEL:-}" ]] || args+=(--model "$SWITCHYARD_MODEL")
    resolved="$("$config_python" "$PROJECT_ROOT/scripts/config.py" "${args[@]}")" || exit 1
    eval "$resolved"
}
setup_cluster_environment() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside an existing Slurm allocation"
    [[ -d "$MODEL_PATH" ]] || die "model directory not found: $MODEL_PATH"
    source "$PROJECT_ROOT/recipes/jarvis.sh"
    switchyard_runtime "$CUDA_HOME" "$GCC_MODULE" "$CUDA_MODULE"
    source "$DYNAMO_VENV/bin/activate"
}
