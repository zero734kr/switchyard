#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H200_JOB_ID=${H200_JOB_ID:?export H200_JOB_ID for g213}
H100_JOB_ID=${H100_JOB_ID:?export H100_JOB_ID for g013}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

pids=()
cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

children_running() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -0 "$pid" 2>/dev/null || return 1
    done
}

wait_for() {
    local url=$1 label=$2
    local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-1200}))
    until curl --fail --silent --show-error "$url" >/dev/null 2>&1; do
        children_running || { echo "a service step exited while waiting for $label" >&2; return 1; }
        (( SECONDS < deadline )) || { echo "timeout waiting for $label" >&2; return 1; }
        sleep 5
    done
    echo "$label ready"
}

wait_for_model() {
    local url=$1 label=$2
    local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-1200}))
    until curl --fail --silent "$url/v1/models" | grep -q 'qwen3.6-27b'; do
        children_running || { echo "a service step exited while waiting for $label" >&2; return 1; }
        (( SECONDS < deadline )) || { echo "timeout waiting for $label" >&2; return 1; }
        sleep 5
    done
    echo "$label ready"
}

DYN_FILE_KV="$HOME/tmp/dynamo-$RUN_ID-h200" \
srun --jobid="$H200_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$PROJECT_ROOT/scripts/serve.sh" h200-pd "$RUN_DIR/h200-pd" \
    >"$RUN_DIR/h200-pd.srun.log" 2>&1 &
pids+=("$!")
DYN_FILE_KV="$HOME/tmp/dynamo-$RUN_ID-h100" \
srun --jobid="$H100_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$PROJECT_ROOT/scripts/serve.sh" h100-1p2d "$RUN_DIR/h100-1p2d" \
    >"$RUN_DIR/h100-1p2d.srun.log" 2>&1 &
pids+=("$!")

wait_for_model http://g213:18000 "H200 P/D backend"
wait_for_model http://g013:18000 "H100 1P2D backend"

srun --jobid="$H200_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$HOME/.venvs/vllm/bin/python3" "$PROJECT_ROOT/scripts/gateway.py" \
    >"$RUN_DIR/gateway.log" 2>&1 &
pids+=("$!")
wait_for http://g213:19000/health "heterogeneous gateway"

srun --jobid="$H200_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$PROJECT_ROOT/scripts/benchmark.sh" h200-pd http://g213:18000 "$RUN_DIR/results/h200" h200-pd \
    2>&1 | tee "$RUN_DIR/benchmark-h200.log"
srun --jobid="$H100_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$PROJECT_ROOT/scripts/benchmark.sh" h100-1p2d http://g013:18000 "$RUN_DIR/results/h100" h100-1p2d \
    2>&1 | tee "$RUN_DIR/benchmark-h100.log"
srun --jobid="$H200_JOB_ID" --overlap --nodes=1 --ntasks=1 \
    "$PROJECT_ROOT/scripts/benchmark.sh" h200-pd http://g213:19000 "$RUN_DIR/results/mixed" heterogeneous-pool \
    2>&1 | tee "$RUN_DIR/benchmark-mixed.log"
curl --fail --silent http://g213:19000/gateway/status >"$RUN_DIR/gateway-status.json"
echo "experiment complete: $RUN_DIR"
