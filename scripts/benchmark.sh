#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$PROJECT_ROOT/scripts/common.sh"

PROFILE_NAME=${1:?usage: scripts/benchmark.sh PROFILE BASE_URL RESULT_DIR TARGET}
BASE_URL=${2:?base URL is required}
RESULT_DIR=${3:?result directory is required}
TARGET=${4:?target label is required}
load_profile "$PROFILE_NAME"
setup_cluster_environment
mkdir -p "$RESULT_DIR"

run_case() {
    local name=$1 input=$2 output=$3 concurrency=$4 prompts=$5
    vllm bench serve \
        --backend openai \
        --base-url "$BASE_URL" \
        --endpoint /v1/completions \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_PATH" \
        --trust-remote-code \
        --dataset-name random \
        --input-len "$input" \
        --output-len "$output" \
        --max-concurrency "$concurrency" \
        --num-prompts "$prompts" \
        --num-warmups 1 \
        --temperature 0 \
        --ignore-eos \
        --save-result \
        --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "$name.json" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metadata scenario="$name" target="$TARGET"
}

# Two short cases expose prefill and decode behavior without consuming the night.
run_case prefill-heavy 8192 256 4 12
run_case decode-heavy 512 2048 8 24
