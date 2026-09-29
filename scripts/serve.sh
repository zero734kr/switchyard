#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$PROJECT_ROOT/scripts/common.sh"

PROFILE_NAME=${1:?usage: scripts/serve.sh PROFILE [RUN_DIR]}
RUN_DIR=${2:-"$PROJECT_ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)-$PROFILE_NAME"}
load_profile "$PROFILE_NAME"
setup_cluster_environment
mkdir -p "$RUN_DIR"
write_manifest "$RUN_DIR/manifest.txt"
cp "$CONFIG_FILE" "$RUN_DIR/config.json"

pids=()
cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill -- "-$pid" 2>/dev/null || true
    done
    wait "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

frontend=(
    python3 -m dynamo.frontend
    --http-host 0.0.0.0
    --http-port "$FRONTEND_PORT"
    --discovery-backend file
    --dyn-chat-processor vllm
    --trust-remote-code
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
)
(
    unset DYN_SYSTEM_PORT DYN_TCP_RPC_PORT
    exec setsid "${frontend[@]}"
) >"$RUN_DIR/frontend.log" 2>&1 &
pids+=("$!")

capture_sizes=()
for size in 1 2 4 8 16 32 64 128 256 512; do
    (( size <= MAX_NUM_SEQS )) && capture_sizes+=("$size")
done
capture_sizes_csv=$(IFS=,; echo "${capture_sizes[*]}")
compile_config="{\"cudagraph_capture_sizes\":[${capture_sizes_csv}],\"pass_config\":{\"fuse_allreduce_rms\":false}}"

common_worker=(
    python3 -m dynamo.vllm
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --tensor-parallel-size 1
    --dtype "$MODEL_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --discovery-backend file
    --reasoning-parser qwen3
    --trust-remote-code
    --performance-mode throughput
    --enable-chunked-prefill
    --enable-prefix-caching
    --gdn-prefill-backend flashinfer
    --compilation-config "$compile_config"
    --language-model-only
)
if (( MTP_TOKENS > 0 )); then
    common_worker+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
fi

(
    export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"
    export DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT"
    exec setsid "${common_worker[@]}" --disaggregation-mode prefill \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$KV_EVENT_PORT\",\"enable_kv_cache_events\":true}"
) >"$RUN_DIR/prefill.log" 2>&1 &
pids+=("$!")

IFS=, read -r -a decode_gpus <<<"$DECODE_GPUS"
IFS=, read -r -a system_ports <<<"$DECODE_SYSTEM_PORTS"
IFS=, read -r -a nixl_ports <<<"$DECODE_NIXL_PORTS"
for index in "${!decode_gpus[@]}"; do
    (
        export CUDA_VISIBLE_DEVICES="${decode_gpus[$index]}"
        export DYN_SYSTEM_PORT="${system_ports[$index]}"
        export VLLM_NIXL_SIDE_CHANNEL_PORT="${nixl_ports[$index]}"
        exec setsid "${common_worker[@]}" --disaggregation-mode decode \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    ) >"$RUN_DIR/decode-$index.log" 2>&1 &
    pids+=("$!")
done

echo "[$PROFILE] starting ${#pids[@]} processes; logs: $RUN_DIR"
wait -n "${pids[@]}"
die "$PROFILE process exited; inspect $RUN_DIR"
