#!/usr/bin/env bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config.json}"

die() {
    echo "error: $*" >&2
    exit 1
}

load_profile() {
    local profile=${1:?profile is required}
    local resolved
    resolved="$(python3 "$PROJECT_ROOT/scripts/config.py" "$profile" --config "$CONFIG_FILE")" || exit
    eval "$resolved"

    # Hardware settings remain calibration knobs because allocations differ.
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-$MODEL_PATH}"
    MTP_TOKENS="${MTP_TOKENS_OVERRIDE:-$MTP_TOKENS}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-$GPU_MEMORY_UTILIZATION}"
}

setup_cluster_environment() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside an active Slurm allocation"

    local actual_node=${SLURMD_NODENAME:-$(hostname -s)}
    if [[ "${ALLOW_NODE_MISMATCH:-0}" != 1 && "$actual_node" != "$EXPECTED_NODE" ]]; then
        die "profile $PROFILE expects $EXPECTED_NODE, current node is $actual_node (set ALLOW_NODE_MISMATCH=1 to override)"
    fi
    [[ -d "$MODEL_PATH" ]] || die "model directory not found: $MODEL_PATH"
    [[ -f "$DYNAMO_VENV/bin/activate" ]] || die "virtual environment not found: $DYNAMO_VENV"

    if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
    type module >/dev/null 2>&1 || die "environment-modules command is unavailable"

    module purge
    module load "$GCC_MODULE"
    module load "$CUDA_MODULE"
    # shellcheck disable=SC1090
    source "$DYNAMO_VENV/bin/activate"

    export CUDA_HOME
    export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"

    local libstdcxx gcc_libdir
    libstdcxx="$(g++ -print-file-name=libstdc++.so.6)"
    gcc_libdir="$(dirname "$(readlink -f "$libstdcxx")")"
    export LD_LIBRARY_PATH="$gcc_libdir:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

    export FLASHINFER_MAX_NUM_TOKENS="$MAX_NUM_BATCHED_TOKENS"
    export TORCH_FLOAT32_MATMUL_PRECISION=high
    export VLLM_USE_FLASHINFER_SAMPLER=1
    export VLLM_SSM_CONV_STATE_LAYOUT=DS
    export UCX_TLS=cuda_ipc,cuda_copy,tcp
    export UCX_RCACHE_MAX_UNRELEASED=1024

    unset UCX_NET_DEVICES UCX_IB_REG_METHODS NCCL_IB_HCA NCCL_IB_DISABLE

    export DYN_DISCOVERY_BACKEND=file
    export DYN_FILE_KV="${DYN_FILE_KV:-$HOME/tmp/dynamo-${SLURM_JOB_ID}-${PROFILE}}"
    mkdir -p "$DYN_FILE_KV"

    ulimit -d unlimited
    ulimit -l unlimited
}

write_manifest() {
    local destination=${1:?manifest destination is required}
    {
        echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "profile=$PROFILE"
        echo "topology=$TOPOLOGY"
        echo "expected_node=$EXPECTED_NODE"
        echo "actual_node=${SLURMD_NODENAME:-$(hostname -s)}"
        echo "slurm_job_id=$SLURM_JOB_ID"
        echo "accelerator=$ACCELERATOR"
        echo "model_path=$MODEL_PATH"
        echo "model_name=$MODEL_NAME"
        echo "mtp_tokens=$MTP_TOKENS"
        echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
        echo "max_model_len=$MAX_MODEL_LEN"
        echo "max_num_seqs=$MAX_NUM_SEQS"
        echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        echo
        python3 -m pip show ai-dynamo vllm nixl 2>/dev/null || true
        echo
        nvidia-smi -L 2>/dev/null || true
        echo
        nvidia-smi topo -m 2>/dev/null || true
    } >"$destination"
}
