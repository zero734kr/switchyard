#!/usr/bin/env bash
# Shared runtime setup; all configurable values come from config.yaml.
switchyard_runtime() {
    local cuda_home=$1
    shift
    if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
    module load "$@"
    export CUDA_HOME="$cuda_home"
    local libstdcxx
    libstdcxx=$(g++ -print-file-name=libstdc++.so.6)
    export LD_LIBRARY_PATH="$(dirname "$(readlink -f "$libstdcxx")"):$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
}
