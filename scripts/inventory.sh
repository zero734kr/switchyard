#!/usr/bin/env bash
# CPU-side inventory only. Does not start inference or benchmark GPU transfers.
set -euo pipefail
hostname
printf '\nGPU inventory (metadata only)\n'
nvidia-smi --query-gpu=name,memory.total,pci.bus_id --format=csv
nvidia-smi topo -m
printf '\nNetwork interfaces\n'
ip -brief address
printf '\nRDMA devices\n'
if command -v ibstat >/dev/null; then ibstat; fi
if command -v ibdev2netdev >/dev/null; then ibdev2netdev; fi
if command -v ucx_info >/dev/null; then
    ucx_info -d | awk '/Transport:|Device:|Type:|bandwidth:|memory types:/'
fi
for path in /sys/class/infiniband/*/ports/*/{rate,state}; do
    [[ ! -f "$path" ]] || { printf '%s: ' "$path"; cat "$path"; }
done
