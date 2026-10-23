# Deployment notes

`config.yaml` is the single source of runtime and placement settings. The `switchyard` CLI validates allocation ownership and node placement, snapshots the config for each run, and starts one Slurm step per service inside the declared job. It uses existing allocations; it does not submit, resize, or cancel jobs. `plan` prints the steps, `preflight` checks the runtime and model, and `control-check` starts the control plane and checks reachability without loading a model.

## Allocation and CPU/NUMA placement

Recipes can name several allocations on one node. The short form `node: job_id` names a job after its node; use `name: {node: ..., job: ...}` when that node has multiple jobs, then point each worker at its allocation. `gpu_ids` indexes that allocation's Slurm-visible `CUDA_VISIBLE_DEVICES`. Indices can repeat in separate allocations, but workers sharing one allocation must claim disjoint GPUs. Set `control_allocation` when the control services should use a particular job on their node; their Slurm steps are CPU-only.

Placement runs after Slurm has granted the allocation. Switchyard reads the allowed CPU set and GPU PCI locality, partitions physical CPU cores among workers in the same allocation, and binds process memory to the nearby NUMA node when available. Control services receive CPU cores when they share an allocation with workers. This requires no `cpus_per_task` or `numa_node` setting in `config.yaml`: request adequate CPU capacity through Slurm. `cpus_per_task` is an optional process-affinity cap, not a resource request; `numa_node` is a host-specific override. If Slurm grants too few local CPUs or GPU locality cannot be read, Switchyard logs the fallback and stays within the granted CPU set.

Clients connect directly to the Dynamo frontend on port 19000, using its embedded KV-aware router. etcd provides discovery, NATS carries events, Dynamo routes prefill/decode requests, and NIXL transfers model state between workers on different nodes. Explicit producer/consumer roles and failure-on-load-error prevent a missing transfer from silently becoming an intended local fallback. MTP, prefix caching, and tool parsing are configured together. The validated hybrid-model recipe uses equal TP widths; arbitrary mixed TP is not supported by this launcher.

Dynamo 1.5.0 selects the response-stream interface automatically; explicit RPC/NIXL addresses are advertised per node. Workers let UCX discover available transports and HCAs by default; `transport.environment` can override that policy.

## GPU and HCA locality

`uv run switchyard topology <recipe>` probes each Slurm allocation separately and reports visible GPUs, CPU/NUMA locality, HCA proximity, and the UCX device choice. The sample config enables `fabric.hca_selection: local`: workers use their GPU's best-ranked HCA when the path is within a NUMA node, while cross-socket or unknown paths remain under UCX selection. An explicit `UCX_NET_DEVICES` in `transport.environment` takes precedence. This selects a transport path; it is not evidence of a serving-speed improvement and should be measured on the target allocation.

## Fabric budget before placement tuning

Following [Doubleword's disaggregation article](https://blog.doubleword.ai/when-to-disaggregate), choose the smallest prefill pool that sustains the workload and devote remaining capacity to decode only after checking transfer cost. `switchyard inventory` records topology, `switchyard fabric` estimates required transfer bandwidth, and `switchyard monitor` samples live NIXL bytes and RDMA counters. For example:

```bash
uv run switchyard fabric --model dense --prompt-tokens 8192 --requests-per-second 1
```

The budget counts full-attention KV plus GDN recurrent and convolution state, including modeled MTP lookahead. Prefix hits do not automatically remove transfer bytes: decode-side reuse must be explicitly verified. This is a logical payload estimate, not a wire measurement. Alignment, state copies, connector layout, contention, and effective bandwidth must be checked before selecting a P:D ratio.

[`doublewordai/inference-lab`](https://github.com/doublewordai/inference-lab) informed the deployment/measurement workflow; this repository keeps a smaller Slurm-specific launcher.
