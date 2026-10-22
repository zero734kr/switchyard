# Deployment notes

`config.yaml` is the single source of runtime and placement settings. `scripts/switchyard.py` snapshots it for each run, verifies allocation ownership and node placement, and launches services through `srun --jobid ... --overlap --exact`. It uses existing allocations and never cancels them. The `heterogeneous` recipe is deliberately incomplete until suitable allocations are available.

Clients connect directly to the Dynamo frontend on port 19000, using its embedded KV-aware router. etcd provides discovery, NATS carries events, Dynamo routes prefill/decode requests, and NIXL transfers model state between workers on different nodes. Explicit producer/consumer roles and failure-on-load-error prevent a missing transfer from silently becoming an intended local fallback. MTP, prefix caching, and tool parsing are configured together. The validated hybrid-model recipe uses equal TP widths; arbitrary mixed TP is not supported by this launcher.

Dynamo 1.5.0 selects the response-stream interface automatically; explicit RPC/NIXL addresses are advertised per node. Transport overrides remain in YAML without forcing TCP-only UCX.

## Fabric budget before placement tuning

Following [Doubleword's disaggregation article](https://blog.doubleword.ai/when-to-disaggregate), choose the smallest prefill pool that sustains the workload and devote remaining capacity to decode only after checking transfer cost. `scripts/inventory.sh` records topology; `scripts/fabric.py` estimates required bandwidth from the checkpoint configuration, prompt length, and request rate. For example:

```bash
uv run python scripts/fabric.py --model dense --prompt-tokens 8192 --requests-per-second 1
```

The budget counts full-attention KV plus GDN recurrent and convolution state, including modeled MTP lookahead. Prefix hits do not automatically remove transfer bytes: decode-side reuse must be explicitly verified. This is a logical payload estimate, not a wire measurement. Alignment, state copies, connector layout, contention, and effective bandwidth must be checked before selecting a P:D ratio.

[`doublewordai/inference-lab`](https://github.com/doublewordai/inference-lab) informed the deployment/measurement workflow; this repository keeps a smaller Slurm-specific launcher.