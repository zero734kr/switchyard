# Switchyard

Cross-node LLM serving on Stevens Jarvis, built around NVIDIA Dynamo, Slurm and vLLM, with Qwen3.6-specific Triton kernel work.

## Provisional result

A short Qwen3.6-27B BF16 run reached **755 output tokens/s**, **8.56 ms mean TPOT**, and **516 ms mean TTFT** on a decode-heavy workload. It used one H100 NVL for prefill, one for decode, MTP=3, and a separate control allocation. The prefill-heavy run reached **190.81 output tokens/s**.

These are provisional serving results: one run per workload, no matched baseline or task-quality comparison, and no verified custom-kernel contribution. See [RESULTS.md](RESULTS.md) for the method and [kernels/README.md](kernels/README.md) for operator results and dispatch evidence.

## Deployment

- Dynamo routes requests to vLLM prefill and decode workers; NIXL transfers KV and recurrent state across nodes.
- Recipes use existing Slurm allocations and support multiple jobs on one node. CPU/NUMA placement is selected after allocation.
- Qwen3.6-27B is the default; use `--model moe` for Qwen3.6-35B-A3B.
- MTP, prefix caching, tool parsing, and guarded Triton kernels are configurable in `config.yaml`.

## Run on Jarvis

Update model paths, modules, nodes, and active job IDs in `config.yaml`. The CLI environment is separate from the inference venv at `~/.venvs/vllm`.

```bash
uv sync
uv run switchyard config --list
uv run switchyard plan initial-h100
uv run switchyard preflight initial-h100
uv run switchyard bootstrap
uv run switchyard run initial-h100
```

Replace expired job IDs before running. `switchyard topology <recipe>` reports GPU, NUMA, and HCA locality. See [DESIGN.md](DESIGN.md) for allocation and fabric details.

## Project files

- [`config.yaml`](config.yaml): model, runtime, and Jarvis recipes.
- [`src/switchyard/`](src/switchyard/): CLI and serving launcher.
- [`kernels/`](kernels/README.md): Triton operators and vLLM adapters.
- [`RESULTS.md`](RESULTS.md): serving measurements and caveats.

## Special Thanks

- [Doubleword AI](https://github.com/doublewordai) for:
  - [“When to Disaggregate”](https://blog.doubleword.ai/when-to-disaggregate)
  - [Inference Lab](https://github.com/doublewordai/inference-lab)
- [DeepFusionKernel authors](https://aclanthology.org/2026.acl-short.15/)
