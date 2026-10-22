# Triton kernels for Qwen3.6

I started from **DeepFusionKernel**'s published fusion ideas and tuned Triton kernels for Qwen3.6 on Jarvis's Hopper GPUs. The work specializes dense/MoE computation and GDN integration for the model dimensions and vLLM runtime.

## Performance contribution by feature

| Feature | Change | Measured contribution |
| --- | --- | --- |
| **Dense MLP, batch 1** | Fuse gate/up + SwiGLU; specialize for 5120 × 17408 with BF16 intermediate rounding. | H200 NVL: **85.3–85.4 µs** vs public DeepFusionKernel A2 **88.2 µs** (~3% lower latency), or public matmul + activation **98.0–98.6 µs** (~13% lower). |
| **Dense MLP, batches 2–16** | Reuse weight tiles across token rows; fall back above 16. | H100 NVL synthetic operator: **~9–12% faster** at tested batches 2, 4, 8, 16, reported by the batch-tuning session. |
| **MoE routed experts** | Fuse gate/up + SwiGLU, then down projection + top-k reduction; bypass single-token sorting/padding. | H200 NVL synthetic operator: **25.39 → 18.50 µs (1.37×)** vs installed vLLM `fused_experts`. |
| **GDN projection pair** | Compute QKVZ and BA in one Triton launch. | **Two projection launches → one**; individual serving gain not yet measured. |
| **Public GDN recurrence tuning** | Tune the existing public kernel configuration. | Individual serving gain not yet measured; credited as public-kernel tuning. |

**Scope:** these are operator measurements, not additive model speedups. The batch-1 comparison screened 144 public A2 vector configurations on merged row-major weights; it does not cover every upstream kernel. The MoE test rotated 16 synthetic activation/routing sets and excluded routing time on both sides.

**Serving check:** an MTP=3 smoke selected the batched dense path at 4, 8, and 16 and completed concurrent requests. Model-wide performance and quality validation remain pending.

The [755 tok/s P/D + MTP result](../RESULTS.md) predates this batch extension. It does not isolate the kernels' contribution. MTP and prefix caching are serving features, not custom Triton optimizations.

## Integration

Set `engine.kernels: custom` in `config.yaml` to load process-local adapters through `sitecustomize.py`; `baseline` disables them. Integration is pinned to vLLM 0.28.0. Shape/layout guards retain public fallbacks; request concurrency differs from kernel token-row count under MTP and CUDA graphs.

- [`jarvis_dense.py`](jarvis_dense.py): dense batch-1 and batch 2–16 MLP paths.
- [`original_moe.py`](original_moe.py): single-token BF16 routed-expert path for Qwen3.6-35B-A3B.
- [`pair_kernel.py`](pair_kernel.py) and [`recipe_plugin.py`](recipe_plugin.py): GDN projection pairing and recurrence configuration.

## References

- Zixi Zhang, Zhiwen Mo, Yiren Zhao, Robert D. Mullins. [Deep Kernel Fusion for Transformers, ACL 2026](https://aclanthology.org/2026.acl-short.15/); [DeepFusionKernel source](https://github.com/ZixiBenZhang/deepfusionkernel/tree/e7a05189171c023499c4fb60dfd95c51f0edb295). Initial inspiration and public MLP fusion comparison.
- [vLLM v0.28.0](https://github.com/vllm-project/vllm/tree/v0.28.0): integration interfaces, public operator baselines, fallbacks, and bundled recurrence implementation.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention): GDN/linear-attention implementation background.
- [SFLLM Qwen MoE implementation](https://github.com/wejoncy/sfllm/blob/main/docs/qwen3_5_moe.md): related shared/routed expert fusion work.
