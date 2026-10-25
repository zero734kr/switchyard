# Results

## Five-H200 serving: throughput and latency

**6.6–6.7K output tokens/s at concurrency 64**, with mean TTFT around **0.9 s** and observed p99 around **3 s**. Reducing concurrency to 32 delivered **4.94K output tokens/s** and TTFT improvements with **4.64 ms mean TPOT**.

Measured on October 24, 2026 (America/New_York): **Qwen3.8-27B BF16**, five H200 NVL GPUs across two GPU nodes, **one prefill and four decode workers**, TP=1 per worker, and a separate CPU-only node for Dynamo frontend, etcd and NATS.

| Workload | Concurrency · completed | Output tok/s | Mean TTFT | p99 TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode-heavy | 64 · 192/192 | **6,740.73** | 899 ms | 3.06 s | 6.21 ms |
| Decode-heavy, second seed | 64 · 192/192 | **6,567.19** | 895 ms | 2.99 s | 5.91 ms |
| Decode-heavy, lower latency | 32 · 96/96 | **4,937.18** | **596 ms** | **1.63 s** | **4.64 ms** |
| Prefill-heavy, longer input | 4 · 12/12 | 319.56 | 1.92 s | 2.96 s | 4.40 ms |

All **492 measured requests completed without failures**. C64 raises throughput with higher per-token latency; C32 is the lower-latency option. The prefill-heavy row checks longer-input behavior; its output tok/s slowed down due to the computational-heavy nature of prefill operations.

## Recipe and measurement

- **Stack:** vLLM 0.28.0, Dynamo 1.5.0, NIXL 1.3.2, CUDA toolkit 13.0.2. DFlash2 draft `incoai/Qwen3.8-27B-DFlash2`, seven speculative tokens, prefix caching and FlashInfer GDN prefill. Context limit 32K, 32 maximum sequences per worker, 8,192 batched tokens and GPU memory utilization 0.97.
- **Workload:** `vllm bench serve` through the Dynamo frontend, OpenAI `/v1/completions`, random inputs, temperature 0, ignore EOS, infinite request rate and one warmup per case. Decode requests use nominal 512 input / 2,048 output tokens; prefill-heavy requests use 8,192 / 256.
- **Metrics:** aggregate output throughput is generated tokens divided by the complete benchmark duration. Mean TPOT averages each request's generation time after its first token, divided by its remaining output tokens. Speculation emits token groups, so inter-event latency is not interchangeable with TPOT. The two C64 runs use different prompts, rather than repeating an identical workload.

**Scope:** synthetic serving measurements, with empirical p99 estimates from these request counts. Custom adapters were enabled, but timed custom-kernel replay and an incremental end-to-end Triton gain were not established for these runs.

Triton implementation details and the separately measured **8.1–9.4% dense-operator latency reduction** are in [kernels/README.md](kernels/README.md).

## Measurement records

[Configuration](results/h200-serving-20261024/configuration.json) · [C64](results/h200-serving-20261024/decode-c64.json) · [C64 second seed](results/h200-serving-20261024/decode-c64-repeat.json) · [C32](results/h200-serving-20261024/decode-c32.json) · [Prefill C4](results/h200-serving-20261024/prefill-c4.json).

Records contain selected vLLM metrics, per-request lengths, TTFT and TPOT, plus hashes of the original benchmark JSONs.