# Results

## Latest result: provisional cross-node H100 serving smoke

Measured 2026-10-21 (America/New_York). Qwen3.6-27B BF16, Dynamo 1.5.0 / vLLM 0.28.0 / NIXL 1.3.2; MTP 3, prefix caching, FlashInfer GDN prefill, 32K context limit, 32 maximum sequences, 8192 batched tokens, GPU memory utilization 0.88.

One H100 NVL (~94 GiB) on g003 performed prefill and one on g004 performed decode. g001 hosted the gateway/frontend/etcd/NATS; its allocated GPU was not used for inference. All 36 measured requests completed without failures.

| Workload | Nominal input → output | Concurrency / requests | Output tok/s | Mean TTFT | Mean TPOT |
| --- | --- | --- | ---: | ---: | ---: |
| Prefill-heavy | 8192 → 256 | 4 / 12 | 190.81 | 2702.67 ms | 8.80 ms |
| Decode-heavy | 512 → 2048 | 8 / 24 | **755.01** | **515.74 ms** | **8.56 ms** |

**Provisional:** one run per workload, with no matched baseline/public/custom comparison or task-quality evaluation. Adapter loading did not establish custom decode execution in this run. The later [batch 2–16 kernel smoke](kernels/README.md) is separate evidence.

### Method and evidence

These H100 runs used `scripts/benchmark.sh` and `vllm bench serve` with the OpenAI completions backend, random inputs, temperature 0, `ignore-eos`, one warmup per workload, and detailed TTFT/TPOT/ITL/E2E output. The current `switchyard bench` command carries the same cases and vLLM arguments. Requests ran through the gateway and cross-node Dynamo/NIXL path. Worker logs showed prefill activity on the producer and generation with externally cached prefixes on the consumer. Adapter import/configuration markers alone do not prove decode dispatch; graph capture can also produce markers.

The server's `/tokenize` endpoint was unavailable, so requested input lengths were nominal. Actual aggregate inputs were **94,009** and **11,610** tokens respectively; outputs were exactly **3,072** and **49,152**. Durations were 16.1000 s and 65.1014 s. Median TTFT was 2160.50 / 390.29 ms; median TPOT was 8.07 / 8.33 ms. Throughput is aggregate output tokens divided by benchmark duration, not per-request generation speed. MTP emits chunks, so ITL must not be substituted for TPOT.

A [numeric summary](results/h100-cross-node-20261021.json) is included in the repository.

---

## Historical measurements (previous deployment)

### H200 + H100 run: 2026-09-29

`runs/h100-128-completions-20260929/` completed with no failed benchmark requests. It used Qwen3.6-27B, bf16, MTP disabled, CUDA 13, Dynamo 1.5.0, vLLM 0.28.0, and deterministic completion requests (`temperature=0`, `ignore_eos=true`). Each target received the same token lengths, concurrency, request count, and random seed.

| Target | Hardware | Workload | Success | Output tok/s | Mean TTFT | Mean TPOT |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `h200-pd` | 2 × H200 NVL | 8,192 in / 256 out | 12 / 12 | 94.36 | 5.79 s | 17.22 ms |
| `h100-1p2d` | 3 × H100 SXM | 8,192 in / 256 out | 12 / 12 | 80.99 | 7.30 s | 20.70 ms |
| `heterogeneous-pool` | 5 GPUs total | 8,192 in / 256 out | 12 / 12 | 161.39 | 851 ms | 18.41 ms |
| `h200-pd` | 2 × H200 NVL | 512 in / 2,048 out | 24 / 24 | 440.08 | 886 ms | 17.74 ms |
| `h100-1p2d` | 3 × H100 SXM | 512 in / 2,048 out | 24 / 24 | 369.62 | 1.23 s | 20.85 ms |
| `heterogeneous-pool` | 5 GPUs total | 512 in / 2,048 out | 24 / 24 | 385.71 | 587 ms | 18.72 ms |

The gateway sent six prefill-heavy and twelve decode-heavy completions to each backend, verified from unique frontend request IDs. The pooled prefill case reused deterministic prompts after the direct runs, so its prefix caches could be warm. The five-GPU pool and direct targets also used different GPU counts; this is routing evidence, not a matched hardware ranking.

### Producer/consumer role migration: 2026-09-29

`runs/role-smoke-20260929T141608Z/` verified the current Dynamo role split on the H200 P/D deployment. The prefill worker used `kv_producer` and the decode worker used `kv_consumer`. One completion transferred 195.8 MB of KV data, and Dynamo recorded separate prefill and decode worker IDs for completed benchmark requests. The run logs contain no `kv_both` role warning.

| Workload | Success | Output tok/s | Mean TTFT | Mean TPOT | Prior H200 output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 in / 256 out | 12 / 12 | 94.29 | 5.75 s | 17.24 ms | 94.36 |
| 512 in / 2,048 out | 24 / 24 | 440.54 | 849 ms | 17.74 ms | 440.08 |

The result is one controlled rerun, so it shows no measured regression from the role migration. It does not establish a throughput gain from it.
