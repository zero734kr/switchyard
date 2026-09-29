# Results

## H200 + H100 run: 2026-09-29

`runs/h100-128-completions-20260929/` completed with no failed benchmark requests. It used Qwen3.6-27B, bf16, MTP disabled, CUDA 13, Dynamo 1.5.0, vLLM 0.28.0, and deterministic completion requests (`temperature=0`, `ignore_eos=true`). Each target received the same token lengths, concurrency, request count, and random seed.

| Target | Hardware | Workload | Success | Output tok/s | Mean TTFT | Mean TPOT |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `h200-pd` | 2 × H200 NVL | 8,192 in / 256 out | 12 / 12 | 94.36 | 5.79 s | 17.22 ms |
| `h100-1p2d` | 3 × H100 SXM | 8,192 in / 256 out | 12 / 12 | 80.99 | 7.30 s | 20.70 ms |
| `heterogeneous-pool` | 5 GPUs total | 8,192 in / 256 out | 12 / 12 | 161.39 | 851 ms | 18.41 ms |
| `h200-pd` | 2 × H200 NVL | 512 in / 2,048 out | 24 / 24 | 440.08 | 886 ms | 17.74 ms |
| `h100-1p2d` | 3 × H100 SXM | 512 in / 2,048 out | 24 / 24 | 369.62 | 1.23 s | 20.85 ms |
| `heterogeneous-pool` | 5 GPUs total | 512 in / 2,048 out | 24 / 24 | 385.71 | 587 ms | 18.72 ms |

### Routing evidence

The pooled workload sent six prefill-heavy completion requests and twelve decode-heavy completion requests to **each** backend. This is verified from unique `x_request_id` values in both Dynamo frontend logs. The gateway status at shutdown recorded zero active requests and zero failures on both backends (`h200-pd`: 30 proxied HTTP calls; `h100-1p2d`: 18). Those counters include benchmark helper calls such as `/tokenize` and `/metrics`, so the frontend request IDs are the generation-routing evidence.

### What this run does and does not show

- It proves one streaming endpoint sent live completion traffic to an H200 P/D deployment and an H100 1-prefill/2-decode deployment in the same run.
- Direct H200 results were faster than the H100 deployment for both workloads under these exact profiles.
- The equal-weight gateway improved warm prefill-heavy throughput. It is **not** a cold-cache speedup result: the pooled prefill case ran after the direct cases using the same deterministic prompt seed, so Dynamo prefix caches could already contain the prompt prefixes.
- The pooled decode-heavy result was below the direct H200 decode result. With only eight concurrent requests, equal request splitting reduces H200 batch size and sends half the traffic to the slower H100 branch. A capacity-aware weight or a higher offered load is the next experiment, not a conclusion from this one sample.

### H100 fit findings

The H100 profile first failed with the H200 defaults. These are measured vLLM admission limits, so the final H100 profile uses 98,304 context tokens and 128 sequences:

| Check | H100 result |
| --- | --- |
| MTP2 startup | CUDA graph profiling OOM: 1.56 GiB requested with 1.52 GiB free |
| MTP disabled, 131,072 context | 6.95 GiB KV available; Qwen3.6 required 8.33 GiB |
| 98,304 context, 512 sequences | 145 Mamba cache blocks; CUDA graph capture rejected 512 sequences |
| Final H100 profile | MTP disabled, 98,304 context, 128 sequences |

All benchmark requests remain within the final H100 limits.

## Verified exploratory observations

These measurements came from the earlier two-H200 tuning session. They establish the path to the current configuration but are not a controlled comparison with the heterogeneous pool.

### Correctness and transfer

| Observation | Value |
| --- | ---: |
| External prefix reuse | 95.5–100% |
| Direct CUDA peer bandwidth, GPU 0 → 1 | ~122.96 GiB/s |
| Direct CUDA peer bandwidth, GPU 1 → 0 | ~122.98 GiB/s |
| 256 MiB peer copy | ~2.033 ms |

NIXL telemetry included transfers around 211 MB, 1,326 MB, and 2,124 MB. Its reported transfer duration includes completion polling behavior and must not be presented as raw NVLink bandwidth.

### Decode backend regression

| Configuration | Warm single-request decode |
| --- | ---: |
| Forced `--attention-backend FLASHINFER`, MTP enabled | ~25–28 tokens/s |
| vLLM automatic decode backend, MTP enabled | ~64–71 tokens/s |

The forced backend caused speculative decode to fall back to piecewise CUDA graphs. Removing that override produced roughly a 2.5× recovery while remote reuse remained 96–100%. This is a configuration recovery, not an end-to-end P/D speedup claim.

## Interpretation rules

- Compare completed requests and failures before comparing throughput.
- Treat the pooled result as a gateway-routing result with a warm-cache caveat. It uses five GPUs, while each direct target uses only its own allocation.
- Use TTFT for prefill behavior and TPOT/ITL for decode behavior.
- A mixed pool can improve aggregate throughput while worsening tail latency; report both.
- Gateway request counts must show nonzero traffic on both backend names before calling the run heterogeneous.
- One run is an exploratory systems result, not a statistically precise hardware ranking.
