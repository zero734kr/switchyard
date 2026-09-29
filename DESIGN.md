# Design

## Goal

Demonstrate one reproducible heterogeneous inference experiment using hardware that was available at this moment of writing: two H200 NVL GPUs on `g213` and three H100 SXM GPUs on `g013`.

The project must answer three concrete questions:

1. Can Qwen3.6-27B run correctly with local prefill/decode separation on both GPU classes?
2. Can one streaming endpoint use both deployments during the same workload?
3. How do direct H200, direct H100, and pooled service compare on prefill-heavy and decode-heavy traffic?

## Boundary

Dynamo owns model serving, OpenAI request processing, worker discovery, P/D routing, and NIXL transfer inside each node. `gateway.py` only selects a complete deployment and streams bytes in both directions. It does not inspect prompts, move KV state, retry generation, or schedule individual GPUs.

Dynamo file discovery is machine local. Cross-node P/D would require etcd/NATS plus a validated network transfer path, so this prototype does not imply that an H200 prefill worker transfers state to an H100 decode worker.

## Placement

| Profile | Frontend | Prefill | Decode | Purpose |
| --- | --- | --- | --- | --- |
| `h200-pd` | `g213:18000` | GPU 0 | GPU 1 | H200 direct control |
| `h100-pd` | `g013:18000` | GPU 0 | GPU 1 | symmetric H100 debugging profile |
| `h100-1p2d` | `g013:18000` | GPU 0 | GPUs 1, 2 | full H100 allocation and pooled comparison |

The gateway runs on `g213:19000`. It chooses the backend minimizing `(active requests + 1) / weight`. Equal weights are the honest initial condition; `config.json` keeps weight available as a hardware calibration knob after measurements justify a change.

The H100 profiles use a 98,304-token context limit and a 128-sequence limit. At 131,072 tokens, vLLM measured 6.95 GiB of available H100 KV memory where Qwen3.6 required 8.33 GiB. At this context, it also reported 145 Mamba cache blocks, so 512 decode sequences could not start CUDA graph capture. The H200 profile retains 131,072 tokens and 512 sequences. The benchmark workloads stay far below both limits.

## Reproducibility

Each run stores:

- the resolved configuration;
- node, Slurm job, GPU topology, and package versions;
- separate frontend, prefill, decode, gateway, and benchmark logs;
- detailed `vllm bench serve` JSON for every target and workload;
- final per-backend gateway request and failure counts.

All three target comparisons share model, dtype, MTP count, memory fraction, benchmark seed, token lengths, concurrency, and request count. The H100 context and sequence limits differ because vLLM admission measured lower usable cache capacity on that node; every benchmark request stays within both profiles.

## Failure behavior

- Configuration is validated before GPUs start.
- A profile refuses to run on the wrong node unless explicitly overridden.
- Startup waits for the model to appear in `/v1/models`, not only for an HTTP health response.
- NIXL load failures are fatal.
- The experiment times out instead of waiting forever.
- Signal cleanup targets only process IDs created by the current driver.
- The gateway reports backend connection errors as HTTP 502 and counts them.

## Deliberate limits (as this is only a prototype)

- No Kubernetes, container build, database, custom metrics stack, or web dashboard.
- No automatic Slurm submission or allocation management.
- No learned or prompt-aware routing policy.
- No performance claim until the saved benchmark JSON exists.
