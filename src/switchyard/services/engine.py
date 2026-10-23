"""`dynamo.vllm` arguments for one worker.

Everything the launcher flags on the engine lives here, so a change to a
serving feature is a change to one function instead of one string list.
"""

import json

from ..paths import expand


def kv_transfer_config(role):
    # Explicit producer/consumer roles, and a failed load must not silently
    # become an intended local fallback.
    return json.dumps(
        {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_load_failure_policy": "fail",
        }
    )


def speculative_config(mtp_tokens):
    return json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens})


def kv_events_config(port):
    return json.dumps(
        {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{port}",
            "enable_kv_cache_events": True,
        }
    )


def args(launch, worker):
    """The complete argument vector passed to ``python -m dynamo.vllm``."""
    settings = launch.settings
    engine = settings.engine
    model = launch.model
    values = [
        "-m",
        "dynamo.vllm",
        "--model",
        expand(model.path),
        "--served-model-name",
        model.served_name,
        "--tensor-parallel-size",
        str(worker.tp),
        "--dtype",
        engine.dtype,
        "--gpu-memory-utilization",
        str(engine.gpu_memory_utilization),
        "--max-model-len",
        str(engine.max_model_len),
        "--max-num-seqs",
        str(engine.max_num_seqs),
        "--max-num-batched-tokens",
        str(engine.max_num_batched_tokens),
        "--discovery-backend",
        "etcd",
        "--disaggregation-mode",
        worker.role,
        "--language-model-only",
        "--enable-chunked-prefill",
        "--reasoning-parser",
        "qwen3",
        "--gdn-prefill-backend",
        engine.gdn_prefill_backend,
        "--kv-transfer-config",
        kv_transfer_config(worker.role),
    ]
    if engine.prefix_caching:
        values.append("--enable-prefix-caching")
    if engine.enforce_eager:
        values.append("--enforce-eager")
    if engine.mtp_tokens:
        values += ["--speculative-config", speculative_config(engine.mtp_tokens)]
    if worker.role == "prefill" and worker.event_port is not None:
        values += ["--kv-events-config", kv_events_config(worker.event_port)]
    return values
