"""CPU-only hybrid Qwen transfer budget; modeled payload, not measured wire traffic."""

import argparse
import json
from pathlib import Path
from config import DEFAULT_CONFIG, expand, read_config


def budget(
    c,
    model_config,
    tokens,
    rps,
    cache_hit,
    decode_reuse,
    state_copies,
    aligned_tokens=None,
):
    t = model_config.get("text_config", model_config)
    layers = t["layer_types"]
    f = c["fabric"]
    full, linear = layers.count("full_attention"), layers.count("linear_attention")
    kv_per_token = (
        full * 2 * t["num_key_value_heads"] * t["head_dim"] * f["kv_bytes_per_element"]
    )
    recurrent = (
        linear
        * t["linear_num_value_heads"]
        * t["linear_key_head_dim"]
        * t["linear_value_head_dim"]
        * f["recurrent_bytes_per_element"]
    )
    channels = (
        2 * t["linear_num_key_heads"] * t["linear_key_head_dim"]
        + t["linear_num_value_heads"] * t["linear_value_head_dim"]
    )
    # Physical conv state has MTP lookahead; actual connector layout must be verified.
    conv = (
        linear
        * channels
        * (t["linear_conv_kernel_dim"] - 1 + c["engine"]["mtp_tokens"])
        * f["conv_bytes_per_element"]
    )
    transferred_tokens = (
        aligned_tokens
        if aligned_tokens is not None
        else tokens * (1 - cache_hit if decode_reuse else 1)
    )
    payload = kv_per_token * transferred_tokens + (recurrent + conv) * state_copies
    return {
        "attention_layers": full,
        "gdn_layers": linear,
        "kv_bytes_per_token": kv_per_token,
        "recurrent_bytes_per_request": recurrent,
        "conv_bytes_per_request": conv,
        "state_copies": state_copies,
        "transferred_kv_tokens": transferred_tokens,
        "modeled_bytes_per_request": payload,
        "requests_per_second": rps,
        "required_gbps": payload * rps * 8 / 1e9,
        "required_link_gbps_at_target_utilization": payload
        * rps
        * 8
        / 1e9
        / f["target_utilization"],
        "scope": "Aggregate TP=1 logical payload; excludes block padding unless supplied, replicated transfers, protocol overhead and contention. Confirm connector state copies, dtype and actual bytes before choosing P/D ratio.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=("dense", "moe"), default="dense")
    p.add_argument("--model-config", type=Path)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--requests-per-second", type=float)
    g.add_argument(
        "--prefill-tokens-per-second",
        type=float,
        help="aggregate uncached token throughput",
    )
    p.add_argument("--cache-hit-rate", type=float, default=0)
    p.add_argument(
        "--decode-prefix-reuse",
        action="store_true",
        help="only use after connector reuse is verified",
    )
    p.add_argument("--state-copies", type=int, default=1)
    p.add_argument(
        "--transferred-kv-tokens",
        type=int,
        help="override with measured/aligned transfer count",
    )
    p.add_argument(
        "--effective-gbps",
        type=float,
        help="measured application payload bandwidth, not NIC nameplate",
    )
    a = p.parse_args()
    if a.prompt_tokens <= 0 or not 0 <= a.cache_hit_rate < 1 or a.state_copies < 1:
        p.error("invalid prompt length, cache-hit rate or state copies")
    if a.transferred_kv_tokens is not None and a.transferred_kv_tokens < 0:
        p.error("transferred KV tokens must be nonnegative")
    rps = (
        a.requests_per_second
        if a.requests_per_second is not None
        else a.prefill_tokens_per_second / (a.prompt_tokens * (1 - a.cache_hit_rate))
    )
    if rps <= 0 or (a.effective_gbps is not None and a.effective_gbps <= 0):
        p.error("rates must be positive")
    c = read_config(a.config)
    path = a.model_config or Path(expand(c["models"][a.model]["path"])) / "config.json"
    result = budget(
        c,
        json.loads(path.read_text()),
        a.prompt_tokens,
        rps,
        a.cache_hit_rate,
        a.decode_prefix_reuse,
        a.state_copies,
        a.transferred_kv_tokens,
    )
    if a.effective_gbps:
        result.update(
            link_utilization=result["required_gbps"] / a.effective_gbps,
            isolated_transfer_ms=result["modeled_bytes_per_request"]
            * 8
            / (a.effective_gbps * 1e6),
            within_target=result["required_link_gbps_at_target_utilization"]
            <= a.effective_gbps,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
