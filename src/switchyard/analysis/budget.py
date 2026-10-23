"""Model the prefill->decode transfer payload before choosing a P:D ratio.

This is a logical payload estimate, not a wire measurement: block alignment,
connector layout, contention and achieved bandwidth all still have to be
measured before a ratio is fixed.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..paths import expand


@dataclass(frozen=True)
class Budget:
    attention_layers: int
    gdn_layers: int
    kv_bytes_per_token: float
    recurrent_bytes_per_request: float
    conv_bytes_per_request: float
    state_copies: int
    transferred_kv_tokens: float
    modeled_bytes_per_request: float
    requests_per_second: float
    required_gbps: float
    required_link_gbps_at_target_utilization: float
    scope: str

    def as_dict(self):
        return asdict(self)


SCOPE = (
    "Aggregate TP=1 logical payload; excludes block padding unless supplied, "
    "replicated transfers, protocol overhead and contention. Confirm connector "
    "state copies, dtype and actual bytes before choosing P/D ratio. DFlash "
    "draft and verification state is not modeled."
)


def estimate(
    settings,
    text_config,
    tokens,
    rps,
    cache_hit=0.0,
    decode_reuse=False,
    state_copies=1,
    aligned_tokens=None,
):
    """Per-request bytes and the link rate they imply at ``rps``."""
    fabric = settings.fabric
    layers = text_config["layer_types"]
    full = layers.count("full_attention")
    linear = layers.count("linear_attention")
    kv_per_token = (
        full
        * 2
        * text_config["num_key_value_heads"]
        * text_config["head_dim"]
        * fabric.kv_bytes_per_element
    )
    recurrent = (
        linear
        * text_config["linear_num_value_heads"]
        * text_config["linear_key_head_dim"]
        * text_config["linear_value_head_dim"]
        * fabric.recurrent_bytes_per_element
    )
    channels = (
        2 * text_config["linear_num_key_heads"] * text_config["linear_key_head_dim"]
        + text_config["linear_num_value_heads"] * text_config["linear_value_head_dim"]
    )
    # Physical conv state has MTP lookahead; the connector layout must be
    # verified against what actually crosses the wire.
    conv = (
        linear
        * channels
        * (text_config["linear_conv_kernel_dim"] - 1 + settings.engine.mtp_tokens)
        * fabric.conv_bytes_per_element
    )
    transferred = (
        aligned_tokens
        if aligned_tokens is not None
        else tokens * (1 - cache_hit if decode_reuse else 1)
    )
    payload = kv_per_token * transferred + (recurrent + conv) * state_copies
    required = payload * rps * 8 / 1e9
    return Budget(
        attention_layers=full,
        gdn_layers=linear,
        kv_bytes_per_token=kv_per_token,
        recurrent_bytes_per_request=recurrent,
        conv_bytes_per_request=conv,
        state_copies=state_copies,
        transferred_kv_tokens=transferred,
        modeled_bytes_per_request=payload,
        requests_per_second=rps,
        required_gbps=required,
        required_link_gbps_at_target_utilization=required / fabric.target_utilization,
        scope=SCOPE,
    )


def model_text_config(path):
    """The checkpoint's own config.json, or its ``text_config`` block."""
    raw = json.loads(Path(path).read_text())
    return raw.get("text_config", raw)


def measure(budget, effective_gbps):
    """Compare the model against a measured application-payload bandwidth."""
    return {
        **budget.as_dict(),
        "link_utilization": budget.required_gbps / effective_gbps,
        "isolated_transfer_ms": budget.modeled_bytes_per_request
        * 8
        / (effective_gbps * 1e6),
        "within_target": budget.required_link_gbps_at_target_utilization
        <= effective_gbps,
    }


def checkpoint_path(launch):
    return Path(expand(launch.model.path)) / "config.json"
