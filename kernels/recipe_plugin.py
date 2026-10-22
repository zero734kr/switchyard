"""Opt-in Qwen single-token projection pairing and FP32 GDN-state tiling."""

import os
import torch
import triton
from pair_kernel import project_pair
from shared_combine import shared_gate_mul as _shared_gate_mul
from state_kernel import step

_pair_seen = set()
_merged_weights = {}


@torch.library.custom_op("jarvis_recipe::public_merged", mutates_args=())
def public_merged(x: torch.Tensor, wq: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    # Pack once after loading. Keep references so pointer reuse cannot alias a cache entry.
    key = (wq.data_ptr(), wb.data_ptr())
    if key not in _merged_weights:
        _merged_weights[key] = (wq, wb, torch.cat((wq, wb), dim=0))
        print(
            f"JARVIS_PUBLIC_MERGED_ENABLED h={wq.shape[1]} dq={wq.shape[0]} db={wb.shape[0]}",
            flush=True,
        )
    return torch.mm(x, _merged_weights[key][2].T)


@public_merged.register_fake
def fake_public_merged(x, wq, wb):
    return x.new_empty((x.shape[0], wq.shape[0] + wb.shape[0]))


@torch.library.custom_op("jarvis_recipe::project_pair", mutates_args=())
def pair(
    x: torch.Tensor, wq: torch.Tensor, wb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    oq = x.new_empty((x.shape[0], wq.shape[0]))
    ob = x.new_empty((x.shape[0], wb.shape[0]))
    h, dq, db = wq.shape[1], wq.shape[0], wb.shape[0]
    if x.shape[0] == 1:
        if h not in _pair_seen:
            print(f"JARVIS_PROJECTION_PAIR_ENABLED h={h} dq={dq} db={db}", flush=True)
            _pair_seen.add(h)
        rows, warps = (1, 4) if h == 5120 else (2, 8)
        project_pair[(triton.cdiv(dq, rows) + triton.cdiv(db, rows),)](
            x,
            wq,
            wb,
            oq,
            ob,
            h,
            dq,
            db,
            triton.next_power_of_2(h),
            rows,
            num_warps=warps,
        )
    else:
        torch.mm(x, wq.T, out=oq)
        torch.mm(x, wb.T, out=ob)
    return oq, ob


@pair.register_fake
def fake_pair(x, wq, wb):
    return x.new_empty((x.shape[0], wq.shape[0])), x.new_empty(
        (x.shape[0], wb.shape[0])
    )


@torch.library.custom_op("jarvis_recipe::shared_gate_mul", mutates_args=())
def shared_gate_mul(
    gate_logits: torch.Tensor, shared_output: torch.Tensor
) -> torch.Tensor:
    return _shared_gate_mul(gate_logits, shared_output)


@shared_gate_mul.register_fake
def fake_shared_gate_mul(gate_logits, shared_output):
    return shared_output.new_empty(shared_output.shape)


def install_shared_gate_mul():
    """Opt into a graph-safe Triton gate/multiply for Qwen3.6-35B shared MLPs."""
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    cls = Qwen2MoeMLP
    if getattr(cls, "_jarvis_shared_gate_mul", False):
        return
    original = cls.forward
    announced = set()

    def forward(self, x):
        supported = (
            self.expert_gate is not None
            and x.ndim == 2
            and x.shape[0] <= 64
            and x.shape[1] == 2048
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and tuple(self.gate_up_proj.weight.shape) == (1024, 2048)
            and tuple(self.down_proj.weight.shape) == (2048, 512)
            and tuple(self.expert_gate.weight.shape) == (1, 2048)
            and self.gate_up_proj.weight.dtype == torch.bfloat16
            and self.down_proj.weight.dtype == torch.bfloat16
            and self.expert_gate.weight.dtype == torch.bfloat16
        )
        if not supported:
            return original(self, x)

        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        gate, _ = self.expert_gate(x)
        if x.shape[0] not in announced:
            print(
                f"JARVIS_SHARED_GATE_MUL_ENABLED tokens={x.shape[0]} hidden=2048 intermediate=512",
                flush=True,
            )
            announced.add(x.shape[0])
        return torch.ops.jarvis_recipe.shared_gate_mul(gate, out)

    cls.forward = forward
    cls._jarvis_shared_gate_mul = True


def install():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as module
    from vllm.utils.torch_utils import _encode_layer_name

    cls = module.QwenGatedDeltaNetAttention
    if not getattr(cls, "_jarvis_config_inspected", False):
        original_init = cls.__init__
        announced_config = []

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not announced_config:
                import json

                print(
                    "JARVIS_PAIR_CONFIGURATION "
                    + json.dumps(
                        {
                            "qkvz": list(self.in_proj_qkvz.weight.shape),
                            "ba": list(self.in_proj_ba.weight.shape),
                            "dtype": str(self.in_proj_ba.weight.dtype),
                            "qkvz_contiguous": self.in_proj_qkvz.weight.is_contiguous(),
                            "ba_contiguous": self.in_proj_ba.weight.is_contiguous(),
                            "qkvz_quant": type(self.in_proj_qkvz.quant_method).__name__,
                            "ba_quant": type(self.in_proj_ba.quant_method).__name__,
                            "fused_decode": self.enable_fused_gdn_decode,
                            "packed_decode": self.enable_packed_recurrent_decode,
                            "tp": self.tp_size,
                            "interleaved": self.gqa_interleaved_layout,
                        }
                    ),
                    flush=True,
                )
                announced_config.append(True)

        cls.__init__ = init
        cls._jarvis_config_inspected = True
    if os.environ.get("JARVIS_PROJECTION_PAIR") in (
        "1",
        "merged",
        "decodepair",
    ) and not getattr(cls, "_jarvis_pair", False):
        original = cls.forward_cuda

        def forward(self, x):
            wq, wb = self.in_proj_qkvz.weight, self.in_proj_ba.weight
            expected = {5120: (16384, 96), 2048: (12288, 64)}.get(x.shape[-1])
            supported = (
                x.dtype == torch.bfloat16
                and x.is_contiguous()
                and expected is not None
                and self.tp_size == 1
                and not self.gqa_interleaved_layout
                and self.enable_fused_gdn_decode
                and tuple(wq.shape) == (expected[0], x.shape[-1])
                and tuple(wb.shape) == (expected[1], x.shape[-1])
                and wq.dtype == wb.dtype == torch.bfloat16
                and wq.is_contiguous()
                and wb.is_contiguous()
                and self.in_proj_qkvz.bias is None
                and self.in_proj_ba.bias is None
                and type(self.in_proj_qkvz.quant_method).__name__
                == "UnquantizedLinearMethod"
                and type(self.in_proj_ba.quant_method).__name__
                == "UnquantizedLinearMethod"
            )
            if not supported:
                return original(self, x)
            projection_mode = os.environ.get("JARVIS_PROJECTION_PAIR")
            # Match publicopt's merged GEMM for prefill-sized inputs, and use
            # the custom Triton pair only for one-token decode. This keeps the
            # experiment focused on decode and avoids the two-GEMM prefill
            # fallback used by the original "1" mode.
            use_public_merged = projection_mode == "merged" or (
                projection_mode == "decodepair" and x.shape[0] != 1
            )
            if use_public_merged:
                joined = public_merged(x, wq, wb)
                qkvz = joined[:, : wq.shape[0]].contiguous()
                ba = joined[:, wq.shape[0] :].contiguous()
            else:
                qkvz, ba = pair(x, wq, wb)
            core = torch.zeros(
                (x.shape[0], self.num_v_heads, self.head_v_dim),
                dtype=x.dtype,
                device=x.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                qkvz, ba, core, layer_name=_encode_layer_name(self.prefix)
            )
            return self.out_proj(core.flatten(-2))[0]

        cls.forward_cuda = forward
        cls._jarvis_pair = True

    if os.environ.get("JARVIS_GDN_TILED") in ("1", "public", "bv4") and not getattr(
        module, "_jarvis_state", False
    ):
        original_step = module.fused_recurrent_gated_delta_rule_packed_decode
        announced = set()

        def packed(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel=False,
        ):
            hv, v, k = initial_state.shape[-3:]
            supported = (
                mixed_qkv.shape[0] == 1
                and hv in (32, 48)
                and v == k == 128
                and mixed_qkv.shape[1] == (32 + hv) * 128
                and mixed_qkv.dtype == a.dtype == b.dtype == out.dtype == torch.bfloat16
                and initial_state.dtype == torch.float32
                and tuple(initial_state.stride()[-3:]) == (16384, 128, 1)
                and initial_state.stride(0) >= hv * 16384
                and mixed_qkv.stride(-1) == a.stride(-1) == b.stride(-1) == 1
                and A_log.is_contiguous()
                and dt_bias.is_contiguous()
                and ssm_state_indices.is_contiguous()
                and out.is_contiguous()
                and use_qk_l2norm_in_kernel
                and scale == 128.0**-0.5
            )
            if not supported:
                return original_step(
                    mixed_qkv,
                    a,
                    b,
                    A_log,
                    dt_bias,
                    scale,
                    initial_state,
                    out,
                    ssm_state_indices,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            if os.environ.get("JARVIS_GDN_TILED") == "public":
                from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
                    fused_recurrent_gated_delta_rule_packed_decode_kernel as kernel,
                )

                if hv not in announced:
                    print(
                        f"JARVIS_PUBLIC_GDN_RETUNED_ENABLED hv={hv} bv=16 warps=4 state_stride={initial_state.stride(0)}",
                        flush=True,
                    )
                    announced.add(hv)
                kernel[(8, hv)](
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    o=out,
                    h0=initial_state,
                    ht=initial_state,
                    ssm_state_indices=ssm_state_indices,
                    scale=scale,
                    stride_mixed_qkv_tok=mixed_qkv.stride(0),
                    stride_a_tok=a.stride(0),
                    stride_b_tok=b.stride(0),
                    stride_init_state_token=initial_state.stride(0),
                    stride_final_state_token=initial_state.stride(0),
                    stride_indices_seq=1,
                    H=16,
                    HV=hv,
                    K=128,
                    V=128,
                    BK=128,
                    BV=16,
                    SOFTPLUS_THRESHOLD=20.0,
                    USE_QK_L2NORM_IN_KERNEL=True,
                    SPLIT_BATCH_HEAD_GRID=False,
                    num_warps=4,
                    num_stages=3,
                )
                return out, initial_state
            bv, warps = (
                (4, 1) if os.environ.get("JARVIS_GDN_TILED") == "bv4" else (16, 4)
            )
            if hv not in announced:
                print(
                    f"JARVIS_GDN_TILED_ENABLED hv={hv} bv={bv} warps={warps} algebraic=0 state_stride={initial_state.stride(0)}",
                    flush=True,
                )
                announced.add(hv)
            step[(triton.cdiv(128, bv), hv)](
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                initial_state,
                ssm_state_indices,
                mixed_qkv,
                mixed_qkv,
                out,
                16,
                hv,
                bv,
                False,
                False,
                SS=initial_state.stride(0),
                num_warps=warps,
            )
            return out, initial_state

        module.fused_recurrent_gated_delta_rule_packed_decode = packed
        module._jarvis_state = True
