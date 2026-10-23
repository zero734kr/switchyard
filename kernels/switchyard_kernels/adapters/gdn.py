"""GDN integration: paired QKVZ/BA projection and the retuned state kernel.

Two independent patches, both opt-in through the adapter and both guarded by
exact shape, dtype, stride and configuration checks:

* the projection pair replaces two launches with one Triton kernel for a single
  decode token, and falls back to two ``torch.mm`` calls for larger batches;
* the state patch launches the bundled public recurrence kernel with a tuned
  tiling instead of the installed default.
"""

import json

import torch
import triton

from ..announce import announce, announce_payload
from ..ops.gdn_step import step
from ..ops.projection_pair import project_pair

PAIR_ATTRIBUTE = "_switchyard_pair"
STATE_ATTRIBUTE = "_switchyard_state"

#: hidden size -> (QKVZ rows, BA rows) for the two served checkpoints.
PAIR_SHAPES = {5120: (16384, 96), 2048: (12288, 64)}
#: Retuned public kernel: (block over V, warps).
PUBLIC_TILING = (16, 4)


@torch.library.custom_op("jarvis_recipe::project_pair", mutates_args=())
def pair(
    x: torch.Tensor, wq: torch.Tensor, wb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    oq = x.new_empty((x.shape[0], wq.shape[0]))
    ob = x.new_empty((x.shape[0], wb.shape[0]))
    hidden, dq, db = wq.shape[1], wq.shape[0], wb.shape[0]
    if x.shape[0] == 1:
        announce("JARVIS_PROJECTION_PAIR_ENABLED", key=hidden, h=hidden, dq=dq, db=db)
        rows, warps = (1, 4) if hidden == 5120 else (2, 8)
        project_pair[(triton.cdiv(dq, rows) + triton.cdiv(db, rows),)](
            x,
            wq,
            wb,
            oq,
            ob,
            hidden,
            dq,
            db,
            triton.next_power_of_2(hidden),
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


def _pair_supported(module, x, wq, wb):
    expected = PAIR_SHAPES.get(x.shape[-1])
    return (
        x.dtype is torch.bfloat16
        and x.is_contiguous()
        and expected is not None
        and module.tp_size == 1
        and not module.gqa_interleaved_layout
        and module.enable_fused_gdn_decode
        and tuple(wq.shape) == (expected[0], x.shape[-1])
        and tuple(wb.shape) == (expected[1], x.shape[-1])
        and wq.dtype is wb.dtype
        and wq.dtype is torch.bfloat16
        and wq.is_contiguous()
        and wb.is_contiguous()
        and module.in_proj_qkvz.bias is None
        and module.in_proj_ba.bias is None
        and type(module.in_proj_qkvz.quant_method).__name__ == "UnquantizedLinearMethod"
        and type(module.in_proj_ba.quant_method).__name__ == "UnquantizedLinearMethod"
    )


def _install_pair(module):
    cls = module.QwenGatedDeltaNetAttention
    if getattr(cls, PAIR_ATTRIBUTE, False):
        return
    original = cls.forward_cuda

    def forward_cuda(self, x):
        wq, wb = self.in_proj_qkvz.weight, self.in_proj_ba.weight
        if not _pair_supported(self, x, wq, wb):
            return original(self, x)
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

    cls.forward_cuda = forward_cuda
    setattr(cls, PAIR_ATTRIBUTE, True)


def _encode_layer_name(prefix):
    from vllm.utils.torch_utils import _encode_layer_name as encode

    return encode(prefix)


def _state_supported(
    mixed_qkv, a, b, A_log, dt_bias, initial_state, out, ssm_state_indices, scale
):
    hv, v, k = initial_state.shape[-3:]
    return (
        mixed_qkv.shape[0] == 1
        and hv in (32, 48)
        and v == k == 128
        and mixed_qkv.shape[1] == (32 + hv) * 128
        and mixed_qkv.dtype == a.dtype == b.dtype == out.dtype == torch.bfloat16
        and initial_state.dtype == torch.float32
        and tuple(initial_state.stride()[-3:]) == (16384, 128, 1)
        and initial_state.stride(0) >= hv * 16384
        and mixed_qkv.stride(-1) == a.stride(-1) == b.stride(-1) == 1
        and scale == 128.0**-0.5
        and all(
            tensor.is_contiguous()
            for tensor in (A_log, dt_bias, out, ssm_state_indices)
        )
    )


def _install_state(module):
    if getattr(module, STATE_ATTRIBUTE, False):
        return
    original = module.fused_recurrent_gated_delta_rule_packed_decode

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
        if not (
            use_qk_l2norm_in_kernel
            and _state_supported(
                mixed_qkv,
                a,
                b,
                A_log,
                dt_bias,
                initial_state,
                out,
                ssm_state_indices,
                scale,
            )
        ):
            return original(
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
        hv = initial_state.shape[-3]
        from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode_kernel as kernel,
        )

        bv, warps = PUBLIC_TILING
        announce(
            "JARVIS_PUBLIC_GDN_RETUNED_ENABLED",
            key=hv,
            hv=hv,
            bv=bv,
            warps=warps,
            state_stride=initial_state.stride(0),
        )
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
            BV=bv,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=True,
            SPLIT_BATCH_HEAD_GRID=False,
            num_warps=warps,
            num_stages=3,
        )
        return out, initial_state

    module.fused_recurrent_gated_delta_rule_packed_decode = packed
    setattr(module, STATE_ATTRIBUTE, True)


def _announce_configuration(cls):
    if getattr(cls, "_switchyard_config", False):
        return
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        announce_payload(
            "JARVIS_PAIR_CONFIGURATION",
            json.dumps(
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
            key="once",
        )

    cls.__init__ = init
    cls._switchyard_config = True


def install():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as module

    _announce_configuration(module.QwenGatedDeltaNetAttention)
    _install_pair(module)
    _install_state(module)


__all__ = ["install", "pair"]
