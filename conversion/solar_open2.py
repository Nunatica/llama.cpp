from __future__ import annotations

"""Solar-Open2 (upstage/Solar-Open2-250B) converter for llama.cpp.

Arch: 48-layer hybrid. 36 KDA layers (Kimi Delta Attention -- IDENTICAL
tensor names to Kimi-Linear: q/k/v_conv1d, f_a/f_b_proj, g_a/g_b_proj,
b_proj, A_log, dt_bias, o_norm) + 12 GQA layers (idx 0,4,...,44: standard
attention 64q/8kv hd128, NO rope, head-wise output gate g_proj) + MoE on
every layer (320 routed top-8 + 1 shared, DeepSeek-V3-style
e_score_correction_bias router, norm_topk_prob).

Reuses existing gguf tensor slots: SSM_CONV1D_Q/K/V, SSM_F_A/F_B,
SSM_G_A/G_B, SSM_BETA, SSM_A, SSM_DT, SSM_NORM (Kimi Linear slots),
ATTN_GATE (step3.5 slot), FFN_*_EXP / FFN_*_SHEXP / FFN_EXP_PROBS_B
(deepseek slots). New MODEL_ARCH.SOLAR_OPEN2 required in gguf-py
constants (see patches/gguf_constants.patch.md).
"""

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("SolarOpen2ForCausalLM")
class SolarOpen2Model(TextModel):
    """Upstage Solar-Open2: hybrid KDA + gated-GQA, 320-expert MoE."""
    model_arch = gguf.MODEL_ARCH.SOLAR_OPEN2

    _experts: list[dict[str, Tensor]] | None = None

    def set_vocab(self):
        # tokenizer.json ships in-repo; standard BPE path.
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        n_layer = hp["num_hidden_layers"]

        self.gguf_writer.add_vocab_size(hp["vocab_size"])

        # --- hybrid layer schedule -------------------------------------
        # gqa_layers is 0-indexed (0,4,...,44). Kimi convention:
        # per-layer n_head_kv, 0 => KDA (recurrent), >0 => full attention.
        gqa_layers = set(hp["gqa_layers"])
        n_kv = hp["num_key_value_heads"]
        head_counts_kv = [n_kv if il in gqa_layers else 0 for il in range(n_layer)]
        self.gguf_writer.add_head_count_kv(head_counts_kv)

        # --- KDA params (linear_attn_config) ----------------------------
        lac = hp["linear_attn_config"]
        self.gguf_writer.add_ssm_conv_kernel(lac["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(lac["head_dim"])

        # --- attention geometry -----------------------------------------
        # use_rope=false: GQA layers are NoPE (position comes from the
        # recurrent KDA layers). rope_dimension_count=0 signals no rope;
        # C++ arch must respect this (see patches/llama_cpp_arch.md).
        self.gguf_writer.add_rope_dimension_count(0)
        self.gguf_writer.add_rope_freq_base(hp.get("rope_theta", 10000))

        # --- MoE ----------------------------------------------------------
        self.gguf_writer.add_expert_count(hp["n_routed_experts"])
        self.gguf_writer.add_expert_used_count(hp["num_experts_per_tok"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        if hp.get("norm_topk_prob", False):
            self.gguf_writer.add_expert_weights_norm(True)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftovers = [k for d in self._experts for k in d.keys()]
            if leftovers:
                raise ValueError(f"Unprocessed experts: {leftovers[:8]} "
                                 f"(+{max(0, len(leftovers)-8)} more)")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # --- KDA conv1d: HF [d_inner, d_conv] -> numpy (1, d_inner, 1, d_conv)
        # so gguf writes ggml ne = [d_conv, 1, d_inner, 1] (Kimi convention;
        # memory layout identical, conv step fastest).
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
            elif data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # --- KDA decay: store A = -exp(A_log) at convert time (Kimi).
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)

        # --- dt_bias -> dt_proj.bias so it lands in the SSM_DT slot (Kimi).
        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # --- router e_score bias -> deepseek-v3 slot name.
        if name.endswith("mlp.gate.e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        # --- stack the 320 routed experts into fused 3D tensors ---------
        if ".mlp.experts." in name:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name, tname in [("gate_proj", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                      ("down_proj", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                      ("up_proj",   gguf.MODEL_TENSOR.FFN_UP_EXP)]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    stacked = torch.stack(datas, dim=0)
                    new_name = self.format_tensor_name(tname, bid)
                    yield from super().modify_tensors(stacked, new_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
