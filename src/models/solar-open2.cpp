// solar-open2.cpp -- llama.cpp arch implementation for upstage/Solar-Open2-250B
// Target: src/models/solar-open2.cpp
//
// DERIVATION (assembled from proven in-tree parts, not invented):
//   - KDA recurrent branch: adapted from src/models/kimi-linear.cpp (Solar's
//     36 linear-attention layers use the IDENTICAL tensor set: q/k/v_conv1d,
//     f_a/f_b, g_a/g_b, b_proj->beta, A_log (stored as -exp at convert),
//     dt_bias->ssm_dt bias, o_norm; sigmoid-gated RMSNorm output).
//   - GQA branch: standard grouped-query attention, NoPE (use_rope=false),
//     with head-wise sigmoid output gate g_proj (ATTN_GATE slot; same
//     pre-o_proj gating pattern as qwen3next gated attention).
//   - MoE: deepseek-v3 style -- 320 routed top-8 + 1 shared expert +
//     e_score_correction_bias, norm_topk_prob=true, scale 1.0. All 48
//     layers MoE (first_k_dense_replace=0); dense branch kept for safety.
//
// Layer schedule: GQA at il % 4 == 0 (0,4,...,44) carried in GGUF as
// per-layer n_head_kv array (8 = attention, 0 = KDA/recurrent), same
// convention as kimi-linear/jamba. hparams.is_recr(il) derives from it.
//
// Solar dims: n_embd 4096, n_layer 48, n_head 64, n_head_kv(attn) 8,
// head_dim 128 (both GQA and KDA), d_inner = 64*128 = 8192, d_conv 4,
// n_expert 320, n_expert_used 8, n_ff_exp 1280, shared 1, vocab 196608.

#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_solar_open2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // KDA layers are recurrent: n_head_kv == 0 marks them (kimi/jamba convention)
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    // MoE
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func,   false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        // e_score_correction_bias present => DeepSeek-V3-style noaux routing
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_250B_A15B; break; // Solar-Open2-250B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_solar_open2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int64_t head_dim   = hparams.n_embd_head_kda; // 128, same for GQA and KDA
    const int64_t d_inner    = head_dim * n_head;       // 8192
    const int64_t ssm_d_conv = hparams.ssm_d_conv;      // 4

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            // === KDA layer (identical slot set to kimi-linear) ===
            // conv1d: 4D [d_conv, 1, d_inner, 1], tolerate 3D after quant
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_k_conv) {
                layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_v_conv) {
                layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }

            // q,k,v projections all d_inner-wide on KDA layers
            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            // forget gate (low-rank): f_b(f_a(x))
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {head_dim, d_inner}, 0);

            // beta mixing coefficient
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // A (=-exp(A_log), applied at convert): [1, n_head, 1, 1] or [1, n_head]
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_a) {
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
            }

            // dt bias [d_inner]
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            // output gate (low-rank): g_b(g_a(x))
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {head_dim, d_inner}, 0);

            // gated RMSNorm before o_proj
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            // === GQA layer: 64 q-heads / 8 kv-heads, NoPE, head-wise out gate ===
            const int64_t n_head_kv_i  = hparams.n_head_kv(i); // 8
            const int64_t n_embd_gqa_i = head_dim * n_head_kv_i; // 1024

            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, d_inner}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa_i}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa_i}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);

            // head-wise output gate (self_attn.g_proj -> ATTN_GATE slot)
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, d_inner}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        const int64_t n_ff_exp = hparams.n_ff_exp; // 1280

        if (i < (int) hparams.n_layer_dense_lead) {
            // dense lead layers (Solar: n_layer_dense_lead = 0, branch unused but kept)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            const int64_t n_ff_shexp = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);

            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_solar_open2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal conv1d for Q/K/V on KDA layers -- verbatim mechanism from
// kimi-linear.cpp (qkv: 0=Q 1=K 2=V; states packed [Q|K|V] per seq).
static ggml_tensor * solar_causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
        ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv,
        ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head,
        int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size; // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_solar_open2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // Solar: use_rope = false everywhere (NoPE GQA + recurrent KDA) -> no inp_pos.

    auto * inp_kv      = build_inp_mem_hybrid();
    auto * inp_rs      = inp_kv->get_recr();
    auto * inp_attn_kv = inp_kv->get_attn();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head       = hparams.n_head();          // 64
    const int64_t head_dim     = hparams.n_embd_head_kda;   // 128
    const int64_t d_conv       = hparams.ssm_d_conv;        // 4
    const int64_t d_inner      = n_head * head_dim;         // 8192
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // === KDA layer (kimi-linear mechanism, verbatim) ===
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = solar_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = solar_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = solar_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = A * softplus(f_b(f_a(x)) + dt_bias)   (A = -exp(A_log), baked at convert)
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            cb(g1, "kda_f", il);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            // Solar: kda_gate_lower_bound = -5.0 (upstage modeling_solar_open2
            // torch_kda_gate: g = clamp(g, min=lower_bound)). g <= 0 always.
            g1 = ggml_clamp(ctx0, g1, -5.0f, 0.0f);
            cb(g1, "kda_g1", il);
            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);
            // Solar: kda_allow_neg_eigval = true -> beta = sigmoid(.) * 2.0
            // (upstage reference: "if self.allow_neg_eigval: beta = beta * 2.0")
            beta = ggml_scale(ctx0, beta, 2.0f);
            cb(beta, "kda_beta", il);

            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;
            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            cb(output, "kda_attn_out", il);

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // out gate g2 = g_b(g_a(x)); Solar KDA: sigmoid-gated RMSNorm (kimi convention)
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            cb(g2, "kda_g2", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
            ggml_tensor * gated = ggml_mul(ctx0, normed, gate);

            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
        } else {
            // === GQA layer: NoPE + head-wise sigmoid output gate ===
            const int64_t n_head_kv_i = hparams.n_head_kv(il); // 8

            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head,      n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_head_kv_i, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_head_kv_i, n_tokens);
            cb(Qcur, "gqa_Q", il);
            cb(Kcur, "gqa_K", il);
            cb(Vcur, "gqa_V", il);

            // NoPE: no rope application, positions handled by KDA recurrence.
            // wo = nullptr -> raw concatenated head output, gate applies pre-o_proj.
            ggml_tensor * attn = build_attn(inp_attn_kv, nullptr, NULL, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(attn, "gqa_attn_raw", il);

            // head-wise output gate: attn * sigmoid(g_proj(x))
            ggml_tensor * g = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
            g = ggml_sigmoid(ctx0, g);
            cb(g, "gqa_gate", il);
            attn = ggml_mul(ctx0, attn, g);

            cur = ggml_mul_mat(ctx0, layer.wo, attn);
            cb(cur, "gqa_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
