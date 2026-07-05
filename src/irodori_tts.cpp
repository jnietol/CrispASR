// irodori_tts.cpp — native ggml runtime for Aratako/Irodori-TTS.
//
// Architecture:
//   1. TextEncoder: Embedding(102400, 1280) + 14 TextBlock (RoPE self-attn + SwiGLU)
//   2. ReferenceLatentEncoder: Linear(128→1280) + 14 TextBlock (same structure)
//   3. TimestepEmbedding: sinusoidal(512) → Linear(512,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048,6144)
//   4. DiT: 24 DiffusionBlock, each:
//      - LowRankAdaLN(rank=256): timestep → shift/scale/gate via low-rank residual MLP
//      - JointAttention: self-KV + text-context-KV + speaker-context-KV, half-RoPE
//      - SwiGLU(2048, 5888)
//   5. DurationPredictor: token_sum with speaker AdaRN-Zero
//   6. Euler RF ODE solver: 40 steps with independent CFG
//   7. DAC-VAE decoder (separate model, handled externally for now)
//
// The implementation builds ggml graphs per inference stage using gallocr.

#include "irodori_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/sentencepiece.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Debug gating ────────────────────────────────────────────────────

static bool irodori_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_IRODORI_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

#define IRODORI_DBG(...)                                                                                               \
    do {                                                                                                               \
        if (irodori_debug_enabled())                                                                                   \
            std::fprintf(stderr, __VA_ARGS__);                                                                         \
    } while (0)

// ── Hyperparameters ─────────────────────────────────────────────────

struct irodori_hparams {
    int latent_dim = 128;
    int latent_patch_size = 1;
    int model_dim = 2048;
    int num_layers = 24;
    int num_heads = 16;
    float mlp_ratio = 2.875f;
    int text_dim = 1280;
    int text_layers = 14;
    int text_heads = 10;
    float text_mlp_ratio = 2.6f;
    int text_vocab_size = 102400;
    int speaker_dim = 1280;
    int speaker_layers = 14;
    int speaker_heads = 10;
    float speaker_mlp_ratio = 2.6f;
    int speaker_patch_size = 1;
    int timestep_embed_dim = 512;
    int adaln_rank = 256;
    float norm_eps = 1e-5f;
    int use_duration_predictor = 1;
    int duration_aux_dim = 14;
    int duration_hidden_dim = 1024;
    int duration_layers = 3;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    int sample_rate = 48000;
    int codec_hop_length = 1920; // Semantic-DACVAE-Japanese-32dim: strides [12,10,8,2] = 1920

    int patched_latent_dim() const { return latent_dim * latent_patch_size; }
    int head_dim() const { return model_dim / num_heads; }
    int text_head_dim() const { return text_dim / text_heads; }
    int speaker_head_dim() const { return speaker_dim / speaker_heads; }
    int ff_dim() const { return (int)(model_dim * mlp_ratio); }
    int text_ff_dim() const { return (int)(text_dim * text_mlp_ratio); }
    int speaker_ff_dim() const { return (int)(speaker_dim * speaker_mlp_ratio); }
};

// ── Weight structures ───────────────────────────────────────────────

struct irodori_text_block_weights {
    ggml_tensor* attn_norm = nullptr; // (text_dim,)
    ggml_tensor* wq = nullptr;        // (text_dim, text_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;   // gated attention
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* mlp_norm = nullptr; // (text_dim,)
    ggml_tensor* mlp_w1 = nullptr;   // gate (text_dim, ff_dim)
    ggml_tensor* mlp_w2 = nullptr;   // down (ff_dim, text_dim)
    ggml_tensor* mlp_w3 = nullptr;   // up (text_dim, ff_dim)
};

struct irodori_low_rank_adaln_weights {
    // shift/scale/gate each have: down (model_dim, rank) + up (rank, model_dim) + up_bias (model_dim,)
    ggml_tensor* shift_down = nullptr;
    ggml_tensor* shift_up_w = nullptr;
    ggml_tensor* shift_up_b = nullptr;
    ggml_tensor* scale_down = nullptr;
    ggml_tensor* scale_up_w = nullptr;
    ggml_tensor* scale_up_b = nullptr;
    ggml_tensor* gate_down = nullptr;
    ggml_tensor* gate_up_w = nullptr;
    ggml_tensor* gate_up_b = nullptr;
};

struct irodori_dit_block_weights {
    // JointAttention
    ggml_tensor* wq = nullptr; // (model_dim, model_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* wk_text = nullptr; // (text_dim, model_dim)
    ggml_tensor* wv_text = nullptr;
    ggml_tensor* wk_spk = nullptr; // (speaker_dim, model_dim)
    ggml_tensor* wv_spk = nullptr;

    // LowRankAdaLN for attention and MLP
    irodori_low_rank_adaln_weights adaln_attn;
    irodori_low_rank_adaln_weights adaln_mlp;

    // SwiGLU
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
};

struct irodori_dur_block_weights {
    ggml_tensor* norm = nullptr; // (hidden_dim,)
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
    ggml_tensor* mod_w = nullptr; // (speaker_dim, hidden_dim*3)
    ggml_tensor* mod_b = nullptr; // (hidden_dim*3,)
};

struct irodori_weights {
    // Text encoder
    ggml_tensor* text_emb = nullptr; // (vocab_size, text_dim)
    std::vector<irodori_text_block_weights> text_blocks;
    ggml_tensor* text_norm = nullptr; // (text_dim,)

    // Speaker encoder
    ggml_tensor* spk_in_proj_w = nullptr; // (patched_latent_dim, speaker_dim)
    ggml_tensor* spk_in_proj_b = nullptr; // (speaker_dim,)
    std::vector<irodori_text_block_weights> spk_blocks;
    ggml_tensor* spk_norm = nullptr; // (speaker_dim,)

    // Timestep conditioning MLP
    ggml_tensor* cond_0_w = nullptr; // (timestep_embed_dim, model_dim)
    ggml_tensor* cond_2_w = nullptr; // (model_dim, model_dim)
    ggml_tensor* cond_4_w = nullptr; // (model_dim, model_dim*3)

    // Input/output projections
    ggml_tensor* in_proj_w = nullptr;  // (patched_latent_dim, model_dim)
    ggml_tensor* in_proj_b = nullptr;  // (model_dim,)
    ggml_tensor* out_norm = nullptr;   // (model_dim,)
    ggml_tensor* out_proj_w = nullptr; // (model_dim, patched_latent_dim)
    ggml_tensor* out_proj_b = nullptr; // (patched_latent_dim,)

    // DiT blocks
    std::vector<irodori_dit_block_weights> dit_blocks;

    // Duration predictor
    ggml_tensor* dur_input_proj_w = nullptr;
    ggml_tensor* dur_input_proj_b = nullptr;
    ggml_tensor* dur_null_speaker = nullptr;
    std::vector<irodori_dur_block_weights> dur_blocks;
    ggml_tensor* dur_out_norm = nullptr;
    ggml_tensor* dur_out_proj_w = nullptr;
    ggml_tensor* dur_out_proj_b = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────

struct irodori_tts_context {
    irodori_hparams hparams;
    irodori_weights weights;

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_context* w_ctx = nullptr; // weight tensor context (kept alive)

    // Tokenizer (SentencePiece unigram)
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float> token_scores;
    int bos_token_id = -1;
    int pad_token_id = -1;

    // Reference latent (speaker conditioning)
    std::vector<float> ref_latent; // (T_ref * latent_dim)
    int ref_latent_frames = 0;

    // Runtime params
    int seed = 0;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    float speed = 1.0f;
    float duration_scale = 1.0f;
    float max_seconds = 30.0f;
    int verbosity = 1;
    int n_threads = 4;
};

// ── RoPE precomputation ─────────────────────────────────────────────

static void precompute_freqs_cis(int dim, int max_len, float theta, std::vector<float>& cos_cache,
                                 std::vector<float>& sin_cache) {
    int half = dim / 2;
    cos_cache.resize(max_len * half);
    sin_cache.resize(max_len * half);
    for (int t = 0; t < max_len; t++) {
        for (int k = 0; k < half; k++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * k) / (float)dim);
            float angle = (float)t * freq;
            cos_cache[t * half + k] = std::cos(angle);
            sin_cache[t * half + k] = std::sin(angle);
        }
    }
}

// ── RMSNorm (graph op) ──────────────────────────────────────────────

static ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    x = ggml_rms_norm(ctx, x, eps);
    if (weight->type != GGML_TYPE_F32)
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    return ggml_mul(ctx, x, weight);
}

// mul_mat + cast to F32 (weights may be F16/Q4_K, need F32 for element-wise ops)
static ggml_tensor* mul_mat_f32(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (y->type != GGML_TYPE_F32)
        y = ggml_cast(ctx, y, GGML_TYPE_F32);
    return y;
}

// ── Timestep embedding (CPU) ────────────────────────────────────────

static void compute_timestep_embedding(float t, int dim, std::vector<float>& out) {
    out.resize(dim);
    int half = dim / 2;
    for (int k = 0; k < half; k++) {
        float freq = 1000.0f * std::exp(-std::log(10000.0f) * (float)k / (float)half);
        float arg = t * freq;
        out[k] = std::cos(arg);
        out[half + k] = std::sin(arg);
    }
}

// ── RoPE is applied via ggml_rope_ext in the graph builders below. ──

// ── Graph builders ──────────────────────────────────────────────────

// Evaluate a ggml graph on the backend using gallocr.
static bool eval_graph(ggml_backend_t backend, ggml_cgraph* gf) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        return false;
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(galloc);
    return true;
}

// ── GGUF loading ────────────────────────────────────────────────────

// Load weights using the core_gguf map-based getter pattern.
template <typename GetFn, typename TryGetFn>
static bool load_weights_from_map(irodori_tts_context* ctx, GetFn get, TryGetFn /*try_get*/) {
    auto& w = ctx->weights;
    const auto& hp = ctx->hparams;

    // Text encoder
    w.text_emb = get("irodori.text_enc.emb");
    w.text_norm = get("irodori.text_norm");
    w.text_blocks.resize(hp.text_layers);
    for (int i = 0; i < hp.text_layers; i++) {
        auto& blk = w.text_blocks[i];
        std::string p = "irodori.text_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Speaker encoder
    w.spk_in_proj_w = get("irodori.spk_enc.in_proj.w");
    w.spk_in_proj_b = get("irodori.spk_enc.in_proj.b");
    w.spk_norm = get("irodori.spk_norm");
    w.spk_blocks.resize(hp.speaker_layers);
    for (int i = 0; i < hp.speaker_layers; i++) {
        auto& blk = w.spk_blocks[i];
        std::string p = "irodori.spk_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Top-level
    w.cond_0_w = get("irodori.cond.0.w");
    w.cond_2_w = get("irodori.cond.2.w");
    w.cond_4_w = get("irodori.cond.4.w");
    w.in_proj_w = get("irodori.in_proj.w");
    w.in_proj_b = get("irodori.in_proj.b");
    w.out_norm = get("irodori.out_norm");
    w.out_proj_w = get("irodori.out_proj.w");
    w.out_proj_b = get("irodori.out_proj.b");

    // DiT blocks
    w.dit_blocks.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        auto& blk = w.dit_blocks[i];
        std::string p = "irodori.dit.blk." + std::to_string(i) + ".";
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.wk_text = get(p + "attn.wk_text");
        blk.wv_text = get(p + "attn.wv_text");
        blk.wk_spk = get(p + "attn.wk_spk");
        blk.wv_spk = get(p + "attn.wv_spk");

        // AdaLN attention
        auto load_adaln = [&](irodori_low_rank_adaln_weights& adaln, const std::string& ap) {
            adaln.shift_down = get(ap + "shift_down");
            adaln.shift_up_w = get(ap + "shift_up.w");
            adaln.shift_up_b = get(ap + "shift_up.b");
            adaln.scale_down = get(ap + "scale_down");
            adaln.scale_up_w = get(ap + "scale_up.w");
            adaln.scale_up_b = get(ap + "scale_up.b");
            adaln.gate_down = get(ap + "gate_down");
            adaln.gate_up_w = get(ap + "gate_up.w");
            adaln.gate_up_b = get(ap + "gate_up.b");
        };
        load_adaln(blk.adaln_attn, p + "adaln_attn.");
        load_adaln(blk.adaln_mlp, p + "adaln_mlp.");

        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Duration predictor
    if (hp.use_duration_predictor) {
        w.dur_input_proj_w = get("irodori.dur.input_proj.w");
        w.dur_input_proj_b = get("irodori.dur.input_proj.b");
        w.dur_null_speaker = get("irodori.dur.null_speaker");
        w.dur_blocks.resize(hp.duration_layers);
        for (int i = 0; i < hp.duration_layers; i++) {
            auto& blk = w.dur_blocks[i];
            std::string p = "irodori.dur.blk." + std::to_string(i) + ".";
            blk.norm = get(p + "norm");
            blk.mlp_w1 = get(p + "mlp.w1");
            blk.mlp_w2 = get(p + "mlp.w2");
            blk.mlp_w3 = get(p + "mlp.w3");
            blk.mod_w = get(p + "mod.w");
            blk.mod_b = get(p + "mod.b");
        }
        w.dur_out_norm = get("irodori.dur.out_norm");
        w.dur_out_proj_w = get("irodori.dur.out_proj.w");
        w.dur_out_proj_b = get("irodori.dur.out_proj.b");
    }

    return true;
}

// ── Graph: Text encoder forward ─────────────────────────────────────

// Build text encoder graph for a batch of token IDs.
// Input: token_ids (T_text,) I32, pos_ids (T_text,) I32
// Output: text_state (text_dim, T_text) F32
static ggml_tensor* build_text_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                             ggml_tensor* token_ids, ggml_tensor* mask_f, ggml_tensor* pos_ids,
                                             int T_text) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Embedding lookup
    ggml_tensor* x = ggml_get_rows(ctx, w.text_emb, token_ids);
    // x: (text_dim, T_text) after get_rows
    x = ggml_cast(ctx, x, GGML_TYPE_F32);

    // Apply mask (zero out padding)
    x = ggml_mul(ctx, x, mask_f);

    // Process through transformer blocks
    for (int i = 0; i < hp.text_layers; i++) {
        const auto& blk = w.text_blocks[i];

        // Self-attention with pre-norm
        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        // Q, K, V projections (F32 output for element-wise ops)
        ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h);
        ggml_tensor* k = mul_mat_f32(ctx, blk.wk, h);
        ggml_tensor* v = mul_mat_f32(ctx, blk.wv, h);

        // Reshape to (head_dim, heads, T)
        int hd = hp.text_head_dim();
        int nh = hp.text_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_text);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_text);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_text);

        // QK norm (per-head RMSNorm) — cast norms to F32 if F16
        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

        // RoPE in (hd, nh, T) layout — ne[2]=T matches pos_ids length
        q = ggml_rope_ext(ctx, q, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to (hd, T, nh) for flash_attn_ext
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_reshape_3d(ctx, v, hd, nh, T_text);
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        // Flash attention: Q(hd, T, nh), K(hd, T, nh), V(hd, T, nh)
        ggml_tensor* attn_out =
            ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.text_dim, T_text);

        // Gated attention: y = y * sigmoid(gate(h))
        ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);

        // Output projection
        attn_out = mul_mat_f32(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        // FFN with pre-norm (use mul_mat_f32 for F16/Q4K weight safety)
        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h);
        ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h);
        ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
        mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
        x = ggml_add(ctx, residual, mlp_out);

        // Re-apply mask
        x = ggml_mul(ctx, x, mask_f);
    }

    // Final text norm
    x = rms_norm(ctx, x, w.text_norm, hp.norm_eps);
    return x;
}

// ── Graph: Speaker encoder forward ──────────────────────────────────

static ggml_tensor* build_speaker_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                                ggml_tensor* latent_in, ggml_tensor* mask_f, ggml_tensor* pos_ids,
                                                int T_ref) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Input projection + scale by 1/6
    ggml_tensor* x = mul_mat_f32(ctx, w.spk_in_proj_w, latent_in);
    x = ggml_add(ctx, x, w.spk_in_proj_b);
    x = ggml_scale(ctx, x, 1.0f / 6.0f);
    x = ggml_mul(ctx, x, mask_f);

    // Same transformer block structure as text encoder (with same fixes)
    for (int i = 0; i < hp.speaker_layers; i++) {
        const auto& blk = w.spk_blocks[i];

        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h);
        ggml_tensor* k = mul_mat_f32(ctx, blk.wk, h);
        ggml_tensor* v = mul_mat_f32(ctx, blk.wv, h);

        int hd = hp.speaker_head_dim();
        int nh = hp.speaker_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_ref);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_ref);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_ref);

        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

        // RoPE in (hd, nh, T) layout then permute to (hd, T, nh) for flash attn
        q = ggml_rope_ext(ctx, q, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.speaker_dim, T_ref);

        ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);
        attn_out = mul_mat_f32(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h);
        ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h);
        ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
        mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
        x = ggml_add(ctx, residual, mlp_out);
        x = ggml_mul(ctx, x, mask_f);
    }

    // Speaker norm
    x = rms_norm(ctx, x, w.spk_norm, hp.norm_eps);
    return x;
}

// ── Graph: LowRankAdaLN ─────────────────────────────────────────────

// Apply LowRank AdaLN: returns (normed_modulated_x, gate)
// cond_embed: (model_dim*3, 1) — pre-chunked into 3 parts of model_dim each
static void apply_low_rank_adaln(ggml_context* ctx, const irodori_low_rank_adaln_weights& adaln, ggml_tensor* x,
                                 ggml_tensor* cond_shift, ggml_tensor* cond_scale, ggml_tensor* cond_gate,
                                 float norm_eps, ggml_tensor** out_x, ggml_tensor** out_gate) {
    // Low-rank residual modulation:
    //   shift = shift_up(shift_down(silu(cond_shift))) + cond_shift
    //   scale = scale_up(scale_down(silu(cond_scale))) + cond_scale
    //   gate  = gate_up(gate_down(silu(cond_gate))) + cond_gate

    auto lr_mod = [&](ggml_tensor* cond, ggml_tensor* down, ggml_tensor* up_w, ggml_tensor* up_b) -> ggml_tensor* {
        ggml_tensor* h = ggml_silu(ctx, cond);
        h = mul_mat_f32(ctx, down, h);
        h = mul_mat_f32(ctx, up_w, h);
        if (up_b)
            h = ggml_add(ctx, h, up_b);
        return ggml_add(ctx, h, cond); // residual connection
    };

    ggml_tensor* shift =
        ggml_cast(ctx, lr_mod(cond_shift, adaln.shift_down, adaln.shift_up_w, adaln.shift_up_b), GGML_TYPE_F32);
    ggml_tensor* scale =
        ggml_cast(ctx, lr_mod(cond_scale, adaln.scale_down, adaln.scale_up_w, adaln.scale_up_b), GGML_TYPE_F32);
    ggml_tensor* gate =
        ggml_cast(ctx, lr_mod(cond_gate, adaln.gate_down, adaln.gate_up_w, adaln.gate_up_b), GGML_TYPE_F32);

    // RMSNorm + modulate: x_normed * (1 + scale) + shift
    ggml_tensor* x_norm = ggml_rms_norm(ctx, x, norm_eps);
    // (1 + scale) = scale + 1: use ggml_scale to add 1 via add_scalar trick
    // Actually: x * (1 + s) = x + x*s, which avoids allocating a scalar tensor
    ggml_tensor* x_scaled = ggml_add(ctx, x_norm, ggml_mul(ctx, x_norm, scale));
    *out_x = ggml_add(ctx, x_scaled, shift);

    // gate = tanh(gate)
    *out_gate = ggml_tanh(ctx, gate);
}

// ── Graph: DiT block forward ────────────────────────────────────────

// Builds one DiffusionBlock.
// x: (model_dim, T_latent)
// cond_embed: (model_dim*3, 1) — timestep conditioning
// text_state: (text_dim, T_text)
// spk_state: (speaker_dim, T_ref) or nullptr
static ggml_tensor* build_dit_block_graph(ggml_context* ctx, const irodori_tts_context* model,
                                          const irodori_dit_block_weights& blk, ggml_tensor* x, ggml_tensor* cond_embed,
                                          ggml_tensor* text_state, int T_text, ggml_tensor* spk_state, int T_ref,
                                          int T_latent, ggml_tensor* pos_ids) {
    const auto& hp = model->hparams;
    int hd = hp.head_dim();
    int nh = hp.num_heads;

    // Ensure all inputs are F32
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    cond_embed = ggml_cast(ctx, cond_embed, GGML_TYPE_F32);
    text_state = ggml_cast(ctx, text_state, GGML_TYPE_F32);
    if (spk_state)
        spk_state = ggml_cast(ctx, spk_state, GGML_TYPE_F32);

    // Null check all weight tensors
    if (!blk.wq || !blk.wk || !blk.wv || !blk.wo || !blk.gate || !blk.q_norm || !blk.k_norm || !blk.wk_text ||
        !blk.wv_text || !blk.adaln_attn.shift_down || !blk.adaln_attn.shift_up_w || !blk.adaln_mlp.shift_down ||
        !blk.mlp_w1 || !blk.mlp_w2 || !blk.mlp_w3) {
        std::fprintf(stderr, "[irodori] ERROR: null weight tensor in DiT block\n");
        return x; // bail out
    }

    // Split cond_embed into 3 parts: (model_dim,) each
    int D = hp.model_dim;
    IRODORI_DBG("[irodori]       view cond_embed (D=%d, ne0=%d)...\n", D, (int)cond_embed->ne[0]);
    ggml_tensor* cond_shift = ggml_view_1d(ctx, cond_embed, D, 0);
    ggml_tensor* cond_scale = ggml_view_1d(ctx, cond_embed, D, D * sizeof(float));
    ggml_tensor* cond_gate_raw = ggml_view_1d(ctx, cond_embed, D, 2 * D * sizeof(float));
    IRODORI_DBG("[irodori]       applying AdaLN attn...\n");

    // ── Attention path ──
    ggml_tensor* h_attn = nullptr;
    ggml_tensor* gate_attn = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_attn, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_attn,
                         &gate_attn);
    IRODORI_DBG("[irodori]       AdaLN attn done\n");

    // Self Q/K/V
    IRODORI_DBG("[irodori]       QKV self proj (wq: %dx%d, h_attn: %dx%d)...\n", (int)blk.wq->ne[0], (int)blk.wq->ne[1],
                (int)h_attn->ne[0], (int)h_attn->ne[1]);
    ggml_tensor* q = mul_mat_f32(ctx, blk.wq, h_attn);
    ggml_tensor* k_self = mul_mat_f32(ctx, blk.wk, h_attn);
    ggml_tensor* v_self = mul_mat_f32(ctx, blk.wv, h_attn);
    IRODORI_DBG("[irodori]       QKV self done\n");

    // Text context K/V
    IRODORI_DBG("[irodori]       text KV proj (wk_text: %dx%d, text_state: %dx%d)...\n", (int)blk.wk_text->ne[0],
                (int)blk.wk_text->ne[1], (int)text_state->ne[0], (int)text_state->ne[1]);
    ggml_tensor* k_text = mul_mat_f32(ctx, blk.wk_text, text_state);
    ggml_tensor* v_text = mul_mat_f32(ctx, blk.wv_text, text_state);
    IRODORI_DBG("[irodori]       text KV done\n");

    // Speaker context K/V
    ggml_tensor* k_spk = nullptr;
    ggml_tensor* v_spk = nullptr;
    int T_kv_total = T_latent + T_text;
    if (spk_state && T_ref > 0) {
        k_spk = mul_mat_f32(ctx, blk.wk_spk, spk_state);
        v_spk = mul_mat_f32(ctx, blk.wv_spk, spk_state);
        T_kv_total += T_ref;
    }

    // All tensors start as (hd*nh, T) from mul_mat → reshape to (hd, nh, T)
    // Apply QK norm and half-RoPE in (hd, nh, T) layout (ne[2]=T matches pos_ids)
    // Then permute to (hd, T, nh) for flash_attn_ext

    q = ggml_reshape_3d(ctx, q, hd, nh, T_latent);
    q = ggml_rms_norm(ctx, q, hp.norm_eps);
    q = ggml_mul(ctx, q, ggml_cast(ctx, blk.q_norm, GGML_TYPE_F32));

    k_self = ggml_reshape_3d(ctx, k_self, hd, nh, T_latent);
    k_self = ggml_rms_norm(ctx, k_self, hp.norm_eps);
    k_self = ggml_mul(ctx, k_self, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));

    v_self = ggml_reshape_3d(ctx, v_self, hd, nh, T_latent);

    // Half-RoPE in (hd, nh, T) layout: split heads dim (ne[1]) in half
    {
        int half_nh = nh / 2;
        size_t nb1 = q->nb[1]; // stride for head dim
        size_t nb2 = q->nb[2]; // stride for T dim

        // Q: first half_nh heads get RoPE
        ggml_tensor* q_rot = ggml_view_3d(ctx, q, hd, half_nh, T_latent, nb1, nb2, 0);
        ggml_tensor* q_pass = ggml_view_3d(ctx, q, hd, nh - half_nh, T_latent, nb1, nb2, half_nh * nb1);
        q_rot = ggml_cont(ctx, q_rot);
        q_rot = ggml_rope_ext(ctx, q_rot, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_concat(ctx, q_rot, ggml_cont(ctx, q_pass), 1); // concat on head dim

        // K_self: same half-RoPE
        nb1 = k_self->nb[1];
        nb2 = k_self->nb[2];
        ggml_tensor* ks_rot = ggml_view_3d(ctx, k_self, hd, half_nh, T_latent, nb1, nb2, 0);
        ggml_tensor* ks_pass = ggml_view_3d(ctx, k_self, hd, nh - half_nh, T_latent, nb1, nb2, half_nh * nb1);
        ks_rot = ggml_cont(ctx, ks_rot);
        ks_rot = ggml_rope_ext(ctx, ks_rot, pos_ids, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k_self = ggml_concat(ctx, ks_rot, ggml_cont(ctx, ks_pass), 1);
    }

    // Permute all to (hd, T, nh) for flash_attn_ext
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k_self = ggml_cont(ctx, ggml_permute(ctx, k_self, 0, 2, 1, 3));
    v_self = ggml_cont(ctx, ggml_permute(ctx, v_self, 0, 2, 1, 3));

    // Text K/V: no RoPE (context KV), just norm + permute
    k_text = ggml_reshape_3d(ctx, k_text, hd, nh, T_text);
    k_text = ggml_rms_norm(ctx, k_text, hp.norm_eps);
    k_text = ggml_mul(ctx, k_text, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));
    k_text = ggml_cont(ctx, ggml_permute(ctx, k_text, 0, 2, 1, 3));
    v_text = ggml_reshape_3d(ctx, v_text, hd, nh, T_text);
    v_text = ggml_cont(ctx, ggml_permute(ctx, v_text, 0, 2, 1, 3));

    // Concatenate K and V along sequence dim (dim=1): (hd, T_total, nh)
    ggml_tensor* k_cat = ggml_concat(ctx, k_self, k_text, 1);
    ggml_tensor* v_cat = ggml_concat(ctx, v_self, v_text, 1);
    if (k_spk && v_spk) {
        k_spk = ggml_reshape_3d(ctx, k_spk, hd, nh, T_ref);
        k_spk = ggml_rms_norm(ctx, k_spk, hp.norm_eps);
        k_spk = ggml_mul(ctx, k_spk, ggml_cast(ctx, blk.k_norm, GGML_TYPE_F32));
        k_spk = ggml_cont(ctx, ggml_permute(ctx, k_spk, 0, 2, 1, 3));
        v_spk = ggml_reshape_3d(ctx, v_spk, hd, nh, T_ref);
        v_spk = ggml_cont(ctx, ggml_permute(ctx, v_spk, 0, 2, 1, 3));
        k_cat = ggml_concat(ctx, k_cat, k_spk, 1);
        v_cat = ggml_concat(ctx, v_cat, v_spk, 1);
    }

    // Flash attention: Q(hd, T_q, nh), K(hd, T_kv, nh), V(hd, T_kv, nh)
    ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k_cat, v_cat, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    attn_out = ggml_cast(ctx, attn_out, GGML_TYPE_F32);
    attn_out = ggml_reshape_2d(ctx, attn_out, D, T_latent);

    // Gated attention
    ggml_tensor* g = mul_mat_f32(ctx, blk.gate, h_attn);
    g = ggml_sigmoid(ctx, g);
    attn_out = ggml_mul(ctx, attn_out, g);
    attn_out = mul_mat_f32(ctx, blk.wo, attn_out);

    // gate_attn * attn_out (gate is 1D, attn_out is 2D — larger tensor first for ggml_mul)
    attn_out = ggml_mul(ctx, attn_out, gate_attn);
    x = ggml_add(ctx, x, attn_out);

    // ── MLP path ──
    ggml_tensor* h_mlp = nullptr;
    ggml_tensor* gate_mlp = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_mlp, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_mlp, &gate_mlp);

    // SwiGLU may produce F16 from quantized weights — need F32 for gate mul
    ggml_tensor* gate_proj = mul_mat_f32(ctx, blk.mlp_w1, h_mlp);
    ggml_tensor* up_proj = mul_mat_f32(ctx, blk.mlp_w3, h_mlp);
    ggml_tensor* mlp_out = ggml_mul(ctx, ggml_silu(ctx, gate_proj), up_proj);
    mlp_out = mul_mat_f32(ctx, blk.mlp_w2, mlp_out);
    mlp_out = ggml_mul(ctx, mlp_out, gate_mlp);
    x = ggml_add(ctx, x, mlp_out);

    return x;
}

// ── Tokenizer ───────────────────────────────────────────────────────

static std::vector<int32_t> tokenize_text(const irodori_tts_context* ctx, const char* text) {
    // Override token IDs from env var for parity testing:
    //   CRISPASR_IRODORI_TOKEN_IDS="1,19144,52839,302,275"
    const char* override_ids = std::getenv("CRISPASR_IRODORI_TOKEN_IDS");
    if (override_ids && *override_ids) {
        std::vector<int32_t> tokens;
        std::string s(override_ids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t end = s.find(',', pos);
            if (end == std::string::npos)
                end = s.size();
            tokens.push_back(std::atoi(s.substr(pos, end - pos).c_str()));
            pos = end + 1;
        }
        IRODORI_DBG("[irodori] using override token IDs (%d tokens)\n", (int)tokens.size());
        return tokens;
    }

    std::vector<int32_t> tokens;

    // Prepend BOS if configured
    if (ctx->bos_token_id >= 0) {
        tokens.push_back(ctx->bos_token_id);
    }

    if (!ctx->token_to_id.empty() && !ctx->token_scores.empty()) {
        // Use SentencePiece Viterbi tokenizer from core/sentencepiece.h
        core_spm::Config cfg;
        cfg.unk_id = 0;
        cfg.utf8_aligned = true;
        cfg.merge_consecutive_unk = true;
        auto sp_tokens = core_spm::tokenize(text, ctx->token_to_id, ctx->token_scores, cfg, true);
        tokens.insert(tokens.end(), sp_tokens.begin(), sp_tokens.end());
    } else {
        // Byte-level fallback (works but suboptimal)
        IRODORI_DBG("[irodori] WARNING: no SentencePiece vocab loaded, using byte fallback\n");
        const uint8_t* p = (const uint8_t*)text;
        while (*p) {
            tokens.push_back((int32_t)*p);
            p++;
        }
    }

    return tokens;
}

// ── Public API ──────────────────────────────────────────────────────

struct irodori_tts_params irodori_tts_default_params(void) {
    irodori_tts_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.seed = 0;
    p.ode_steps = 40;
    p.cfg_scale_text = 3.0f;
    p.cfg_scale_speaker = 5.0f;
    p.speed = 1.0f;
    p.duration_scale = 1.0f;
    p.max_seconds = 30.0f;
    return p;
}

struct irodori_tts_context* irodori_tts_init_from_file(const char* path_model, struct irodori_tts_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new irodori_tts_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->seed = params.seed;
    ctx->ode_steps = params.ode_steps > 0 ? params.ode_steps : 40;
    ctx->cfg_scale_text = params.cfg_scale_text;
    ctx->cfg_scale_speaker = params.cfg_scale_speaker;
    ctx->speed = params.speed > 0 ? params.speed : 1.0f;
    ctx->duration_scale = params.duration_scale > 0 ? params.duration_scale : 1.0f;
    ctx->max_seconds = params.max_seconds > 0 ? params.max_seconds : 30.0f;

    // Initialize CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        std::fprintf(stderr, "[irodori] failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load GGUF
    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] loading model: %s\n", path_model);
    }

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        std::fprintf(stderr, "[irodori] failed to open GGUF metadata: %s\n", path_model);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Read hyperparameters
    auto& hp = ctx->hparams;
    hp.latent_dim = core_gguf::kv_i32(meta, "irodori.latent_dim", hp.latent_dim);
    hp.latent_patch_size = core_gguf::kv_i32(meta, "irodori.latent_patch_size", hp.latent_patch_size);
    hp.model_dim = core_gguf::kv_i32(meta, "irodori.model_dim", hp.model_dim);
    hp.num_layers = core_gguf::kv_i32(meta, "irodori.num_layers", hp.num_layers);
    hp.num_heads = core_gguf::kv_i32(meta, "irodori.num_heads", hp.num_heads);
    hp.mlp_ratio = core_gguf::kv_f32(meta, "irodori.mlp_ratio", hp.mlp_ratio);
    hp.text_dim = core_gguf::kv_i32(meta, "irodori.text_dim", hp.text_dim);
    hp.text_layers = core_gguf::kv_i32(meta, "irodori.text_layers", hp.text_layers);
    hp.text_heads = core_gguf::kv_i32(meta, "irodori.text_heads", hp.text_heads);
    hp.text_mlp_ratio = core_gguf::kv_f32(meta, "irodori.text_mlp_ratio", hp.text_mlp_ratio);
    hp.text_vocab_size = core_gguf::kv_i32(meta, "irodori.text_vocab_size", hp.text_vocab_size);
    hp.speaker_dim = core_gguf::kv_i32(meta, "irodori.speaker_dim", hp.speaker_dim);
    hp.speaker_layers = core_gguf::kv_i32(meta, "irodori.speaker_layers", hp.speaker_layers);
    hp.speaker_heads = core_gguf::kv_i32(meta, "irodori.speaker_heads", hp.speaker_heads);
    hp.speaker_mlp_ratio = core_gguf::kv_f32(meta, "irodori.speaker_mlp_ratio", hp.speaker_mlp_ratio);
    hp.speaker_patch_size = core_gguf::kv_i32(meta, "irodori.speaker_patch_size", hp.speaker_patch_size);
    hp.timestep_embed_dim = core_gguf::kv_i32(meta, "irodori.timestep_embed_dim", hp.timestep_embed_dim);
    hp.adaln_rank = core_gguf::kv_i32(meta, "irodori.adaln_rank", hp.adaln_rank);
    hp.norm_eps = core_gguf::kv_f32(meta, "irodori.norm_eps", hp.norm_eps);
    hp.use_duration_predictor = core_gguf::kv_i32(meta, "irodori.use_duration_predictor", hp.use_duration_predictor);
    hp.duration_aux_dim = core_gguf::kv_i32(meta, "irodori.duration_aux_dim", hp.duration_aux_dim);
    hp.duration_hidden_dim = core_gguf::kv_i32(meta, "irodori.duration_hidden_dim", hp.duration_hidden_dim);
    hp.duration_layers = core_gguf::kv_i32(meta, "irodori.duration_layers", hp.duration_layers);
    hp.ode_steps = core_gguf::kv_i32(meta, "irodori.ode_steps", hp.ode_steps);
    hp.cfg_scale_text = core_gguf::kv_f32(meta, "irodori.cfg_scale_text", hp.cfg_scale_text);
    hp.cfg_scale_speaker = core_gguf::kv_f32(meta, "irodori.cfg_scale_speaker", hp.cfg_scale_speaker);
    hp.sample_rate = core_gguf::kv_i32(meta, "irodori.sample_rate", hp.sample_rate);
    hp.codec_hop_length = core_gguf::kv_i32(meta, "irodori.codec_hop_length", hp.codec_hop_length);

    // Load tokenizer from GGUF metadata
    {
        auto vocab_strs = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
        ctx->bos_token_id = core_gguf::kv_i32(meta, "tokenizer.ggml.bos_token_id", -1);
        ctx->pad_token_id = core_gguf::kv_i32(meta, "tokenizer.ggml.pad_token_id", -1);

        if (!vocab_strs.empty()) {
            ctx->vocab.resize(vocab_strs.size());
            for (size_t i = 0; i < vocab_strs.size(); i++) {
                ctx->vocab[i] = vocab_strs[i];
                ctx->token_to_id[vocab_strs[i]] = (int32_t)i;
            }
            // Load scores
            auto scores = core_gguf::kv_f32_array(meta, "tokenizer.ggml.scores");
            if (!scores.empty()) {
                ctx->token_scores = scores;
            }
            if (ctx->verbosity >= 1) {
                std::fprintf(stderr, "[irodori] tokenizer: %zu tokens, %zu scores, bos=%d, pad=%d\n", ctx->vocab.size(),
                             ctx->token_scores.size(), ctx->bos_token_id, ctx->pad_token_id);
            }
        } else {
            if (ctx->verbosity >= 1) {
                std::fprintf(stderr, "[irodori] WARNING: no tokenizer in GGUF, using byte fallback\n");
            }
        }
    }

    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr,
                     "[irodori] model: dim=%d, layers=%d, heads=%d, text_dim=%d, "
                     "text_layers=%d, latent_dim=%d\n",
                     hp.model_dim, hp.num_layers, hp.num_heads, hp.text_dim, hp.text_layers, hp.latent_dim);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "irodori_tts", wl)) {
        std::fprintf(stderr, "[irodori] failed to load weights\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Load weight tensors
    auto& ts = wl.tensors;
    auto get = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ts, name.c_str(), "irodori_tts");
    };
    auto try_get = [&](const std::string& name) -> ggml_tensor* { return core_gguf::try_get(ts, name.c_str()); };

    if (!load_weights_from_map(ctx, get, try_get)) {
        std::fprintf(stderr, "[irodori] failed to resolve weight tensors\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Keep weight context + buffer alive (tensors reference both)
    ctx->w_ctx = wl.ctx;
    ctx->buf_weights = wl.buf;

    // Set BOS token: prefer GGUF tokenizer value, fall back to sarashina2.2 default (1)
    if (ctx->bos_token_id < 0) {
        ctx->bos_token_id = 1; // sarashina2.2 BOS token ID
    }

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] model loaded successfully\n");
    }

    return ctx;
}

void irodori_tts_free(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_weights)
        ggml_backend_buffer_free(ctx->buf_weights);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int irodori_tts_set_reference(struct irodori_tts_context* ctx, const float* ref_pcm, int n_samples, int sample_rate) {
    if (!ctx || !ref_pcm || n_samples <= 0 || sample_rate <= 0)
        return -1;

    // TODO: Encode reference audio through DAC-VAE encoder.
    // For now, this requires pre-encoded latents via irodori_tts_set_reference_latent().
    // The DAC-VAE encoder needs to be ported or called externally.
    std::fprintf(stderr, "[irodori] WARNING: raw PCM reference not yet supported. "
                         "Use irodori_tts_set_reference_latent() with pre-encoded DAC-VAE latents.\n");
    return -1;
}

int irodori_tts_set_reference_latent(struct irodori_tts_context* ctx, const float* latent, int n_frames) {
    if (!ctx || !latent || n_frames <= 0)
        return -1;

    int D = ctx->hparams.latent_dim * ctx->hparams.latent_patch_size;
    ctx->ref_latent.resize(n_frames * D);
    std::memcpy(ctx->ref_latent.data(), latent, n_frames * D * sizeof(float));
    ctx->ref_latent_frames = n_frames;

    IRODORI_DBG("[irodori] set reference latent: %d frames x %d dims\n", n_frames, D);
    return 0;
}

void irodori_tts_clear_reference(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->ref_latent.clear();
    ctx->ref_latent_frames = 0;
}

// ── CPU-side forward pass helpers ────────────────────────────────────

// Run text encoder: token_ids → text_state (T_text × text_dim).
// Builds a ggml graph, evaluates on backend, returns CPU-side result.
static std::vector<float> run_text_encoder(irodori_tts_context* ctx, const int32_t* token_ids, int T_text) {
    const auto& hp = ctx->hparams;
    const int D = hp.text_dim;

    // Tensor count: embedding + per-block (Q,K,V,O,gate,norms,mlp,permute,cast) + overhead
    const int n_tensors = 512 + hp.text_layers * 80;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    // Input tensors
    ggml_tensor* ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_text);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);

    ggml_tensor* mask_f = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1, T_text);
    ggml_set_name(mask_f, "text_mask_f");
    ggml_set_input(mask_f);

    ggml_tensor* pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_text);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    // Build graph
    ggml_tensor* out = build_text_encoder_graph(g, ctx, ids, mask_f, pos, T_text);
    ggml_set_name(out, "text_state");
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, out);

    // Allocate and set inputs
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(ids, token_ids, 0, T_text * sizeof(int32_t));
    std::vector<float> mask_data(T_text, 1.0f);
    ggml_backend_tensor_set(mask_f, mask_data.data(), 0, T_text * sizeof(float));
    std::vector<int32_t> pos_data(T_text);
    for (int i = 0; i < T_text; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(pos, pos_data.data(), 0, T_text * sizeof(int32_t));

    // Compute
    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output
    std::vector<float> result(T_text * D);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return result;
}

// Run one DiT forward pass: x_t + timestep + text_state + spk_state → velocity prediction.
// Returns velocity as CPU-side array (T_latent × patched_latent_dim).
static std::vector<float> run_dit_forward(irodori_tts_context* ctx, const float* x_t_data, int T_latent,
                                          const float* cond_embed_data, // (model_dim*3,)
                                          const float* text_state_data, int T_text, const float* spk_state_data,
                                          int T_ref) {
    const auto& hp = ctx->hparams;
    const auto& w = ctx->weights;
    const int D = hp.model_dim;
    const int latent_d = hp.patched_latent_dim();

    const int n_tensors = 2048 + hp.num_layers * 200; // extra for half-RoPE views/concats
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    // Input tensors
    ggml_tensor* x_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, latent_d, T_latent);
    ggml_set_name(x_in, "x_t");
    ggml_set_input(x_in);

    ggml_tensor* cond_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, D * 3);
    ggml_set_name(cond_in, "cond_embed");
    ggml_set_input(cond_in);

    ggml_tensor* text_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.text_dim, T_text);
    ggml_set_name(text_in, "text_state");
    ggml_set_input(text_in);

    ggml_tensor* pos_latent = ggml_new_tensor_1d(g, GGML_TYPE_I32, T_latent);
    ggml_set_name(pos_latent, "pos_latent");
    ggml_set_input(pos_latent);

    ggml_tensor* spk_in = nullptr;
    if (spk_state_data && T_ref > 0) {
        spk_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.speaker_dim, T_ref);
        ggml_set_name(spk_in, "spk_state");
        ggml_set_input(spk_in);
    }

    // Build graph: in_proj → DiT blocks → out_norm → out_proj
    IRODORI_DBG("[irodori]     in_proj: (%d,%d) x (%d,%d)\n", (int)w.in_proj_w->ne[0], (int)w.in_proj_w->ne[1],
                latent_d, T_latent);
    ggml_tensor* x = mul_mat_f32(g, w.in_proj_w, x_in);
    x = ggml_add(g, x, w.in_proj_b);
    IRODORI_DBG("[irodori]     after in_proj: (%d,%d)\n", (int)x->ne[0], (int)x->ne[1]);

    // Build just the first layer for debugging, then add more once stable
    int n_build_layers = hp.num_layers;
    const char* env_layers = std::getenv("CRISPASR_IRODORI_LAYERS");
    if (env_layers)
        n_build_layers = std::min(std::atoi(env_layers), hp.num_layers);
    for (int i = 0; i < n_build_layers; i++) {
        IRODORI_DBG("[irodori]     building DiT block %d...\n", i);
        x = build_dit_block_graph(g, ctx, w.dit_blocks[i], x, cond_in, text_in, T_text, spk_in, T_ref, T_latent,
                                  pos_latent);
        IRODORI_DBG("[irodori]     DiT block %d built OK\n", i);
    }

    x = rms_norm(g, x, w.out_norm, hp.norm_eps);
    x = mul_mat_f32(g, w.out_proj_w, x);
    x = ggml_add(g, x, w.out_proj_b);
    ggml_set_name(x, "v_pred");
    ggml_set_output(x);

    IRODORI_DBG("[irodori]     DiT graph built, creating cgraph (%d tensors)...\n", n_tensors);
    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, x);
    IRODORI_DBG("[irodori]     cgraph has %d nodes, allocating...\n", ggml_graph_n_nodes(gf));

    // Allocate and set inputs
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    bool res_ok = ggml_gallocr_reserve(galloc, gf);
    IRODORI_DBG("[irodori]     gallocr reserve: %s\n", res_ok ? "OK" : "FAILED");
    bool alloc_ok = ggml_gallocr_alloc_graph(galloc, gf);
    IRODORI_DBG("[irodori]     gallocr alloc: %s\n", alloc_ok ? "OK" : "FAILED");
    if (!res_ok || !alloc_ok) {
        std::fprintf(stderr, "[irodori] ERROR: failed to allocate DiT graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    // Set inputs (only if allocated — unused inputs have null buffers)
    if (x_in->buffer)
        ggml_backend_tensor_set(x_in, x_t_data, 0, T_latent * latent_d * sizeof(float));
    if (cond_in->buffer)
        ggml_backend_tensor_set(cond_in, cond_embed_data, 0, D * 3 * sizeof(float));
    if (text_in->buffer)
        ggml_backend_tensor_set(text_in, text_state_data, 0, T_text * hp.text_dim * sizeof(float));
    if (spk_in && spk_in->buffer) {
        ggml_backend_tensor_set(spk_in, spk_state_data, 0, T_ref * hp.speaker_dim * sizeof(float));
    }
    if (pos_latent->buffer) {
        std::vector<int32_t> pos_data(T_latent);
        for (int i = 0; i < T_latent; i++)
            pos_data[i] = i;
        ggml_backend_tensor_set(pos_latent, pos_data.data(), 0, T_latent * sizeof(int32_t));
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    // Read velocity prediction
    std::vector<float> v_pred(T_latent * latent_d);
    ggml_backend_tensor_get(x, v_pred.data(), 0, v_pred.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return v_pred;
}

// Compute timestep conditioning: sinusoidal(t) → MLP → cond_embed (model_dim*3,)
static std::vector<float> run_timestep_cond(irodori_tts_context* ctx, float t) {
    const auto& hp = ctx->hparams;
    const auto& w = ctx->weights;
    const int D = hp.model_dim;

    // 1. Sinusoidal timestep embedding (CPU side)
    std::vector<float> t_emb;
    compute_timestep_embedding(t, hp.timestep_embed_dim, t_emb);

    // 2. MLP: Linear → SiLU → Linear → SiLU → Linear
    const int n_tensors = 32;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_tensors, false);
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(ip);

    ggml_tensor* emb_in = ggml_new_tensor_1d(g, GGML_TYPE_F32, hp.timestep_embed_dim);
    ggml_set_name(emb_in, "t_emb");
    ggml_set_input(emb_in);

    // cond_module: Linear(512,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048, 2048*3)
    ggml_tensor* h = mul_mat_f32(g, w.cond_0_w, emb_in);
    h = ggml_silu(g, h);
    h = mul_mat_f32(g, w.cond_2_w, h);
    h = ggml_silu(g, h);
    h = mul_mat_f32(g, w.cond_4_w, h);
    ggml_set_name(h, "cond_embed");
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_tensors, false);
    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(g);
        return {};
    }

    ggml_backend_tensor_set(emb_in, t_emb.data(), 0, t_emb.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    std::vector<float> cond(D * 3);
    ggml_backend_tensor_get(h, cond.data(), 0, cond.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(g);
    return cond;
}

// ── Main synthesize ─────────────────────────────────────────────────

int irodori_tts_synthesize(struct irodori_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    const auto& hp = ctx->hparams;
    *sample_rate_out = hp.sample_rate;

    // Tokenize
    std::vector<int32_t> token_ids = tokenize_text(ctx, text);
    int T_text = (int)token_ids.size();
    if (T_text == 0)
        return 0;

    IRODORI_DBG("[irodori] text tokens: %d\n", T_text);

    // Determine output length (heuristic: ~6.3 frames per token, or override)
    const char* t_lat_override = std::getenv("CRISPASR_IRODORI_T_LATENT");
    int latent_steps;
    if (t_lat_override && std::atoi(t_lat_override) > 0) {
        latent_steps = std::atoi(t_lat_override);
    } else {
        float frames_per_token = 6.3f;
        latent_steps = (int)(T_text * frames_per_token * ctx->duration_scale / ctx->speed);
    }
    int max_steps = (int)(ctx->max_seconds * hp.sample_rate / hp.codec_hop_length);
    latent_steps = std::min(latent_steps, max_steps);
    latent_steps = std::max(latent_steps, 1);
    int patched_steps = latent_steps / std::max(hp.latent_patch_size, 1);

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] synthesize: %d tokens → %d latent frames (%.2f sec)\n", T_text, latent_steps,
                     (float)latent_steps * hp.codec_hop_length / hp.sample_rate);
    }

    // ── Step 1: Encode text ──
    IRODORI_DBG("[irodori] encoding text...\n");
    auto text_state = run_text_encoder(ctx, token_ids.data(), T_text);
    if (text_state.empty()) {
        std::fprintf(stderr, "[irodori] text encoder failed\n");
        return 0;
    }
    IRODORI_DBG("[irodori] text_state: %d x %d (first 4: %.4f %.4f %.4f %.4f)\n", T_text, hp.text_dim, text_state[0],
                text_state[1], text_state[2], text_state[3]);

    // Dump text_state for parity comparison
    {
        const char* dump_ts = std::getenv("CRISPASR_IRODORI_DUMP_TEXT_STATE");
        if (dump_ts && *dump_ts && *dump_ts != '0') {
            std::string path = dump_ts;
            if (path == "1")
                path = "irodori_text_state.bin";
            FILE* f = std::fopen(path.c_str(), "wb");
            if (f) {
                std::fwrite(text_state.data(), sizeof(float), text_state.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] text_state dumped to '%s'\n", path.c_str());
            }
        }
    }

    // ── Step 2: Encode speaker reference (or zeros for unconditional) ──
    // The DiT's JointAttention REQUIRES speaker context tensors even when
    // unconditional (no_ref=True). Pass zeros with False mask.
    int T_ref = 1; // minimum 1 frame for the zeroed speaker context
    std::vector<float> spk_state(T_ref * hp.speaker_dim, 0.0f);
    // TODO: when ref_latent is set, run speaker encoder instead of zeros.

    // ── Step 3: Euler RF ODE solver ──
    IRODORI_DBG("[irodori] ODE solver: %d steps\n", ctx->ode_steps);

    const int latent_d = hp.patched_latent_dim();
    const int n_ode = ctx->ode_steps;

    // Timestep schedule: t goes from 0.999 → 0 (noise → clean)
    std::vector<float> t_schedule(n_ode + 1);
    for (int i = 0; i <= n_ode; i++) {
        float u = (float)i / (float)n_ode;
        t_schedule[i] = (1.0f - u) * 0.999f; // decreasing: 0.999 → 0
    }

    // Initial noise x_t ~ N(0, 1) or loaded from reference file
    std::vector<float> x_t(patched_steps * latent_d);
    {
        const char* ref_noise = std::getenv("CRISPASR_IRODORI_REF_NOISE");
        if (ref_noise && *ref_noise) {
            FILE* f = std::fopen(ref_noise, "rb");
            if (f) {
                size_t n = std::fread(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] loaded reference noise from '%s' (%zu floats)\n", ref_noise, n);
            }
        } else {
            unsigned int seed_val = ctx->seed ? (unsigned int)ctx->seed : std::random_device{}();
            std::mt19937 rng(seed_val);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto& v : x_t)
                v = dist(rng);
        }
    }

    // ODE integration loop
    for (int step = 0; step < n_ode; step++) {
        float t_val = t_schedule[step];
        float t_next = t_schedule[step + 1];
        float dt = t_next - t_val; // negative (moving toward t=0)

        // Compute timestep conditioning
        IRODORI_DBG("[irodori]   step %d: computing timestep cond (t=%.4f)...\n", step, t_val);
        auto cond_embed = run_timestep_cond(ctx, t_val);
        if (cond_embed.empty()) {
            std::fprintf(stderr, "[irodori] timestep cond failed at step %d\n", step);
            return 0;
        }

        IRODORI_DBG("[irodori]   step %d: running DiT forward...\n", step);
        // DiT forward: conditioned pass
        auto v_cond = run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_state.data(), T_text,
                                      spk_state.data(), T_ref);
        if (v_cond.empty()) {
            std::fprintf(stderr, "[irodori] DiT forward failed at step %d\n", step);
            return 0;
        }

        // Dump step-0 v_pred for parity comparison
        if (step == 0) {
            const char* dump_v0 = std::getenv("CRISPASR_IRODORI_DUMP_V_PRED0");
            if (dump_v0 && *dump_v0) {
                FILE* f = std::fopen(dump_v0, "wb");
                if (f) {
                    std::fwrite(v_cond.data(), sizeof(float), v_cond.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr, "[irodori] v_pred step0 dumped to '%s'\n", dump_v0);
                }
            }
        }

        // CFG: independent text guidance
        // v = v_cond + cfg_text * (v_cond - v_uncond_text)
        float cfg_text = ctx->cfg_scale_text;
        float cfg_min_t = 0.5f;
        float cfg_max_t = 1.0f;
        bool use_cfg = (cfg_text > 0.0f) && (t_val >= cfg_min_t) && (t_val <= cfg_max_t);

        if (use_cfg) {
            // Unconditioned pass: zero text state
            std::vector<float> text_uncond(T_text * hp.text_dim, 0.0f);
            auto v_uncond = run_dit_forward(ctx, x_t.data(), patched_steps, cond_embed.data(), text_uncond.data(),
                                            T_text, spk_state.empty() ? nullptr : spk_state.data(), T_ref);
            if (!v_uncond.empty()) {
                for (size_t i = 0; i < x_t.size(); i++) {
                    float v = v_cond[i] + cfg_text * (v_cond[i] - v_uncond[i]);
                    x_t[i] += v * dt;
                }
            } else {
                // Fallback: no CFG
                for (size_t i = 0; i < x_t.size(); i++)
                    x_t[i] += v_cond[i] * dt;
            }
        } else {
            // No CFG for this timestep
            for (size_t i = 0; i < x_t.size(); i++) {
                x_t[i] += v_cond[i] * dt;
            }
        }

        if (ctx->verbosity >= 2 || (ctx->verbosity >= 1 && (step == 0 || step == n_ode - 1))) {
            std::fprintf(stderr, "  ODE step %d/%d  t=%.4f→%.4f\n", step + 1, n_ode, t_val, t_next);
        }
    }

    IRODORI_DBG("[irodori] ODE complete. latent shape: %d x %d\n", patched_steps, latent_d);

    // Dump latent for external DACVAE decode (env-gated)
    {
        const char* dump_env = std::getenv("CRISPASR_IRODORI_DUMP_LATENT");
        if (dump_env && *dump_env && *dump_env != '0') {
            std::string lat_path = dump_env;
            if (lat_path == "1")
                lat_path = "irodori_latent.bin";
            FILE* f = std::fopen(lat_path.c_str(), "wb");
            if (f) {
                std::fwrite(x_t.data(), sizeof(float), x_t.size(), f);
                std::fclose(f);
                std::fprintf(stderr, "[irodori] latent dumped to '%s' (%d frames × %d dims)\n", lat_path.c_str(),
                             patched_steps, latent_d);
            }
        }
    }

    // ── Step 4: Output ──
    // The latent (patched_steps × latent_d) needs DAC-VAE decoding to produce PCM.
    // For now, output silence and log that DAC-VAE decode is needed.
    // TODO: integrate DACVAE decoder from dacvae-ja-32dim-f16.gguf
    int n_pcm = latent_steps * hp.codec_hop_length;
    float* out = (float*)std::malloc(n_pcm * sizeof(float));
    if (!out)
        return 0;
    std::memset(out, 0, n_pcm * sizeof(float));

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr,
                     "[irodori] output: %d samples (%.2f sec @ %d Hz) "
                     "[DAC-VAE decode pending — latent produced]\n",
                     n_pcm, (float)n_pcm / hp.sample_rate, hp.sample_rate);
    }

    *pcm_out = out;
    return n_pcm;
}

void irodori_tts_set_seed(struct irodori_tts_context* ctx, int seed) {
    if (ctx)
        ctx->seed = seed;
}

void irodori_tts_set_ode_steps(struct irodori_tts_context* ctx, int steps) {
    if (ctx && steps > 0)
        ctx->ode_steps = steps;
}

void irodori_tts_set_cfg_scale_text(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_text = scale;
}

void irodori_tts_set_cfg_scale_speaker(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_speaker = scale;
}

void irodori_tts_set_speed(struct irodori_tts_context* ctx, float speed) {
    if (ctx && speed > 0)
        ctx->speed = speed;
}

int irodori_tts_sample_rate(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.sample_rate : 48000;
}

int irodori_tts_vocab_size(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.text_vocab_size : 0;
}
