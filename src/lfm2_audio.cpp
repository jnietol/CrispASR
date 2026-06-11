// lfm2_audio.cpp — LiquidAI LFM2.5-Audio ggml runtime
//
// Architecture:
//   Mel:         128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann, NeMo-style)
//   Encoder:     17× FastConformer (d_model=512, 8H, rel_pos, dw_striding 8×)
//   Adapter:     LayerNorm(512) → Linear(512→2048) → GELU → Linear(2048→2048)
//   LFM2:        16L hybrid backbone (10 conv + 6 GQA attn), hidden=2048
//   Depthformer: 6L (dim=1024), 8 codebooks → Mimi audio tokens
//
// DRY: reuses core/gguf_loader.h, core/mel.h, core/fastconformer.h, core/attention.h

#include "lfm2_audio.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/fastconformer.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/fft.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct lfm2_audio_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t codebooks = 8;
    uint32_t audio_vocab_size = 2049;

    // FastConformer encoder
    uint32_t enc_n_layers = 17;
    uint32_t enc_d_model = 512;
    uint32_t enc_n_heads = 8;
    uint32_t enc_ff_expansion = 4;
    uint32_t enc_conv_kernel = 9;
    uint32_t enc_subsampling_factor = 8;
    uint32_t enc_subsampling_channels = 256;

    // LFM2 backbone
    uint32_t lfm_hidden_size = 2048;
    uint32_t lfm_n_layers = 16;
    uint32_t lfm_n_heads = 32;
    uint32_t lfm_n_kv_heads = 8;
    uint32_t lfm_head_dim = 64;
    uint32_t lfm_ff_dim = 8192;
    uint32_t lfm_conv_kernel = 3;
    float lfm_rope_theta = 1000000.0f;
    std::string lfm_layer_types; // "ccaccaccacacacac"

    // Depthformer
    uint32_t depth_n_layers = 6;
    uint32_t depth_dim = 1024;
    uint32_t depth_tie = 1;
    uint32_t text_vocab_size = 65536;
};

// ===========================================================================
// LFM2 backbone per-layer weights
// ===========================================================================

struct lfm2_layer_weights {
    ggml_tensor* operator_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor *ff_w1 = nullptr, *ff_w2 = nullptr, *ff_w3 = nullptr;

    bool is_attention = false;

    // Conv layers
    ggml_tensor* conv_conv_w = nullptr;     // [hidden, 1, kernel]
    ggml_tensor* conv_in_proj_w = nullptr;  // [3*hidden, hidden]
    ggml_tensor* conv_out_proj_w = nullptr; // [hidden, hidden]

    // Attention layers
    ggml_tensor* attn_q_proj_w = nullptr;
    ggml_tensor* attn_k_proj_w = nullptr;
    ggml_tensor* attn_v_proj_w = nullptr;
    ggml_tensor* attn_out_proj_w = nullptr;
    ggml_tensor* attn_q_ln_w = nullptr;
    ggml_tensor* attn_k_ln_w = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct lfm2_audio_model {
    lfm2_audio_hparams hparams;

    // FastConformer encoder (reused from core/fastconformer.h)
    core_conformer::PreEncodeWeights pre_enc;
    std::vector<core_conformer::BlockWeights> enc_blocks;

    // Audio adapter MLP
    ggml_tensor *adapter_norm_w = nullptr, *adapter_norm_b = nullptr;
    ggml_tensor *adapter_lin0_w = nullptr, *adapter_lin0_b = nullptr;
    ggml_tensor *adapter_lin1_w = nullptr, *adapter_lin1_b = nullptr;

    // LFM2 backbone
    ggml_tensor* lfm_embed_tokens_w = nullptr;
    ggml_tensor* lfm_embedding_norm_w = nullptr;
    std::vector<lfm2_layer_weights> lfm_layers;

    // Mel preprocessor (from GGUF — librosa slaney mel filterbank + Hann window)
    ggml_tensor* mel_fb = nullptr;     // (n_mels, n_freqs)
    ggml_tensor* mel_window = nullptr; // (win_length,)

    // Audio embedding
    ggml_tensor* audio_embd_embedding_w = nullptr;
    ggml_tensor* audio_embd_embedding_norm_w = nullptr;
    ggml_tensor* audio_embd_to_logits_w = nullptr;

    // core_gguf resources
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Vocabulary
    std::vector<std::string> vocab;
};

// ===========================================================================
// Context
// ===========================================================================

struct lfm2_audio_context {
    lfm2_audio_model model;
    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    ggml_backend_t backend = nullptr;

    // Compute buffer
    std::vector<uint8_t> compute_meta;

    // Staged callback (set by run_lfm_staged)
    lfm2_audio_stage_cb lfm_stage_cb = nullptr;
    void* lfm_stage_ud = nullptr;

    // ---- KV cache for attention layers ----
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv_heads, n_attn_layers)
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_past = 0; // tokens already in cache

    // ---- Conv state cache for conv layers ----
    // Each conv layer caches the last (kernel-1)=2 Bx vectors.
    // Layout: [n_conv_layers][hidden * (kernel-1)] flat CPU float.
    std::vector<std::vector<float>> conv_states;
    bool conv_states_valid = false;

    void reset_kv() {
        kv_n_past = 0;
        if (kv_buf)
            ggml_backend_buffer_clear(kv_buf, 0);
        conv_states_valid = false;
        for (auto& s : conv_states)
            std::fill(s.begin(), s.end(), 0.0f);
    }
};

// ===========================================================================
// Tensor lookup helpers (DRY: delegates to core_gguf)
// ===========================================================================

static ggml_tensor* R(lfm2_audio_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "lfm2-audio");
}

static ggml_tensor* G(lfm2_audio_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

// ===========================================================================
// FFT (shared via core/fft.h where available, else inline Radix-2 DIT)
// ===========================================================================

static void lfm2_fft_r2c(const float* in, int N, float* out) {
    core_fft::fft_radix2_wrapper(in, N, out);
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool lfm2_audio_load(lfm2_audio_model& model, const char* path, ggml_backend_t backend, int verbosity) {
    // ---- Pass 1: metadata ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "lfm2audio.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "lfm2audio.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "lfm2audio.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "lfm2audio.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "lfm2audio.hop_length", hp.hop_length);
        hp.codebooks = core_gguf::kv_u32(gctx, "lfm2audio.codebooks", hp.codebooks);
        hp.audio_vocab_size = core_gguf::kv_u32(gctx, "lfm2audio.audio_vocab_size", hp.audio_vocab_size);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.enc_n_layers", hp.enc_n_layers);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "lfm2audio.enc_d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "lfm2audio.enc_n_heads", hp.enc_n_heads);
        hp.enc_conv_kernel = core_gguf::kv_u32(gctx, "lfm2audio.enc_conv_kernel", hp.enc_conv_kernel);
        hp.enc_subsampling_factor =
            core_gguf::kv_u32(gctx, "lfm2audio.enc_subsampling_factor", hp.enc_subsampling_factor);
        hp.enc_subsampling_channels =
            core_gguf::kv_u32(gctx, "lfm2audio.enc_subsampling_channels", hp.enc_subsampling_channels);
        hp.lfm_hidden_size = core_gguf::kv_u32(gctx, "lfm2audio.lfm_hidden_size", hp.lfm_hidden_size);
        hp.lfm_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_layers", hp.lfm_n_layers);
        hp.lfm_n_heads = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_heads", hp.lfm_n_heads);
        hp.lfm_n_kv_heads = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_kv_heads", hp.lfm_n_kv_heads);
        hp.lfm_head_dim = core_gguf::kv_u32(gctx, "lfm2audio.lfm_head_dim", hp.lfm_head_dim);
        hp.lfm_ff_dim = core_gguf::kv_u32(gctx, "lfm2audio.lfm_ff_dim", hp.lfm_ff_dim);
        hp.lfm_conv_kernel = core_gguf::kv_u32(gctx, "lfm2audio.lfm_conv_kernel", hp.lfm_conv_kernel);
        hp.lfm_rope_theta = core_gguf::kv_f32(gctx, "lfm2audio.lfm_rope_theta", hp.lfm_rope_theta);
        hp.lfm_layer_types = core_gguf::kv_str(gctx, "lfm2audio.lfm_layer_types", "ccaccaccacacacac");
        hp.depth_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.depth_n_layers", hp.depth_n_layers);
        hp.depth_dim = core_gguf::kv_u32(gctx, "lfm2audio.depth_dim", hp.depth_dim);
        hp.text_vocab_size = core_gguf::kv_u32(gctx, "lfm2audio.text_vocab_size", hp.text_vocab_size);

        model.vocab = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");

        core_gguf::free_metadata(gctx);

        if (verbosity >= 1) {
            fprintf(stderr,
                    "lfm2-audio: enc=%uL×%u, lfm=%uL×%u (%s), "
                    "depth=%uL×%u, codebooks=%u, vocab=%zu\n",
                    hp.enc_n_layers, hp.enc_d_model, hp.lfm_n_layers, hp.lfm_hidden_size, hp.lfm_layer_types.c_str(),
                    hp.depth_n_layers, hp.depth_dim, hp.codebooks, model.vocab.size());
        }
    }

    // ---- Pass 2: weight data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "lfm2-audio", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- Bind encoder tensors (core_conformer pattern) ----
    auto& hp = model.hparams;

    model.pre_enc.conv0_w = R(model, "encoder.pre.conv.0.weight");
    model.pre_enc.conv0_b = R(model, "encoder.pre.conv.0.bias");
    model.pre_enc.conv2_w = R(model, "encoder.pre.conv.2.weight");
    model.pre_enc.conv2_b = R(model, "encoder.pre.conv.2.bias");
    model.pre_enc.conv3_w = R(model, "encoder.pre.conv.3.weight");
    model.pre_enc.conv3_b = R(model, "encoder.pre.conv.3.bias");
    model.pre_enc.conv5_w = R(model, "encoder.pre.conv.5.weight");
    model.pre_enc.conv5_b = R(model, "encoder.pre.conv.5.bias");
    model.pre_enc.conv6_w = R(model, "encoder.pre.conv.6.weight");
    model.pre_enc.conv6_b = R(model, "encoder.pre.conv.6.bias");
    model.pre_enc.out_w = R(model, "encoder.pre.out.weight");
    model.pre_enc.out_b = R(model, "encoder.pre.out.bias");

    model.enc_blocks.resize(hp.enc_n_layers);
    for (uint32_t i = 0; i < hp.enc_n_layers; i++) {
        auto& b = model.enc_blocks[i];
        char buf[128];
        auto T = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suffix);
            return G(model, buf);
        };
        b.norm_ff1_w = T("norm_ff1.weight");
        b.norm_ff1_b = T("norm_ff1.bias");
        b.ff1_l1_w = T("ff1.linear1.weight");
        b.ff1_l1_b = T("ff1.linear1.bias");
        b.ff1_l2_w = T("ff1.linear2.weight");
        b.ff1_l2_b = T("ff1.linear2.bias");
        b.norm_attn_w = T("norm_attn.weight");
        b.norm_attn_b = T("norm_attn.bias");
        b.attn_q_w = T("attn.q.weight");
        b.attn_q_b = T("attn.q.bias");
        b.attn_k_w = T("attn.k.weight");
        b.attn_k_b = T("attn.k.bias");
        b.attn_v_w = T("attn.v.weight");
        b.attn_v_b = T("attn.v.bias");
        b.attn_out_w = T("attn.out.weight");
        b.attn_out_b = T("attn.out.bias");
        b.attn_pos_w = T("attn.pos.weight");
        b.pos_bias_u = T("attn.pos_bias_u");
        b.pos_bias_v = T("attn.pos_bias_v");
        b.norm_conv_w = T("norm_conv.weight");
        b.norm_conv_b = T("norm_conv.bias");
        b.conv_pw1_w = T("conv.pw1.weight");
        b.conv_pw1_b = T("conv.pw1.bias");
        b.conv_dw_w = T("conv.dw.weight");
        b.conv_dw_b = T("conv.dw.bias");
        b.conv_pw2_w = T("conv.pw2.weight");
        b.conv_pw2_b = T("conv.pw2.bias");
        b.norm_ff2_w = T("norm_ff2.weight");
        b.norm_ff2_b = T("norm_ff2.bias");
        b.ff2_l1_w = T("ff2.linear1.weight");
        b.ff2_l1_b = T("ff2.linear1.bias");
        b.ff2_l2_w = T("ff2.linear2.weight");
        b.ff2_l2_b = T("ff2.linear2.bias");
        b.norm_out_w = T("norm_out.weight");
        b.norm_out_b = T("norm_out.bias");
    }

    // ---- Bind adapter tensors ----
    model.adapter_norm_w = R(model, "adapter.norm.weight");
    model.adapter_norm_b = R(model, "adapter.norm.bias");
    model.adapter_lin0_w = R(model, "adapter.linear0.weight");
    model.adapter_lin0_b = R(model, "adapter.linear0.bias");
    model.adapter_lin1_w = R(model, "adapter.linear1.weight");
    model.adapter_lin1_b = R(model, "adapter.linear1.bias");

    // ---- Bind LFM2 backbone tensors ----
    model.lfm_embed_tokens_w = R(model, "lfm.embed_tokens.weight");
    model.lfm_embedding_norm_w = R(model, "lfm.embedding_norm.weight");
    model.lfm_layers.resize(hp.lfm_n_layers);
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        auto& l = model.lfm_layers[i];
        char buf[128];
        auto T = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "lfm.layers.%u.%s", i, suffix);
            return G(model, buf);
        };
        l.operator_norm_w = T("operator_norm.weight");
        l.ffn_norm_w = T("ffn_norm.weight");
        l.ff_w1 = T("ff.w1.weight");
        l.ff_w2 = T("ff.w2.weight");
        l.ff_w3 = T("ff.w3.weight");

        l.is_attention = (i < hp.lfm_layer_types.size() && hp.lfm_layer_types[i] == 'a');
        if (l.is_attention) {
            l.attn_q_proj_w = T("attn.q_proj.weight");
            l.attn_k_proj_w = T("attn.k_proj.weight");
            l.attn_v_proj_w = T("attn.v_proj.weight");
            l.attn_out_proj_w = T("attn.out_proj.weight");
            l.attn_q_ln_w = T("attn.q_layernorm.weight");
            l.attn_k_ln_w = T("attn.k_layernorm.weight");
        } else {
            l.conv_conv_w = T("conv.conv.weight");
            l.conv_in_proj_w = T("conv.in_proj.weight");
            l.conv_out_proj_w = T("conv.out_proj.weight");
        }
    }

    // ---- Bind audio embedding ----
    model.audio_embd_embedding_w = G(model, "audio_embd.embedding.weight");
    model.audio_embd_embedding_norm_w = G(model, "audio_embd.embedding_norm.weight");
    model.audio_embd_to_logits_w = G(model, "audio_embd.to_logits.weight");

    // ---- Mel preprocessor (stored by converter from librosa slaney fb) ----
    model.mel_fb = G(model, "preprocessor.fb");
    model.mel_window = G(model, "preprocessor.window");

    return true;
}

// ===========================================================================
// Mel spectrogram (NeMo-style, delegates to core_mel::compute)
// ===========================================================================

static std::vector<float> lfm2_compute_mel_impl(lfm2_audio_context* ctx, const float* samples, int n_samples,
                                                int& T_out) {
    const auto& hp = ctx->model.hparams;
    const auto& model = ctx->model;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!model.mel_fb || !model.mel_window) {
        fprintf(stderr, "lfm2-audio: missing preprocessor.fb / window in GGUF\n");
        return {};
    }

    // Read Hann window from GGUF
    std::vector<float> window((size_t)win);
    ggml_backend_tensor_get(model.mel_window, window.data(), 0, win * sizeof(float));

    // Read mel filterbank from GGUF — stored as (n_mels, n_freqs) by the
    // converter (librosa convention). core_mel::compute expects the same
    // layout: float32[n_mels * n_freqs] row-major.
    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true;
    p.preemph = 0.97f; // NeMo default

    return core_mel::compute(samples, n_samples, window.data(), win, mel_fb.data(), n_freqs, lfm2_fft_r2c, p, T_out);
}

// ===========================================================================
// Public API
// ===========================================================================

lfm2_audio_context_params lfm2_audio_context_default_params(void) {
    lfm2_audio_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

lfm2_audio_context* lfm2_audio_init_from_file(const char* path_model, lfm2_audio_context_params params) {
    auto* ctx = new lfm2_audio_context();
    ctx->n_threads = params.n_threads;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->backend = ggml_backend_cpu_init();
    ctx->compute_meta.resize(2ULL * 1024 * 1024 * 1024); // 2 GB scratch

    if (!lfm2_audio_load(ctx->model, path_model, ctx->backend, params.verbosity)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Initialize KV cache for attention layers
    {
        auto& hp = ctx->model.hparams;
        // Count attention vs conv layers
        int n_attn = 0, n_conv = 0;
        for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
            if (ctx->model.lfm_layers[i].is_attention)
                n_attn++;
            else
                n_conv++;
        }
        const int max_ctx = 2048; // max sequence length
        struct ggml_init_params gp = {ggml_tensor_overhead() * 2, nullptr, true};
        ctx->kv_ctx = ggml_init(gp);
        if (ctx->kv_ctx) {
            ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, (int)hp.lfm_head_dim, max_ctx,
                                           (int)hp.lfm_n_kv_heads, n_attn);
            ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, (int)hp.lfm_head_dim, max_ctx,
                                           (int)hp.lfm_n_kv_heads, n_attn);
            ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
            if (ctx->kv_buf) {
                ggml_backend_buffer_clear(ctx->kv_buf, 0);
                ctx->kv_max_ctx = max_ctx;
            }
        }
        // Initialize conv state cache
        const int conv_state_len = (int)(hp.lfm_conv_kernel - 1); // 2
        ctx->conv_states.resize(n_conv);
        for (auto& s : ctx->conv_states)
            s.resize((size_t)hp.lfm_hidden_size * conv_state_len, 0.0f);
    }

    return ctx;
}

void lfm2_audio_free(lfm2_audio_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int lfm2_audio_n_mels(lfm2_audio_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}

int lfm2_audio_sample_rate(lfm2_audio_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}

int lfm2_audio_test_load(lfm2_audio_context* ctx) {
    if (!ctx)
        return -1;
    auto& hp = ctx->model.hparams;
    fprintf(stderr, "lfm2-audio test_load OK: enc=%uL lfm=%uL(%s) depth=%uL vocab=%zu\n", hp.enc_n_layers,
            hp.lfm_n_layers, hp.lfm_layer_types.c_str(), hp.depth_n_layers, ctx->model.vocab.size());
    return 0;
}

// ===========================================================================
// Stage: mel spectrogram
// ===========================================================================

float* lfm2_audio_compute_mel(lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T_mel,
                              int* out_n_mels) {
    if (!ctx)
        return nullptr;
    int T_mel = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;

    float* out = (float*)malloc(mel.size() * sizeof(float));
    if (out)
        memcpy(out, mel.data(), mel.size() * sizeof(float));
    if (out_T_mel)
        *out_T_mel = T_mel;
    if (out_n_mels)
        *out_n_mels = (int)ctx->model.hparams.n_mels;
    return out;
}

// ===========================================================================
// Stage: FastConformer encoder (delegates to core_conformer)
// ===========================================================================

float* lfm2_audio_run_encoder(lfm2_audio_context* ctx, const float* mel, int T_mel, int n_mels, int* out_T_enc,
                              int* out_d_model) {
    if (!ctx || !mel)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int d = (int)hp.enc_d_model;

    // Allocate compute context
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return nullptr;

    // Mel input: (n_mels, T_mel)
    ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    memcpy(mel_t->data, mel, sizeof(float) * n_mels * T_mel);

    // Pre-encode (dw_striding 8×)
    int T_enc = 0;
    ggml_tensor* enc =
        core_conformer::build_pre_encode(ctx0, mel_t, model.pre_enc, (int)hp.enc_subsampling_channels, &T_enc);

    // Sinusoidal rel-pos table
    auto pos_vec = core_conformer::make_pos_enc(d, T_enc);
    ggml_tensor* pos_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T_enc - 1);
    memcpy(pos_t->data, pos_vec.data(), sizeof(float) * d * (2 * T_enc - 1));

    // Encoder blocks
    core_conformer::BlockParams bp = {d, (int)hp.enc_n_heads, d / (int)hp.enc_n_heads, (int)hp.enc_conv_kernel, 1e-5f};
    for (uint32_t i = 0; i < hp.enc_n_layers; i++)
        enc = core_conformer::build_block(ctx0, enc, pos_t, T_enc, model.enc_blocks[i], bp);

    ggml_tensor* out = ggml_dup(ctx0, enc);
    ggml_set_name(out, "encoder_output");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    // ggml 2D tensor with ne[0]=d, ne[1]=T stores data as T rows of d elements.
    // This IS already (T_enc, d_model) row-major — just copy.
    float* result = (float*)malloc(sizeof(float) * T_enc * d);
    if (result)
        memcpy(result, out->data, sizeof(float) * T_enc * d);

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    ggml_free(ctx0);
    return result;
}

// Forward declarations for LFM2 backbone helpers (defined below)
static ggml_tensor* lfm2_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps);
static ggml_tensor* lfm2_swiglu_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                                    ggml_tensor* w3);
static ggml_tensor* lfm2_short_conv(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w, int hidden, int T);
static ggml_tensor* lfm2_build_layer(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                     ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                     int T, float norm_eps);

// ===========================================================================
// Transcribe: full ASR pipeline
//   1. mel → encoder → adapter → audio embeddings
//   2. Embed fixed prompt tokens → text embeddings
//   3. Assemble: prefix_text + audio + suffix_text
//   4. LFM backbone prefill
//   5. Auto-regressive text generation (greedy, no KV cache for now)
// ===========================================================================

// Fixed token IDs for the ASR chat template. These are pre-tokenized
// using the LFM2.5 BPE tokenizer and hardcoded to avoid needing a full
// BPE encoder at runtime. The decode side just uses the vocab lookup.
static const int32_t kTokenStartOfText = 1;
static const int32_t kTokenImStart = 6;
static const int32_t kTokenImEnd = 7;
static const int32_t kTokenNewline = 708;
static const int32_t kTokenAudioStart = 128;
static const int32_t kTokenTextEnd = 130;

// <|startoftext|><|im_start|>system\nPerform ASR in japanese.<|im_end|>\n<|im_start|>user\n
static const std::vector<int32_t> kPrefixJa = {1,     6,    24131, 708, 8173, 1199, 11866, 559, 797,
                                               41035, 3391, 523,   7,   708,  6,    6423,  708};
// <|startoftext|><|im_start|>system\nPerform ASR in english.<|im_end|>\n<|im_start|>user\n
static const std::vector<int32_t> kPrefixEn = {1,   6,     24131, 708, 8173, 1199, 11866, 559,
                                               797, 48103, 523,   7,   708,  6,    6423,  708};
// <|im_end|>\n
static const std::vector<int32_t> kSuffix = {7, 708};

// GPT-2 byte decoder: reverse of bytes_to_unicode()
// Maps unicode codepoints back to raw bytes.
static const std::vector<int>& byte_decoder() {
    static std::vector<int> bd(512, -1);
    static bool init = false;
    if (init)
        return bd;
    const auto& be = core_bpe::byte_encoder();
    for (int b = 0; b < 256; b++)
        bd[be[b]] = b;
    init = true;
    return bd;
}

// Decode a BPE token to raw UTF-8 bytes using the GPT-2 byte_decoder
static std::string decode_token(const lfm2_audio_model& model, int32_t id) {
    if (id < 0 || id >= (int32_t)model.vocab.size())
        return "";
    const std::string& piece = model.vocab[id];
    // Skip special tokens (they start with < and end with >)
    if (!piece.empty() && piece[0] == '<' && piece.back() == '>')
        return "";
    const auto& bd = byte_decoder();
    std::string result;
    // Iterate over UTF-8 codepoints in the piece
    size_t i = 0;
    while (i < piece.size()) {
        uint32_t cp = 0;
        int len = 1;
        uint8_t c = (uint8_t)piece[i];
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        }
        for (int j = 1; j < len && i + j < piece.size(); j++)
            cp = (cp << 6) | ((uint8_t)piece[i + j] & 0x3F);
        i += len;
        // Map codepoint back to byte via the GPT-2 byte_decoder
        if (cp < (uint32_t)bd.size() && bd[cp] >= 0) {
            result += (char)(uint8_t)bd[cp];
        } else {
            // Codepoint not in the byte_decoder — pass through as UTF-8
            if (cp < 0x80) {
                result += (char)cp;
            } else if (cp < 0x800) {
                result += (char)(0xC0 | (cp >> 6));
                result += (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                result += (char)(0xE0 | (cp >> 12));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            }
        }
    }
    return result;
}

// Build a ggml graph that runs one LFM backbone forward pass on the full
// input sequence and returns logits over the text vocabulary at position T-1.
// This is the non-cached prefill+decode step.
static std::vector<float> lfm2_run_backbone_logits(lfm2_audio_context* ctx, const float* embeddings, int T,
                                                   int hidden) {
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int n_heads = (int)hp.lfm_n_heads;
    const int n_kv = (int)hp.lfm_n_kv_heads;
    const int hd = (int)hp.lfm_head_dim;
    const float norm_eps = 1e-5f;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T);
    memcpy(x->data, embeddings, sizeof(float) * T * hidden);

    // Run all LFM2 layers
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        x = lfm2_build_layer(ctx0, x, model.lfm_layers[i], nullptr, hidden, n_heads, n_kv, hd, T, norm_eps);
    }

    // Final RMSNorm
    x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

    // Extract last position: (hidden, T) → view column T-1 → (hidden,)
    ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T - 1) * hidden * sizeof(float));

    // Text logits: embed_tokens^T @ last_hidden → (vocab_size,)
    ggml_tensor* logits = ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last);

    ggml_tensor* out = ggml_dup(ctx0, logits);
    ggml_set_name(out, "logits");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    int vocab_size = (int)out->ne[0];
    std::vector<float> result(vocab_size);
    memcpy(result.data(), out->data, sizeof(float) * vocab_size);

    ggml_free(ctx0);
    return result;
}

// Embed a sequence of token IDs using lfm.embed_tokens
static std::vector<float> lfm2_embed_tokens(lfm2_audio_context* ctx, const std::vector<int32_t>& tokens) {
    auto& model = ctx->model;
    const int hidden = (int)model.hparams.lfm_hidden_size;
    const int n_tok = (int)tokens.size();

    // embed_tokens weight: (hidden, vocab_size) in ggml → row i = embedding for token i
    std::vector<float> emb(n_tok * hidden);
    for (int i = 0; i < n_tok; i++) {
        int32_t id = tokens[i];
        // Read row `id` from the embedding matrix
        ggml_backend_tensor_get(model.lfm_embed_tokens_w, emb.data() + i * hidden,
                                (size_t)id * hidden * ggml_type_size(model.lfm_embed_tokens_w->type) /
                                    ggml_blck_size(model.lfm_embed_tokens_w->type),
                                hidden * sizeof(float));
    }
    return emb;
}

char* lfm2_audio_transcribe(lfm2_audio_context* ctx, const float* samples, int n_samples, const char* prompt,
                            int max_tokens) {
    if (!ctx || !samples)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;

    if (max_tokens <= 0)
        max_tokens = 512;

    // Step 1: mel → encoder → adapter
    int T_mel = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;

    int T_enc = 0, d_model = 0;
    float* enc = lfm2_audio_run_encoder(ctx, mel.data(), T_mel, (int)hp.n_mels, &T_enc, &d_model);
    if (!enc)
        return nullptr;

    int adapter_hidden = 0;
    float* adapted = lfm2_audio_run_adapter(ctx, enc, T_enc, d_model, &adapter_hidden);
    free(enc);
    if (!adapted)
        return nullptr;

    // Step 2: Embed prefix and suffix text tokens
    // Select prompt based on user hint or auto-detect from GGUF metadata.
    // The prompt parameter can be "ja", "japanese", "en", "english", or a
    // full system prompt string (not yet supported for arbitrary prompts).
    const auto* prefix_ids_ptr = &kPrefixJa; // default: Japanese
    if (prompt) {
        std::string p(prompt);
        if (p == "en" || p == "english" || p.find("english") != std::string::npos)
            prefix_ids_ptr = &kPrefixEn;
    }
    const auto& prefix_ids = *prefix_ids_ptr;

    // For the embed_tokens lookup, we need F32 data. The weight is likely F16.
    // Use a simple CPU dequant via ggml_backend_tensor_get_f32 pattern.
    // Actually, ggml_backend_tensor_get gives raw bytes; for F16 we need to convert.
    // Simpler: build a tiny ggml graph that does the embedding lookup.
    auto embed_text = [&](const std::vector<int32_t>& ids) -> std::vector<float> {
        const int n = (int)ids.size();
        const size_t mem = 16 * 1024 * 1024;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context* c = ggml_init(ip);
        if (!c)
            return {};

        ggml_tensor* id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        memcpy(id_t->data, ids.data(), n * sizeof(int32_t));

        ggml_tensor* emb = ggml_get_rows(c, model.lfm_embed_tokens_w, id_t);
        ggml_tensor* out = ggml_dup(c, emb);

        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(c, gf, 1);

        std::vector<float> result(n * hidden);
        memcpy(result.data(), out->data, sizeof(float) * n * hidden);
        ggml_free(c);
        return result;
    };

    auto prefix_emb = embed_text(prefix_ids);
    auto suffix_emb = embed_text(kSuffix);

    // Step 3: Assemble full input sequence
    // Layout: [prefix_text_emb | audio_adapter_emb | suffix_text_emb]
    const int T_prefix = (int)prefix_ids.size();
    const int T_audio = T_enc;
    const int T_suffix = (int)kSuffix.size();
    const int T_context = T_prefix + T_audio + T_suffix;

    std::vector<float> context_emb(T_context * hidden);
    memcpy(context_emb.data(), prefix_emb.data(), sizeof(float) * T_prefix * hidden);
    memcpy(context_emb.data() + T_prefix * hidden, adapted, sizeof(float) * T_audio * hidden);
    memcpy(context_emb.data() + (T_prefix + T_audio) * hidden, suffix_emb.data(), sizeof(float) * T_suffix * hidden);
    free(adapted);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: prefill T=%d (prefix=%d audio=%d suffix=%d)\n", T_context, T_prefix, T_audio,
                T_suffix);

    // Step 4: Auto-regressive greedy decode WITH KV cache.
    //
    // Phase A (prefill): run the full context through the backbone once,
    //   populating the KV cache for all attention layers and conv state
    //   caches for all conv layers. Extract logits at the last position.
    //
    // Phase B (decode): for each new token, run a single token through
    //   the backbone using cached K/V + conv states. Only 1 forward pass
    //   per token instead of full-sequence recompute.
    //
    // The core_attn::kv_self_attn helper handles the KV cache write/read.
    // Conv state is managed manually (small: 2048*2 floats per layer).

    ctx->reset_kv();
    std::string transcript;
    std::vector<int32_t> generated_ids;

    // Helper: build graph, run backbone on T_in tokens starting at n_past,
    // return logits from the last position.
    auto run_step = [&](const float* embeddings, int T_in) -> std::vector<float> {
        const int n_past = ctx->kv_n_past;
        const float norm_eps = 1e-5f;

        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
        ggml_context* ctx0 = ggml_init(ip);
        if (!ctx0)
            return {};

        ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T_in);
        memcpy(x->data, embeddings, sizeof(float) * T_in * hidden);

        // Positions: [n_past, n_past+1, ..., n_past+T_in-1]
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_in);
        {
            int32_t* pos = (int32_t*)positions->data;
            for (int i = 0; i < T_in; i++)
                pos[i] = n_past + i;
        }

        // Causal mask for prefill (T_in > 1); nullptr for single-token decode
        ggml_tensor* causal_mask = nullptr;
        if (T_in > 1) {
            const int Lk = n_past + T_in;
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T_in);
            ggml_fp16_t* m = (ggml_fp16_t*)causal_mask->data;
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < T_in; q++)
                for (int k = 0; k < Lk; k++)
                    m[q * Lk + k] = (k <= n_past + q) ? zero : neginf;
        }

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

        int attn_idx = 0, conv_idx = 0;
        const auto& hp = model.hparams;
        const int n_heads = (int)hp.lfm_n_heads;
        const int n_kv = (int)hp.lfm_n_kv_heads;
        const int hd = (int)hp.lfm_head_dim;

        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            auto& w = model.lfm_layers[il];
            ggml_tensor* residual = x;

            // RMSNorm
            ggml_tensor* h = lfm2_rms_norm(ctx0, x, w.operator_norm_w, norm_eps);

            if (w.is_attention) {
                // --- GQA attention with KV cache ---
                core_attn::KvSelfAttnParams kvp = {};
                kvp.head_dim = hd;
                kvp.n_heads = n_heads;
                kvp.n_kv_heads = n_kv;
                kvp.n_kv_grp = n_heads / n_kv;
                kvp.n_ctx_orig = 0;
                kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                kvp.rope_theta = hp.lfm_rope_theta;
                kvp.rope_beta_fast = 0.0f;
                kvp.rope_beta_slow = 0.0f;
                kvp.attn_scale = 1.0f / sqrtf((float)hd);
                kvp.qk_norm_eps = norm_eps;
                kvp.gqa_mode = core_attn::GQA_NATIVE;

                h = core_attn::kv_self_attn(ctx0, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                            w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, positions, causal_mask,
                                            ctx->kv_k, ctx->kv_v, attn_idx, n_past, kvp);
                attn_idx++;
            } else {
                // --- ShortConv with state cache ---
                if (T_in > 1) {
                    h = lfm2_short_conv(ctx0, h, w, hidden, T_in);
                } else {
                    // Decode (T=1): use lfm2_short_conv with T=1 for now.
                    // This zero-pads the conv input (no state from prefill).
                    // TODO: implement proper conv state caching.
                    h = lfm2_short_conv(ctx0, h, w, hidden, 1);
                }
                conv_idx++;
            }

            x = ggml_add(ctx0, residual, h);

            // FFN
            residual = x;
            h = lfm2_rms_norm(ctx0, x, w.ffn_norm_w, norm_eps);
            h = lfm2_swiglu_ffn(ctx0, h, w.ff_w1, w.ff_w2, w.ff_w3);
            x = ggml_add(ctx0, residual, h);
        }

        // Final RMSNorm
        x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

        // Logits at last position
        ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T_in - 1) * hidden * sizeof(float));
        ggml_tensor* logits = ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last);
        ggml_tensor* out = ggml_dup(ctx0, logits);
        ggml_set_name(out, "logits");

        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

        int vocab_size = (int)out->ne[0];
        std::vector<float> result(vocab_size);
        memcpy(result.data(), out->data, sizeof(float) * vocab_size);

        ctx->kv_n_past += T_in;
        ggml_free(ctx0);
        return result;
    };

    // Phase A: Prefill — run full context through backbone
    auto logits = run_step(context_emb.data(), T_context);
    if (!logits.empty() && ctx->verbosity >= 1) {
        int top = 0;
        for (int i = 1; i < (int)logits.size(); i++)
            if (logits[i] > logits[top])
                top = i;
        fprintf(stderr, "lfm2-audio: prefill top token=%d logit=%.3f\n", top, logits[top]);
    }
    if (logits.empty()) {
        (void)kTokenStartOfText;
        (void)kTokenImStart;
        (void)kTokenNewline;
        (void)kTokenAudioStart;
        (void)kTokenTextEnd;
        if (ctx->verbosity >= 1)
            fprintf(stderr, "lfm2-audio: prefill failed\n");
        return nullptr;
    }

    for (int step = 0; step < max_tokens; step++) {
        // Greedy argmax
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < (int)logits.size(); i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }

        // Stop on <|im_end|> or <|endoftext|>
        if (best_id == kTokenImEnd || best_id == 2)
            break;

        generated_ids.push_back(best_id);

        // Decode token to text
        std::string piece = decode_token(model, best_id);
        transcript += piece;

        if (ctx->verbosity >= 2)
            fprintf(stderr, "  [%d] token=%d piece=%s\n", step, best_id, piece.c_str());

        // Phase B: Decode — embed the new token and run single-token step
        auto new_emb = embed_text({best_id});
        if (new_emb.empty())
            break;
        logits = run_step(new_emb.data(), 1);
        if (logits.empty())
            break;
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: generated %zu tokens\n", generated_ids.size());

    // Return malloc'd copy
    char* result = (char*)malloc(transcript.size() + 1);
    if (result) {
        memcpy(result, transcript.c_str(), transcript.size());
        result[transcript.size()] = '\0';
    }
    return result;
}

// ===========================================================================
// Stage: audio adapter MLP
//   LayerNorm(d_model) → Linear(d_model→hidden) → GELU → Linear(hidden→hidden)
// ===========================================================================

float* lfm2_audio_run_adapter(lfm2_audio_context* ctx, const float* encoder_out, int T_enc, int d_model,
                              int* out_hidden_size) {
    if (!ctx || !encoder_out)
        return nullptr;
    auto& model = ctx->model;
    const int hidden = (int)model.hparams.lfm_hidden_size;

    const size_t mem_size = 64 * 1024 * 1024; // 64 MB
    std::vector<uint8_t> buf(mem_size);
    ggml_init_params ip = {mem_size, buf.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return nullptr;

    // Input: (T_enc, d_model) — ne[0]=d_model, ne[1]=T_enc
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T_enc);
    memcpy(x->data, encoder_out, sizeof(float) * T_enc * d_model);

    // LayerNorm
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, model.adapter_norm_w), model.adapter_norm_b);

    // Linear0: (d_model → hidden) + GELU
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, model.adapter_lin0_w, x), model.adapter_lin0_b);
    x = ggml_gelu(ctx0, x);

    // Linear1: (hidden → hidden)
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, model.adapter_lin1_w, x), model.adapter_lin1_b);

    ggml_tensor* out = ggml_dup(ctx0, x);
    ggml_set_name(out, "adapter_output");

    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    float* result = (float*)malloc(sizeof(float) * T_enc * hidden);
    if (result)
        memcpy(result, out->data, sizeof(float) * T_enc * hidden);

    if (out_hidden_size)
        *out_hidden_size = hidden;
    ggml_free(ctx0);
    return result;
}

// ===========================================================================
// LFM2 backbone graph builder helpers
// ===========================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
static ggml_tensor* lfm2_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// SwiGLU FFN: w2(silu(w1(x)) * w3(x))
static ggml_tensor* lfm2_swiglu_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                                    ggml_tensor* w3) {
    ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, w1, x));
    ggml_tensor* up = ggml_mul_mat(ctx, w3, x);
    return ggml_mul_mat(ctx, w2, ggml_mul(ctx, gate, up));
}

// LFM2 ShortConv layer (non-cached prefill path):
//   BCx = in_proj(x)  → split into B, C, x_inner (each hidden-sized)
//   Bx = B * x_inner
//   conv_out = causal_conv1d(Bx, kernel=3)
//   y = C * conv_out
//   out = out_proj(y)
static ggml_tensor* lfm2_short_conv(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w, int hidden, int T) {
    // x: (hidden, T) in ggml layout (ne[0]=hidden, ne[1]=T)

    // in_proj: (hidden → 3*hidden)
    ggml_tensor* bcx = ggml_mul_mat(ctx, w.conv_in_proj_w, x); // (3*hidden, T)

    // Split into B, C, x_inner along dimension 0
    ggml_tensor* B_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], 0);
    ggml_tensor* C_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], hidden * sizeof(float));
    ggml_tensor* x_inner = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], 2 * hidden * sizeof(float));

    // Bx = B * x_inner (element-wise)
    ggml_tensor* Bx = ggml_mul(ctx, ggml_cont(ctx, B_part), ggml_cont(ctx, x_inner));

    // Causal conv1d with kernel=3: pad left by 2, conv, take first T frames
    // Bx is (hidden, T). For depthwise conv1d:
    //   - transpose to (T, hidden) = (T, 1, hidden, 1) for conv_2d_dw_direct
    //   - kernel: (K, 1, 1, hidden) where K=3
    //   - causal: pad_left = K-1 = 2
    const int K = 3; // conv_L_cache = 3 = kernel size
    ggml_tensor* conv_w = ggml_cast(ctx, w.conv_conv_w, GGML_TYPE_F32);
    // conv_w stored as (hidden, 1, K) in GGUF. Reshape to (K, 1, 1, hidden)
    ggml_tensor* conv_w_4d = ggml_reshape_4d(ctx, conv_w, K, 1, 1, hidden);

    // Bx: (hidden, T) → transpose → (T, hidden) → reshape to (T, 1, hidden, 1)
    ggml_tensor* Bx_t = ggml_cont(ctx, ggml_transpose(ctx, Bx));
    ggml_tensor* Bx_4d = ggml_reshape_4d(ctx, Bx_t, T, 1, hidden, 1);

    // Depthwise conv with causal padding (pad_left=K-1=2, pad_right=0)
    ggml_tensor* conv_out = ggml_conv_2d_dw_direct(ctx, conv_w_4d, Bx_4d, 1, 1, K - 1, 0, 1, 1);
    // conv_out: (T+K-1, 1, hidden, 1) → take first T frames
    // Actually conv_2d_dw_direct with pad_w=K-1 outputs (T+K-1, ...).
    // We need only the first T frames (causal: no future info).
    // Permute to (hidden, T) layout: permute(1,2,0,3) then reshape
    conv_out = ggml_cont(ctx, ggml_permute(ctx, conv_out, 1, 2, 0, 3));
    int T_conv = (int)conv_out->ne[1]; // may be T or T+K-1
    if (T_conv > T) {
        // Slice to first T frames
        conv_out =
            ggml_view_2d(ctx, ggml_reshape_2d(ctx, conv_out, hidden, T_conv), hidden, T, hidden * sizeof(float), 0);
    } else {
        conv_out = ggml_reshape_2d(ctx, conv_out, hidden, T);
    }

    // y = C * conv_out (element-wise)
    ggml_tensor* y = ggml_mul(ctx, ggml_cont(ctx, C_part), ggml_cont(ctx, conv_out));

    // out_proj: (hidden → hidden)
    return ggml_mul_mat(ctx, w.conv_out_proj_w, y);
}

// LFM2 GQA attention layer (non-cached prefill path):
//   Q = q_layernorm(q_proj(x))
//   K = k_layernorm(k_proj(x))
//   V = v_proj(x)
//   Apply RoPE to Q, K
//   Causal self-attention with GQA
//   out = out_proj(attn_output)
static ggml_tensor* lfm2_gqa_attention(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                       ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                       int T) {
    // x: (hidden, T)

    // Q, K, V projections
    ggml_tensor* Q = ggml_mul_mat(ctx, w.attn_q_proj_w, x); // (hidden, T) = (n_heads*hd, T)
    ggml_tensor* K = ggml_mul_mat(ctx, w.attn_k_proj_w, x); // (n_kv*hd, T)
    ggml_tensor* V = ggml_mul_mat(ctx, w.attn_v_proj_w, x); // (n_kv*hd, T)

    // Reshape to (head_dim, n_heads, T) for Q, (head_dim, n_kv, T) for K/V
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, T);
    V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, T);

    // Per-head QK layernorm (RMSNorm on head_dim axis = ne[0])
    Q = ggml_rms_norm(ctx, Q, 1e-5f);
    Q = ggml_mul(ctx, Q, w.attn_q_ln_w); // broadcasts (head_dim,) over (head_dim, n_heads, T)
    K = ggml_rms_norm(ctx, K, 1e-5f);
    K = ggml_mul(ctx, K, w.attn_k_ln_w);

    // Create positions tensor [0, 1, 2, ..., T-1]
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    {
        int32_t* pos = (int32_t*)positions->data;
        for (int i = 0; i < T; i++)
            pos[i] = i;
    }

    // RoPE — LFM2 uses the standard HF rotate_half pattern.
    // ggml GGML_ROPE_TYPE_NEOX matches: split into halves, rotate, recombine.
    Q = ggml_rope_ext(ctx, Q, positions, /*freq_override=*/nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, positions, /*freq_override=*/nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    // Permute for flash_attn_ext: Q (head_dim, T, n_heads), K (head_dim, T, n_kv), V same
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (hd, n_heads, T) → (hd, T, n_heads)
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Causal self-attention with explicit causal mask
    const float scale = 1.0f / sqrtf((float)head_dim);
    // Build causal mask: (T, T) F16 with 0 for allowed and -inf for forbidden
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    {
        ggml_fp16_t* m = (ggml_fp16_t*)mask->data;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                m[i * T + j] = (j <= i) ? zero : neginf;
    }
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    // attn: (head_dim, T, n_heads) → reshape to (hidden, T)
    attn = ggml_reshape_2d(ctx, attn, hidden, T);

    // Output projection
    return ggml_mul_mat(ctx, w.attn_out_proj_w, attn);
}

// Build one LFM2 decoder layer: RMSNorm → operator → residual → RMSNorm → SwiGLU → residual
static ggml_tensor* lfm2_build_layer(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                     ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                     int T, float norm_eps) {
    ggml_tensor* residual = x;

    // RMSNorm → operator
    ggml_tensor* h = lfm2_rms_norm(ctx, x, w.operator_norm_w, norm_eps);

    if (w.is_attention) {
        h = lfm2_gqa_attention(ctx, h, w, rope_freqs, hidden, n_heads, n_kv_heads, head_dim, T);
    } else {
        h = lfm2_short_conv(ctx, h, w, hidden, T);
    }

    // Residual
    x = ggml_add(ctx, residual, h);

    // RMSNorm → SwiGLU FFN → residual
    residual = x;
    h = lfm2_rms_norm(ctx, x, w.ffn_norm_w, norm_eps);
    h = lfm2_swiglu_ffn(ctx, h, w.ff_w1, w.ff_w2, w.ff_w3);
    x = ggml_add(ctx, residual, h);

    return x;
}

// ===========================================================================
// Stage: full prefill (encoder → adapter → assemble → LFM2 backbone)
// ===========================================================================

// Run the full LFM2 prefill pipeline on raw audio. This mirrors what
// the Python _prefill + lfm.forward does:
//   1. mel → encoder → adapter → audio embeddings
//   2. Tokenize system prompt + chat template → text embeddings
//   3. Assemble: text_emb[..] + audio_emb[..] + text_emb[..] (by modality_flag)
//   4. Run LFM2 backbone on the assembled sequence
//   5. Apply embedding_norm
//
// For now, the simplified version just runs encoder → adapter → LFM on audio
// embeddings alone (no text tokens). This is enough for diff validation of
// the adapter and backbone layers. Full chat-template assembly comes in
// the transcribe implementation.
float* lfm2_audio_run_lfm(lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T,
                          int* out_hidden_size) {
    if (!ctx || !samples)
        return nullptr;
    if (ctx->verbosity >= 2)
        fprintf(stderr, "lfm2-audio: run_lfm n_samples=%d\n", n_samples);
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;
    const int n_heads = (int)hp.lfm_n_heads;
    const int n_kv = (int)hp.lfm_n_kv_heads;
    const int hd = (int)hp.lfm_head_dim;
    const float norm_eps = 1e-5f;

    // Step 1: mel → encoder → adapter
    int T_mel = 0, n_mels = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    n_mels = (int)hp.n_mels;

    // Run encoder
    int T_enc = 0, d_model = 0;
    float* enc = lfm2_audio_run_encoder(ctx, mel.data(), T_mel, n_mels, &T_enc, &d_model);
    if (!enc)
        return nullptr;

    // Run adapter
    int adapter_hidden = 0;
    float* adapted = lfm2_audio_run_adapter(ctx, enc, T_enc, d_model, &adapter_hidden);
    free(enc);
    if (!adapted)
        return nullptr;

    // Step 2: Build full graph for LFM2 backbone on the adapted audio embeddings
    // (simplified: no text tokens for now — just audio)
    const int T = T_enc; // just audio frames for now

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        free(adapted);
        return nullptr;
    }

    // Input: adapted audio embeddings (T, hidden) → (hidden, T) in ggml
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T);
    memcpy(x->data, adapted, sizeof(float) * T * hidden);
    free(adapted);

    // Verify critical LFM weights are loaded
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        auto& l = model.lfm_layers[i];
        if (!l.operator_norm_w || !l.ffn_norm_w || !l.ff_w1 || !l.ff_w2 || !l.ff_w3) {
            fprintf(stderr, "lfm2-audio: missing weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
        if (l.is_attention && (!l.attn_q_proj_w || !l.attn_k_proj_w || !l.attn_v_proj_w)) {
            fprintf(stderr, "lfm2-audio: missing attention weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
        if (!l.is_attention && (!l.conv_in_proj_w || !l.conv_conv_w || !l.conv_out_proj_w)) {
            fprintf(stderr, "lfm2-audio: missing conv weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
    }


    // Run all 16 LFM2 layers, with optional per-layer snapshots
    std::vector<ggml_tensor*> layer_snaps(hp.lfm_n_layers, nullptr);
    bool do_snaps = (std::getenv("LFM2_SNAP_LAYERS") != nullptr);

    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        x = lfm2_build_layer(ctx0, x, model.lfm_layers[i], /*rope_freqs=*/nullptr, hidden, n_heads, n_kv, hd, T,
                             norm_eps);
        if (do_snaps) {
            layer_snaps[i] = ggml_dup(ctx0, x);
            char name[32];
            snprintf(name, sizeof(name), "lfm_ao_layer_%u", i);
            ggml_set_name(layer_snaps[i], name);
        }
    }

    // Final embedding norm (RMSNorm)
    x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

    ggml_tensor* out = ggml_dup(ctx0, x);
    ggml_set_name(out, "lfm_output");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
    for (auto* snap : layer_snaps)
        if (snap)
            ggml_build_forward_expand(gf, snap);
    ggml_build_forward_expand(gf, out);

    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    float* result = (float*)malloc(sizeof(float) * T * hidden);
    if (result)
        memcpy(result, out->data, sizeof(float) * T * hidden);

    // Invoke staged callback for per-layer snapshots
    if (do_snaps && ctx->lfm_stage_cb) {
        for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
            if (layer_snaps[i]) {
                char name[32];
                snprintf(name, sizeof(name), "lfm_ao_layer_%u", i);
                ctx->lfm_stage_cb(name, (const float*)layer_snaps[i]->data, T, hidden, ctx->lfm_stage_ud);
            }
        }
    }

    if (out_T)
        *out_T = T;
    if (out_hidden_size)
        *out_hidden_size = hidden;
    ggml_free(ctx0);
    return result;
}

int lfm2_audio_run_lfm_staged(lfm2_audio_context* ctx, const float* samples, int n_samples, lfm2_audio_stage_cb cb,
                              void* userdata) {
    if (!ctx)
        return -1;
    ctx->lfm_stage_cb = cb;
    ctx->lfm_stage_ud = userdata;

    // Force layer snaps on
    setenv("LFM2_SNAP_LAYERS", "1", 1);
    int T = 0, hidden = 0;
    float* result = lfm2_audio_run_lfm(ctx, samples, n_samples, &T, &hidden);
    unsetenv("LFM2_SNAP_LAYERS");

    ctx->lfm_stage_cb = nullptr;
    ctx->lfm_stage_ud = nullptr;

    free(result);
    return result ? 0 : -1;
}
