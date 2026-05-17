// voxcpm2_tts.cpp — VoxCPM2 TTS runtime implementation.
//
// Architecture:
//   TSLM:   28L MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, LongRoPE
//   RALM:   8L  MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, no RoPE
//   LocEnc: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, CLS token
//   LocDiT: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, sinusoidal time
//   FSQ:    Linear(2048->512)->tanh->round(x*9)/9->Linear(512->2048)
//   VAE:    causal transposed convolutions, Snake1d, weight-norm, SR conditioning
//
// Loading: two-pass GGUF via core_gguf::open_metadata / load_weights.
static int g_cpu_n_threads = 4;
// Matmul:  tiny ggml graphs on a shared CPU backend (g_cpu_backend) for
//          matrix-vector; manual loop (get_row_f32) for matrix-matrix prefill.
// KV cache: manual std::vector<float> per layer (CPU side).

#include "voxcpm2_tts.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static bool vox_env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0;
}

static double vox_now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Shared CPU backend for tiny ggml graph matmuls
// ---------------------------------------------------------------------------

static ggml_backend_t g_cpu_backend = nullptr;

static ggml_backend_t get_cpu_backend() {
    if (!g_cpu_backend) {
        g_cpu_backend = ggml_backend_cpu_init();
    }
    return g_cpu_backend;
}

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct vox_hparams {
    // TSLM
    uint32_t tslm_n_layers = 28;
    uint32_t tslm_d_model = 2048;
    uint32_t tslm_n_heads = 16;
    uint32_t tslm_n_kv = 2;
    uint32_t tslm_head_dim = 128;
    uint32_t tslm_ff_dim = 6144;
    uint32_t tslm_max_pos = 32768;
    float tslm_rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;

    // RALM
    uint32_t ralm_n_layers = 8;
    uint32_t ralm_d_model = 2048;
    uint32_t ralm_n_heads = 16;
    uint32_t ralm_n_kv = 2;
    uint32_t ralm_head_dim = 128;
    uint32_t ralm_ff_dim = 6144;

    // LocEnc
    uint32_t locenc_n_layers = 12;
    uint32_t locenc_d_model = 1024;
    uint32_t locenc_n_heads = 16;
    uint32_t locenc_n_kv = 16;
    uint32_t locenc_head_dim = 128;
    uint32_t locenc_ff_dim = 4096;

    // LocDiT
    uint32_t locdit_n_layers = 12;
    uint32_t locdit_d_model = 1024;
    uint32_t locdit_n_heads = 16;
    uint32_t locdit_n_kv = 16;
    uint32_t locdit_head_dim = 128;
    uint32_t locdit_ff_dim = 4096;

    // Tokenizer
    uint32_t n_vocab = 100000;
    uint32_t audio_start_token = 0;

    // Patch / VAE
    uint32_t patch_frames = 4;
    uint32_t patch_dim = 256;
};

// ---------------------------------------------------------------------------
// Weight tensor structs
// ---------------------------------------------------------------------------

struct vox_lm_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_enc_layer {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_vae_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct vox_weights {
    // TSLM
    ggml_tensor* tslm_token_embd = nullptr;
    ggml_tensor* tslm_output_norm = nullptr;
    // NOTE: tslm.lm_head.weight is NOT in the GGUF — omitted intentionally
    std::vector<vox_lm_layer> tslm_layers;

    // RoPE factors (LongRoPE)
    ggml_tensor* tslm_rope_short = nullptr;
    ggml_tensor* tslm_rope_long = nullptr;

    // RALM
    ggml_tensor* ralm_output_norm = nullptr;
    std::vector<vox_lm_layer> ralm_layers;

    // FSQ
    ggml_tensor* fsq_in_proj_w = nullptr;
    ggml_tensor* fsq_in_proj_b = nullptr;
    ggml_tensor* fsq_out_proj_w = nullptr;
    ggml_tensor* fsq_out_proj_b = nullptr;

    // LocEnc
    ggml_tensor* locenc_cls_token = nullptr;
    ggml_tensor* locenc_in_proj_w = nullptr; // [64, 1024] feat_dim→d_model
    ggml_tensor* locenc_in_proj_b = nullptr; // [1024]
    ggml_tensor* locenc_norm_w = nullptr;
    std::vector<vox_enc_layer> locenc_layers;

    // LocDiT
    ggml_tensor* locdit_in_proj_w = nullptr; // [64, 1024] raw latent→d_model
    ggml_tensor* locdit_in_proj_b = nullptr;
    ggml_tensor* locdit_cond_proj_w = nullptr; // [64, 1024] cond patch→d_model
    ggml_tensor* locdit_cond_proj_b = nullptr;
    ggml_tensor* locdit_time_mlp_0_w = nullptr; // [1024, 1024] sinusoidal→hidden
    ggml_tensor* locdit_time_mlp_0_b = nullptr;
    ggml_tensor* locdit_time_mlp_1_w = nullptr; // [1024, 1024] hidden→t_emb
    ggml_tensor* locdit_time_mlp_1_b = nullptr;
    ggml_tensor* locdit_dt_mlp_0_w = nullptr; // [1024, 1024] dt sinusoidal→hidden
    ggml_tensor* locdit_dt_mlp_0_b = nullptr;
    ggml_tensor* locdit_dt_mlp_1_w = nullptr; // [1024, 1024] hidden→dt_emb
    ggml_tensor* locdit_dt_mlp_1_b = nullptr;
    ggml_tensor* locdit_norm_w = nullptr;
    ggml_tensor* locdit_out_proj_w = nullptr; // [1024, 64] d_model→feat_dim
    ggml_tensor* locdit_out_proj_b = nullptr;
    std::vector<vox_enc_layer> locdit_layers;

    // Projection heads
    ggml_tensor* enc_to_lm_w = nullptr;
    ggml_tensor* enc_to_lm_b = nullptr;
    ggml_tensor* lm_to_dit_w = nullptr;
    ggml_tensor* lm_to_dit_b = nullptr;
    ggml_tensor* res_to_dit_w = nullptr;
    ggml_tensor* res_to_dit_b = nullptr;
    ggml_tensor* fusion_w = nullptr; // proj.fusion.weight [4096, 2048]
    ggml_tensor* fusion_b = nullptr; // proj.fusion.bias   [2048]

    // Stop predictor
    ggml_tensor* stop_proj_w = nullptr;
    ggml_tensor* stop_proj_b = nullptr;
    ggml_tensor* stop_head_w = nullptr; // [2048, 2] — no bias

    // VAE decoder (optional — graceful degradation if absent)
    // Weights stored in ctx->tensors under "vae.dec.*" keys
    vox_vae_conv vae_in_conv;
    vox_vae_conv vae_out_conv;
    ggml_tensor* vae_out_snake_a = nullptr;
    ggml_tensor* vae_sr_cond_w = nullptr;
    ggml_tensor* vae_sr_cond_b = nullptr;
};

// ---------------------------------------------------------------------------
// KV cache — manual CPU float storage, one vector per layer
// ---------------------------------------------------------------------------

struct vox_kv_cache {
    int n_layers = 0;
    int n_kv = 0;
    int head_dim = 0;
    int max_ctx = 0;
    int n_past = 0;
    // k_cache[l]: [max_ctx * n_kv * head_dim] row-major
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    void init(int layers, int kv_heads, int hd, int max_context) {
        n_layers = layers;
        n_kv = kv_heads;
        head_dim = hd;
        max_ctx = max_context;
        k_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        v_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        n_past = 0;
    }

    void reset() { n_past = 0; }
};

// ---------------------------------------------------------------------------
// Tokenizer — BPE with GPT-2 byte encoding
// ---------------------------------------------------------------------------

struct vox_tokenizer {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" -> rank
    std::vector<std::string> id_to_token;
    int32_t audio_start_token = -1;
    int32_t eos_token = -1;
};

// ---------------------------------------------------------------------------
// MT19937 state (needed by voxcpm2_context for CFM noise generation)
struct mt19937_state {
    uint32_t mt[624];
    int mti;
};

// ---------------------------------------------------------------------------
// Context struct
// ---------------------------------------------------------------------------

struct voxcpm2_context {
    vox_hparams hp;
    vox_weights weights;
    vox_tokenizer tokenizer;
    vox_kv_cache tslm_kv;
    vox_kv_cache ralm_kv;

    // GGUF weight storage — owned here
    ggml_context* ggml_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Runtime params
    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    bool flash_attn = true;
    int inference_steps = 10;
    float cfg_value = 2.0f;
    int max_len = 2000;

    // RNG for CFM noise generation (seeded per synthesis call)
    mt19937_state rng;
};

// Stream struct
struct voxcpm2_stream {
    voxcpm2_context* ctx = nullptr;
    std::vector<float> all_pcm;
    std::vector<float> chunk_buf;
    int chunk_offset = 0;
    bool done = false;
};

// ---------------------------------------------------------------------------
// get_row_f32: copy one row from any weight tensor type to float
// Handles F32, F16, Q4_K (and any type with a to_float trait) via
// ggml_get_type_traits()->to_float.
// ---------------------------------------------------------------------------

static void get_row_f32(const ggml_tensor* t, int row, float* out) {
    const int cols = (int)t->ne[0];
    if (t->type == GGML_TYPE_F32) {
        const float* src = (const float*)((const char*)t->data + (size_t)row * t->nb[1]);
        std::memcpy(out, src, (size_t)cols * sizeof(float));
        return;
    }
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float) {
        const void* src = (const char*)t->data + (size_t)row * t->nb[1];
        tt->to_float(src, out, cols);
        return;
    }
    std::memset(out, 0, (size_t)cols * sizeof(float));
}

// tensor_data_f32: for tensors guaranteed to be F32 (norms, biases).
static const float* tensor_data_f32(const ggml_tensor* t) {
    return (const float*)t->data;
}

// ---------------------------------------------------------------------------
// matmul_mv_ggml: W[rows,cols] x v[cols] -> out[rows] via tiny ggml graph.
// Uses ggml_mul_mat, which dispatches to SIMD / Q4_K native dot products.
// ---------------------------------------------------------------------------

static void matmul_mv_ggml(ggml_backend_t cpu_be, ggml_tensor* W, const float* v, int cols, float* out, int rows) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/(size_t)2 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context* tmp_ctx = ggml_init(ip);

    // Input vector as a 2-D column: [cols, 1]
    ggml_tensor* v_t = ggml_new_tensor_2d(tmp_ctx, GGML_TYPE_F32, cols, 1);
    std::memcpy(v_t->data, v, (size_t)cols * sizeof(float));

    // ggml_mul_mat(a, b): a[ne0, ne1] x b[ne0, 1] -> result[ne1, 1]
    // W tensor from GGUF: ne[0] = inner dim, ne[1] = outer dim
    // We need W->ne[0] == cols for the matmul to work
    if ((int)W->ne[0] != cols) {
        fprintf(stderr, "voxcpm2: matmul_mv_ggml dim mismatch: W[%lld,%lld] x v[%d] (expected W->ne[0]==%d)\n",
                (long long)W->ne[0], (long long)W->ne[1], cols, cols);
    }
    ggml_tensor* result = ggml_mul_mat(tmp_ctx, W, v_t);

    ggml_cgraph* gf = ggml_new_graph(tmp_ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_cpu_set_n_threads(cpu_be, g_cpu_n_threads);
    ggml_backend_graph_compute(cpu_be, gf);

    std::memcpy(out, result->data, (size_t)rows * sizeof(float));
    ggml_free(tmp_ctx);
}

static void matmul_mv(ggml_backend_t cpu_be, ggml_tensor* W, const float* v, int cols, float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
}

static void matmul_mv_bias(ggml_backend_t cpu_be, ggml_tensor* W, ggml_tensor* b_t, const float* v, int cols,
                           float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
    const float* b = tensor_data_f32(b_t);
    for (int i = 0; i < rows; i++)
        out[i] += b[i];
}

// matmul_mm: W[rows,cols] x X[cols,T] -> Y[T,rows]  (manual loop, for prefill)
static void matmul_mm(ggml_tensor* W, const float* X, int cols, int rows, int T, float* Y) {
    std::vector<float> w_row((size_t)cols);
    for (int r = 0; r < rows; r++) {
        get_row_f32(W, r, w_row.data());
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            const float* xc = X + (size_t)t * cols;
            for (int c = 0; c < cols; c++)
                acc += w_row[c] * xc[c];
            Y[(size_t)t * rows + r] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// MT19937 + PyTorch-compatible Gaussian noise (matches torch.randn on CPU)
// Copied from vibevoice.cpp / chatterbox_s3gen.cpp (same implementation).
// (struct mt19937_state defined above, before voxcpm2_context)
// ---------------------------------------------------------------------------

static void mt19937_seed(mt19937_state& s, uint32_t seed) {
    s.mt[0] = seed;
    for (int i = 1; i < 624; i++)
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + (uint32_t)i;
    s.mti = 624;
}

static uint32_t mt19937_next(mt19937_state& s) {
    if (s.mti >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (s.mt[i] & 0x80000000u) | (s.mt[(i + 1) % 624] & 0x7FFFFFFFu);
            s.mt[i] = s.mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1)
                s.mt[i] ^= 0x9908B0DFu;
        }
        s.mti = 0;
    }
    uint32_t y = s.mt[s.mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= (y >> 18);
    return y;
}

static inline float mt_uniform_torch_float(mt19937_state& rng) {
    return (float)(mt19937_next(rng) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static void torch_normal_fill_16(float* data) {
    for (int j = 0; j < 8; j++) {
        const float u1 = 1.0f - data[j];
        const float u2 = data[j + 8];
        const float radius = sqrtf(-2.0f * logf(u1));
        const float theta = 2.0f * (float)M_PI * u2;
        data[j] = radius * cosf(theta);
        data[j + 8] = radius * sinf(theta);
    }
}

static void fill_gaussian_noise(float* data, int n, mt19937_state& rng) {
    if (n <= 0)
        return;

    if (n < 16) {
        float tmp[16];
        for (int i = 0; i < 16; i++)
            tmp[i] = mt_uniform_torch_float(rng);
        torch_normal_fill_16(tmp);
        memcpy(data, tmp, (size_t)n * sizeof(float));
        return;
    }

    for (int i = 0; i < n; i++)
        data[i] = mt_uniform_torch_float(rng);

    int i = 0;
    for (; i <= n - 16; i += 16)
        torch_normal_fill_16(data + i);

    if (i < n) {
        float* tail = data + n - 16;
        for (int j = 0; j < 16; j++)
            tail[j] = mt_uniform_torch_float(rng);
        torch_normal_fill_16(tail);
    }
}

static void fill_gaussian_noise(float* data, int n, uint32_t seed) {
    mt19937_state rng;
    mt19937_seed(rng, seed);
    fill_gaussian_noise(data, n, rng);
}

// ---------------------------------------------------------------------------
// RMS Norm (CPU, in-place-safe: x and y may differ)
// ---------------------------------------------------------------------------

static void rms_norm_cpu(const float* x, const float* w, float* y, int d, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++)
        ss += x[i] * x[i];
    ss = ss / (float)d + eps;
    float inv = 1.0f / std::sqrt(ss);
    for (int i = 0; i < d; i++)
        y[i] = x[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// SwiGLU FFN: out = W_down @ (silu(W_gate @ x) * W_up @ x)
// ---------------------------------------------------------------------------

static void swiglu_ffn_cpu(ggml_backend_t cpu_be, ggml_tensor* gate_w, ggml_tensor* up_w, ggml_tensor* down_w,
                           const float* x, int d_in, int d_ff, int d_out, float* y) {
    std::vector<float> gate(d_ff), up(d_ff), h(d_ff);
    matmul_mv(cpu_be, gate_w, x, d_in, gate.data(), d_ff);
    matmul_mv(cpu_be, up_w, x, d_in, up.data(), d_ff);
    for (int i = 0; i < d_ff; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        h[i] = g * sig * up[i];
    }
    matmul_mv(cpu_be, down_w, h.data(), d_ff, y, d_out);
}

// ---------------------------------------------------------------------------
// NEOX RoPE (LongRoPE baseline: no freq scaling, standard complex rotation)
// Operates on interleaved layout: pairs (i, i+head_dim/2) within each head.
// ---------------------------------------------------------------------------

static void rope_apply_cpu(float* qk, int head_dim, int n_heads, int pos, float theta, int /*n_ctx_orig*/) {
    for (int h = 0; h < n_heads; h++) {
        float* vec = qk + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float x0 = vec[i];
            float x1 = vec[i + head_dim / 2];
            vec[i] = x0 * cos_a - x1 * sin_a;
            vec[i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal attention step — single new token, writes into KV cache, reads full
// history [0, n_past+1), returns output [n_q * head_dim].
// ---------------------------------------------------------------------------

static void causal_attn_step(const float* q_in,  // [n_q  * hd]
                             const float* k_new, // [n_kv * hd]
                             const float* v_new, // [n_kv * hd]
                             float* out,         // [n_q  * hd]
                             vox_kv_cache& cache, int layer, int n_q, int n_kv, int hd, float attn_scale) {
    int n_past = cache.n_past;
    int seq_len = n_past + 1;
    int grp = n_q / n_kv;

    // Write new K/V into cache at position n_past
    {
        float* kd = cache.k_cache[layer].data() + (size_t)n_past * n_kv * hd;
        float* vd = cache.v_cache[layer].data() + (size_t)n_past * n_kv * hd;
        std::memcpy(kd, k_new, (size_t)n_kv * hd * sizeof(float));
        std::memcpy(vd, v_new, (size_t)n_kv * hd * sizeof(float));
    }

    std::vector<float> scores(seq_len), attn_w(seq_len);

    for (int qh = 0; qh < n_q; qh++) {
        const float* q = q_in + qh * hd;
        int kvh = qh / grp;

        // Dot products Q.K^T over cached history
        for (int t = 0; t < seq_len; t++) {
            const float* k_t = cache.k_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++)
                dot += q[d] * k_t[d];
            scores[t] = dot * attn_scale;
        }

        // Stable softmax
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++)
            if (scores[t] > max_s)
                max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            attn_w[t] = std::exp(scores[t] - max_s);
            sum_e += attn_w[t];
        }
        for (int t = 0; t < seq_len; t++)
            attn_w[t] /= sum_e;

        // Weighted sum over V
        float* o = out + qh * hd;
        std::fill(o, o + hd, 0.0f);
        for (int t = 0; t < seq_len; t++) {
            const float* v_t = cache.v_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            for (int d = 0; d < hd; d++)
                o[d] += attn_w[t] * v_t[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Bidirectional (full) attention for LocEnc / LocDiT — no mask
// x_in: [T * d] row-major, out: [T * d] row-major
// ---------------------------------------------------------------------------

static void bidir_attn_full(const float* x_in, int T, int d, ggml_tensor* q_w, ggml_tensor* k_w, ggml_tensor* v_w,
                            ggml_tensor* o_w, int n_q, int n_kv, int hd, float attn_scale, ggml_backend_t cpu_be,
                            float* out) {
    int grp = n_q / n_kv;

    std::vector<float> Q((size_t)T * n_q * hd);
    std::vector<float> K((size_t)T * n_kv * hd);
    std::vector<float> V((size_t)T * n_kv * hd);

    for (int t = 0; t < T; t++) {
        const float* xt = x_in + (size_t)t * d;
        matmul_mv(cpu_be, q_w, xt, d, Q.data() + (size_t)t * n_q * hd, n_q * hd);
        matmul_mv(cpu_be, k_w, xt, d, K.data() + (size_t)t * n_kv * hd, n_kv * hd);
        matmul_mv(cpu_be, v_w, xt, d, V.data() + (size_t)t * n_kv * hd, n_kv * hd);
    }

    std::vector<float> attn_out((size_t)T * n_q * hd, 0.0f);
    std::vector<float> scores(T), aw(T);

    for (int t = 0; t < T; t++) {
        for (int qh = 0; qh < n_q; qh++) {
            const float* q = Q.data() + (size_t)t * n_q * hd + qh * hd;
            int kvh = qh / grp;

            for (int s = 0; s < T; s++) {
                const float* ks = K.data() + (size_t)s * n_kv * hd + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++)
                    dot += q[i] * ks[i];
                scores[s] = dot * attn_scale;
            }

            float max_s = scores[0];
            for (int s = 1; s < T; s++)
                if (scores[s] > max_s)
                    max_s = scores[s];
            float sum_e = 0.0f;
            for (int s = 0; s < T; s++) {
                aw[s] = std::exp(scores[s] - max_s);
                sum_e += aw[s];
            }
            for (int s = 0; s < T; s++)
                aw[s] /= sum_e;

            float* ao = attn_out.data() + (size_t)t * n_q * hd + qh * hd;
            for (int s = 0; s < T; s++) {
                const float* vs = V.data() + (size_t)s * n_kv * hd + kvh * hd;
                for (int i = 0; i < hd; i++)
                    ao[i] += aw[s] * vs[i];
            }
        }
    }

    // Output projection: attn_out[T, n_q*hd] x o_w -> out[T, d]
    for (int t = 0; t < T; t++) {
        matmul_mv(cpu_be, o_w, attn_out.data() + (size_t)t * n_q * hd, n_q * hd, out + (size_t)t * d, d);
    }
}

// ---------------------------------------------------------------------------
// TSLM layer — single-token causal decode with KV cache
// ---------------------------------------------------------------------------

static void tslm_layer_step(voxcpm2_context* ctx, int layer, float* hidden, int pos, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_lm_layer& L = ctx->weights.tslm_layers[layer];
    int d = (int)hp.tslm_d_model;
    int n_q = (int)hp.tslm_n_heads;
    int n_kv = (int)hp.tslm_n_kv;
    int hd = (int)hp.tslm_head_dim;
    float eps = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd), attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    rope_apply_cpu(q.data(), hd, n_q, pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos);
    rope_apply_cpu(k.data(), hd, n_kv, pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos);

    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(), ctx->tslm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++)
        hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data(), d, (int)hp.tslm_ff_dim, d,
                   ffn_out.data());
    for (int i = 0; i < d; i++)
        hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// RALM layer — causal decode, no RoPE
// ---------------------------------------------------------------------------

static void ralm_layer_step(voxcpm2_context* ctx, int layer, float* hidden, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_lm_layer& L = ctx->weights.ralm_layers[layer];
    int d = (int)hp.ralm_d_model;
    int n_q = (int)hp.ralm_n_heads;
    int n_kv = (int)hp.ralm_n_kv;
    int hd = (int)hp.ralm_head_dim;
    float eps = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd), attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    // No RoPE for RALM
    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(), ctx->ralm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++)
        hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data(), d, (int)hp.ralm_ff_dim, d,
                   ffn_out.data());
    for (int i = 0; i < d; i++)
        hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// TSLM prefill — run all text tokens through, filling KV cache
// Returns last hidden state [d_model] (pre-output-norm)
// ---------------------------------------------------------------------------

static std::vector<float> tslm_prefill(voxcpm2_context* ctx, const std::vector<int32_t>& token_ids,
                                       ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int T = (int)token_ids.size();
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        int id = token_ids[t];
        if (id < 0 || id >= (int)hp.n_vocab)
            id = 0;
        get_row_f32(ctx->weights.tslm_token_embd, id, hidden.data());

        for (int l = 0; l < (int)hp.tslm_n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);
        }
        ctx->tslm_kv.n_past = t + 1;
    }

    return hidden;
}

// ---------------------------------------------------------------------------
// TSLM prefill (instrumented) — saves per-position outputs + per-layer hooks
// all_positions: if non-null, appends each position's hidden [d] (after all layers)
// layer_hooks: map from layer index → buffer; saves first N positions of that layer
// ---------------------------------------------------------------------------

struct tslm_prefill_hooks {
    std::vector<float>* all_positions = nullptr;  // [T * d] row-major
    int layer0_capture = -1;                      // layer index for "layer 0" hook
    std::vector<float>* layer0_out = nullptr;     // [N * d]
    int layer_last_capture = -1;                  // layer index for "last layer" hook
    std::vector<float>* layer_last_out = nullptr; // [N * d]
    int max_capture_positions = 8;                // capture first N positions only
};

static std::vector<float> tslm_prefill_ex(voxcpm2_context* ctx, const std::vector<int32_t>& token_ids,
                                          ggml_backend_t cpu_be, const tslm_prefill_hooks& hooks) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int T = (int)token_ids.size();
    int n_layers = (int)hp.tslm_n_layers;
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        int id = token_ids[t];
        if (id < 0 || id >= (int)hp.n_vocab)
            id = 0;
        get_row_f32(ctx->weights.tslm_token_embd, id, hidden.data());

        for (int l = 0; l < n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);

            // Per-layer hooks (capture first N positions)
            if (t < hooks.max_capture_positions) {
                if (l == hooks.layer0_capture && hooks.layer0_out) {
                    hooks.layer0_out->insert(hooks.layer0_out->end(), hidden.data(), hidden.data() + d);
                }
                if (l == hooks.layer_last_capture && hooks.layer_last_out) {
                    hooks.layer_last_out->insert(hooks.layer_last_out->end(), hidden.data(), hidden.data() + d);
                }
            }
        }
        ctx->tslm_kv.n_past = t + 1;

        // Save all-positions output (after all layers, before output norm)
        if (hooks.all_positions && t < hooks.max_capture_positions) {
            hooks.all_positions->insert(hooks.all_positions->end(), hidden.data(), hidden.data() + d);
        }
    }

    return hidden;
}

// ---------------------------------------------------------------------------
// RALM prefill — one token, fills KV cache
// ---------------------------------------------------------------------------

static std::vector<float> ralm_prefill(voxcpm2_context* ctx, const std::vector<float>& input, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.ralm_d_model;
    ctx->ralm_kv.reset();

    std::vector<float> hidden = input;
    if ((int)hidden.size() < d)
        hidden.resize(d, 0.0f);
    if ((int)hidden.size() > d)
        hidden.resize(d);

    for (int l = 0; l < (int)hp.ralm_n_layers; l++) {
        ralm_layer_step(ctx, l, hidden.data(), cpu_be);
    }
    ctx->ralm_kv.n_past = 1;

    return hidden;
}

// Multi-position RALM prefill — processes T tokens sequentially with causal attention.
// Input: [T * d] row-major (T vectors of d dimensions).
// Returns: [T * d] row-major output hidden states (pre-output-norm).
static std::vector<float> ralm_prefill_multi(voxcpm2_context* ctx, const float* input, int T, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.ralm_d_model;
    ctx->ralm_kv.reset();

    std::vector<float> all_out((size_t)T * d);
    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        std::memcpy(hidden.data(), input + (size_t)t * d, (size_t)d * sizeof(float));

        for (int l = 0; l < (int)hp.ralm_n_layers; l++) {
            ralm_layer_step(ctx, l, hidden.data(), cpu_be);
        }
        ctx->ralm_kv.n_past = t + 1;

        std::memcpy(all_out.data() + (size_t)t * d, hidden.data(), (size_t)d * sizeof(float));
    }

    return all_out;
}

// ---------------------------------------------------------------------------
// FSQ: Linear -> tanh -> round(x*9)/9 -> Linear
// ---------------------------------------------------------------------------

static std::vector<float> fsq_forward(voxcpm2_context* ctx, const float* x, ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    int d_in = 2048;
    int d_mid = 512;
    int d_out = 2048;

    std::vector<float> mid(d_mid), quantized(d_mid), out(d_out);

    matmul_mv_bias(cpu_be, W.fsq_in_proj_w, W.fsq_in_proj_b, x, d_in, mid.data(), d_mid);
    for (int i = 0; i < d_mid; i++) {
        float v = std::tanh(mid[i]);
        quantized[i] = std::round(v * 9.0f) / 9.0f;
    }
    matmul_mv_bias(cpu_be, W.fsq_out_proj_w, W.fsq_out_proj_b, quantized.data(), d_mid, out.data(), d_out);

    return out;
}

// ---------------------------------------------------------------------------
// LocEnc forward — bidirectional 12-layer transformer with CLS prepend
// Input: patch flattened [patch_frames * patch_dim]
// Returns: CLS token output [d_locenc]
// ---------------------------------------------------------------------------

static std::vector<float> locenc_forward(voxcpm2_context* ctx, const float* patch, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->weights;
    int d = (int)hp.locenc_d_model; // 1024
    int n_q = (int)hp.locenc_n_heads;
    int n_kv = (int)hp.locenc_n_kv;
    int hd = (int)hp.locenc_head_dim;
    float eps = hp.rms_norm_eps;
    float ascale = 1.0f / std::sqrt((float)hd);

    // Sequence: [CLS, patch_frames tokens]
    // PyTorch: x=[B,T,P,D=64], in_proj projects D=64→d=1024
    // Here: patch is [patch_frames * feat_dim=64], one patch of P=patch_frames frames
    int n_patch_tok = (int)hp.patch_frames; // P = 4 frames per patch
    int T = n_patch_tok + 1;                // CLS + P tokens
    int feat_dim = 64;                      // VAE latent dim per time-frame

    std::vector<float> cur((size_t)T * d, 0.0f);

    // CLS token (d-dimensional learned vector)
    if (W.locenc_cls_token) {
        std::memcpy(cur.data(), tensor_data_f32(W.locenc_cls_token), (size_t)d * sizeof(float));
    }

    // Patch tokens: project each feat_dim=64 frame to d=1024 via in_proj
    for (int f = 0; f < n_patch_tok; f++) {
        float* dst = cur.data() + (size_t)(f + 1) * d;
        const float* src = patch + (size_t)f * feat_dim;
        if (W.locenc_in_proj_w && W.locenc_in_proj_b) {
            // in_proj.weight: [feat_dim, d] stored as [d, feat_dim] in GGUF convention
            // matmul_mv_bias: W[rows=d, cols=feat_dim] x v[feat_dim] -> out[d]
            matmul_mv_bias(cpu_be, W.locenc_in_proj_w, W.locenc_in_proj_b, src, feat_dim, dst, d);
        } else {
            // fallback: copy what fits, zero-pad rest
            int copy_d = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)copy_d * sizeof(float));
        }
    }

    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    for (int l = 0; l < (int)hp.locenc_n_layers; l++) {
        const vox_enc_layer& L = W.locenc_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm1_w), normed.data() + t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w, n_q, n_kv, hd, ascale,
                        cpu_be, attn_out.data());

        for (size_t i = 0; i < (size_t)T * d; i++)
            cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm2_w), normed.data() + t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data() + t * d, d,
                           (int)hp.locenc_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + t * d;
            for (int i = 0; i < d; i++)
                ct[i] += ffn_h[i];
        }
    }

    // Return CLS token (position 0) after optional final norm
    std::vector<float> cls_out(d);
    if (W.locenc_norm_w) {
        rms_norm_cpu(cur.data(), tensor_data_f32(W.locenc_norm_w), cls_out.data(), d, eps);
    } else {
        std::memcpy(cls_out.data(), cur.data(), (size_t)d * sizeof(float));
    }

    return cls_out;
}

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// ---------------------------------------------------------------------------

static std::vector<float> sinusoidal_time_emb(float t_scalar, int dim) {
    std::vector<float> emb(dim, 0.0f);
    for (int i = 0; i < dim / 2; i++) {
        float freq = std::exp(-std::log(10000.0f) * (float)(2 * i) / (float)dim);
        emb[2 * i] = std::cos(t_scalar * freq);
        emb[2 * i + 1] = std::sin(t_scalar * freq);
    }
    return emb;
}

// ---------------------------------------------------------------------------
// LocDiT forward — bidirectional 12-layer DiT, single denoising step
//
// PyTorch signature:
//   x:    (N, C=64, T=patch_size) noisy latent — here patch_size=patch_frames=4
//   mu:   (N, hidden=2048)         conditioning from TSLM+RALM
//   t:    (N,)                     diffusion timestep in [0,1]
//   cond: (N, C=64, T'=patch_size) previous patch condition (zeros for first step)
//   dt:   (N,)                     delta time (zeros for non-mean-mode)
//
// Sequence layout:
//   [mu_reshaped (2 tokens, d=1024), t_emb (1 token), cond_proj (P tokens), x_proj (P tokens)]
//   total T_seq = 2 + 1 + P + P
//
// We extract only the x-portion (last P tokens) and out-project to feat_dim=64.
//
// Arguments here (simplified to single-batch CPU):
//   x_raw:     [feat_dim * patch_frames] = [64 * 4 = 256] noisy latent
//   mu:        [tslm_d_model = 2048] conditioning
//   t_scalar:  timestep scalar in [0,1]
//   cond_raw:  [feat_dim * patch_frames] = [256] previous patch (zeros if first step)
//   dt_scalar: delta time (0.0 for non-mean-mode)
// Returns:     [feat_dim * patch_frames] = [256] predicted velocity
// ---------------------------------------------------------------------------

static std::vector<float> locdit_forward(voxcpm2_context* ctx, const float* x_raw, const float* mu, float t_scalar,
                                         const float* cond_raw, float dt_scalar, ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W = ctx->weights;
    int d = (int)hp.locdit_d_model; // 1024
    int n_q = (int)hp.locdit_n_heads;
    int n_kv = (int)hp.locdit_n_kv;
    int hd = (int)hp.locdit_head_dim;
    float eps = hp.rms_norm_eps;
    float ascale = 1.0f / std::sqrt((float)hd);
    int feat_dim = 64;
    int P = (int)hp.patch_frames; // 4 frames per patch
    int mu_toks = 2;              // 2048 / 1024 = 2 mu tokens
    // Total sequence: [mu_toks, 1_time, P_cond, P_x] = 2+1+4+4 = 11 tokens
    int T = mu_toks + 1 + P + P;
    int x_offset = mu_toks + 1 + P; // position of x tokens in sequence

    // --- Build time embedding: sinusoidal -> two-layer MLP (Linear→SiLU→Linear) ---
    std::vector<float> t_sin = sinusoidal_time_emb(t_scalar, d);
    std::vector<float> dt_sin = sinusoidal_time_emb(dt_scalar, d);

    // time_mlp: Linear(d,d) -> SiLU -> Linear(d,d)
    auto two_layer_mlp = [&](ggml_tensor* w0, ggml_tensor* b0, ggml_tensor* w1, ggml_tensor* b1,
                             const std::vector<float>& inp) -> std::vector<float> {
        std::vector<float> h0(d), h1(d);
        if (w0 && b0) {
            matmul_mv_bias(cpu_be, w0, b0, inp.data(), d, h0.data(), d);
        } else {
            h0 = inp;
        }
        // SiLU activation: x * sigmoid(x)
        for (int i = 0; i < d; i++) {
            float x = h0[i];
            h0[i] = x / (1.0f + std::exp(-x));
        }
        if (w1 && b1) {
            matmul_mv_bias(cpu_be, w1, b1, h0.data(), d, h1.data(), d);
        } else {
            h1 = h0;
        }
        return h1;
    };

    std::vector<float> t_emb = two_layer_mlp(W.locdit_time_mlp_0_w, W.locdit_time_mlp_0_b, W.locdit_time_mlp_1_w,
                                             W.locdit_time_mlp_1_b, t_sin);
    std::vector<float> dt_emb =
        two_layer_mlp(W.locdit_dt_mlp_0_w, W.locdit_dt_mlp_0_b, W.locdit_dt_mlp_1_w, W.locdit_dt_mlp_1_b, dt_sin);
    // t_emb = t_emb + dt_emb
    for (int i = 0; i < d; i++)
        t_emb[i] += dt_emb[i];

    // --- Build sequence ---
    std::vector<float> cur((size_t)T * d, 0.0f);

    // Tokens 0..1: mu reshaped [2048] -> 2 x [1024]
    std::memcpy(cur.data(), mu, (size_t)d * sizeof(float));
    std::memcpy(cur.data() + d, mu + d, (size_t)d * sizeof(float));

    // Token 2: time embedding [1024]
    std::memcpy(cur.data() + 2 * d, t_emb.data(), (size_t)d * sizeof(float));

    // Tokens 3..3+P-1: cond_proj applied to each of P frames of cond_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(mu_toks + 1 + p) * d;
        const float* src = cond_raw + (size_t)p * feat_dim;
        if (W.locdit_cond_proj_w && W.locdit_cond_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_cond_proj_w, W.locdit_cond_proj_b, src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // Tokens 3+P..3+2P-1: in_proj applied to each of P frames of x_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(x_offset + p) * d;
        const float* src = x_raw + (size_t)p * feat_dim;
        if (W.locdit_in_proj_w && W.locdit_in_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_in_proj_w, W.locdit_in_proj_b, src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // --- Bidirectional transformer layers ---
    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    for (int l = 0; l < (int)hp.locdit_n_layers; l++) {
        const vox_enc_layer& L = W.locdit_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm1_w), normed.data() + (size_t)t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d, L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w, n_q, n_kv, hd, ascale,
                        cpu_be, attn_out.data());

        for (size_t i = 0; i < (size_t)T * d; i++)
            cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm2_w), normed.data() + (size_t)t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w, normed.data() + (size_t)t * d, d,
                           (int)hp.locdit_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + (size_t)t * d;
            for (int i = 0; i < d; i++)
                ct[i] += ffn_h[i];
        }
    }

    // --- Extract x-portion and apply final norm + out_proj ---
    // Take only the P tokens corresponding to x (positions x_offset..x_offset+P-1)
    // Output: [feat_dim * P] = [64 * 4 = 256] predicted velocity
    int out_size = feat_dim * P;
    std::vector<float> vel(out_size);

    for (int p = 0; p < P; p++) {
        const float* h_p = cur.data() + (size_t)(x_offset + p) * d;
        float* v_p = vel.data() + (size_t)p * feat_dim;
        std::vector<float> normed_p(d);
        if (W.locdit_norm_w) {
            rms_norm_cpu(h_p, tensor_data_f32(W.locdit_norm_w), normed_p.data(), d, eps);
        } else {
            std::memcpy(normed_p.data(), h_p, (size_t)d * sizeof(float));
        }
        // out_proj: [d, feat_dim] stored as [feat_dim, d] in GGUF → matmul gives [feat_dim]
        if (W.locdit_out_proj_w && W.locdit_out_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_out_proj_w, W.locdit_out_proj_b, normed_p.data(), d, v_p, feat_dim);
        } else {
            int cp = std::min(d, feat_dim);
            std::memcpy(v_p, normed_p.data(), (size_t)cp * sizeof(float));
        }
    }

    return vel;
}

// ---------------------------------------------------------------------------
// CFM Euler solve — sway schedule (t: 1->0), CFG-zero-star
//
// mu:       [tslm_d_model=2048] conditioning from TSLM+RALM
// cond_raw: [feat_dim * patch_frames = 256] previous patch latents
//           (pass all-zeros for first step)
// Returns:  [feat_dim * patch_frames = 256] denoised patch in latent space
// ---------------------------------------------------------------------------

static std::vector<float> cfm_euler_solve(voxcpm2_context* ctx, const float* mu, const float* cond_raw, int steps,
                                          float cfg, ggml_backend_t cpu_be, const float* initial_noise = nullptr) {
    int feat_dim = 64;
    int P = (int)ctx->hp.patch_frames; // 4
    int state_size = feat_dim * P;     // 256

    // Start from initial_noise if provided, otherwise zeros
    std::vector<float> x(state_size, 0.0f);
    if (initial_noise) {
        std::memcpy(x.data(), initial_noise, (size_t)state_size * sizeof(float));
    }

    for (int s = 0; s < steps; s++) {
        // Sway schedule: t from 1.0 to 0.0 linearly
        float t_cur = 1.0f - (float)s / (float)steps;
        float t_next = 1.0f - (float)(s + 1) / (float)steps;
        float dt_val = t_next - t_cur; // negative (Euler step direction)

        // dt_scalar for dt_mlp — 0.0 for non-mean-mode synthesis
        float dt_scalar = 0.0f;

        std::vector<float> v_cond = locdit_forward(ctx, x.data(), mu, t_cur, cond_raw, dt_scalar, cpu_be);

        std::vector<float> v;
        if (cfg > 1.0f) {
            // CFG-zero-star: unconditioned uses zero mu and zero cond
            std::vector<float> zero_mu(ctx->hp.tslm_d_model, 0.0f);
            std::vector<float> zero_cond(state_size, 0.0f);
            std::vector<float> v_uncond =
                locdit_forward(ctx, x.data(), zero_mu.data(), t_cur, zero_cond.data(), dt_scalar, cpu_be);
            v.resize(state_size);
            for (int i = 0; i < state_size; i++) {
                v[i] = v_uncond[i] + cfg * (v_cond[i] - v_uncond[i]);
            }
        } else {
            v = v_cond;
        }

        // Euler step
        for (int i = 0; i < state_size; i++)
            x[i] += dt_val * v[i];
    }

    return x;
}

// ---------------------------------------------------------------------------
// Stop predictor:
//   h = stop.proj(lm_hidden + bias)   [2048 → 2048]
//   logits = stop.head(silu(h))       [2048 → 2, no bias]
//   p_stop = softmax(logits)[1]        (class 1 = stop)
// ---------------------------------------------------------------------------

static float stop_score(voxcpm2_context* ctx, const float* lm_hidden, ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    if (!W.stop_proj_w || !W.stop_proj_b)
        return 0.0f;
    int d_lm = (int)ctx->hp.tslm_d_model; // 2048

    std::vector<float> h(d_lm);
    matmul_mv_bias(cpu_be, W.stop_proj_w, W.stop_proj_b, lm_hidden, d_lm, h.data(), d_lm);
    // SiLU activation (common for projection heads in MiniCPM family)
    for (int i = 0; i < d_lm; i++) {
        float v = h[i];
        h[i] = v / (1.0f + std::exp(-v));
    }

    if (!W.stop_head_w) {
        // Fallback: treat h[0] as raw logit
        return 1.0f / (1.0f + std::exp(-h[0]));
    }

    // stop.head.weight: [d_lm, 2] — matmul gives [2] logits
    float logits[2] = {0.0f, 0.0f};
    matmul_mv(cpu_be, W.stop_head_w, h.data(), d_lm, logits, 2);

    // Softmax over 2 classes, return p(stop) = logits[1]
    float max_l = std::max(logits[0], logits[1]);
    float e0 = std::exp(logits[0] - max_l);
    float e1 = std::exp(logits[1] - max_l);
    return e1 / (e0 + e1);
}

// ===========================================================================
// VAE decoder implementation
// Architecture (sequential GGUF layer numbering):
//   layer.0  : depthwise Conv1d(64,64,k=7,groups=64)  [weight_g,weight_v,bias]
//   layer.1  : pointwise Conv1d(64,2048,k=1)           [weight_g,weight_v,bias]
//   layer.2-7: upsample blocks (rates [8,6,5,2,2,2])
//              .block.0.alpha                 — Snake1d
//              .block.1.{weight_g,weight_v,bias} — CausalTransposeConv1d
//              .block.{2,3,4}.0.alpha         — Snake1d in ResUnit
//              .block.{2,3,4}.1.{weight_g,weight_v,bias} — dilated conv (dils 1,3,9)
//              .block.{2,3,4}.2.alpha         — Snake1d in ResUnit
//              .block.{2,3,4}.3.{weight_g,weight_v,bias} — 1x1 conv
//   layer.8  : final Snake1d (.alpha)
//   layer.9  : final Conv1d(32,1,k=7)         [weight_g,weight_v,bias]
//   sr_cond.{2-7}.scale_embed / bias_embed    — [channels, 4], bucket=3 for 48kHz
//
// GGUF tensor layout: weight_v stored as [k, in_ch, out_ch] (ne[0]=k, ne[1]=in_ch, ne[2]=out_ch)
//                     weight_g stored as [out_ch] (scalar per output channel)
// Weight-norm: for each (ic, oc) pair, normalize across k, scale by g[oc].
// Output for causal_conv1d: [out_ch, in_ch, k] layout.
// ===========================================================================

// ---------------------------------------------------------------------------
// Weight-norm: reconstruct W from weight_g and weight_v stored in GGUF layout.
// GGUF layout: weight_v[ki + ic*ksize + oc*ksize*in_ch]  → [k, in_ch, out_ch]
//              weight_g[oc]                               → [out_ch]
// Normalization: along dim 0 (k) for each (ic, oc) pair.
// Output layout for causal_conv1d: [out_ch, in_ch, k]
//   → w_out[ki + ic*ksize + oc*in_ch*ksize]
// ---------------------------------------------------------------------------
static std::vector<float> wn_reconstruct(const float* weight_g, const float* weight_v, int out_ch, int in_ch,
                                         int ksize) {
    // PyTorch weight_norm (dim=0 for Conv1d): g has shape [out_ch].
    // Norm is computed per out_ch across all (in_ch * ksize) elements.
    // weight_v GGUF layout: v[ki + ic*ksize + oc*ksize*in_ch]
    // Output layout: w[ki + ic*ksize + oc*in_ch*ksize] = [out_ch, in_ch, k]
    int total = out_ch * in_ch * ksize;
    std::vector<float> w(total);
    for (int oc = 0; oc < out_ch; oc++) {
        float g = weight_g[oc];
        // Compute L2 norm across ALL (in_ch * ksize) elements for this oc
        float norm_sq = 0.0f;
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ki = 0; ki < ksize; ki++) {
                float val = weight_v[ki + (size_t)ic * ksize + (size_t)oc * ksize * in_ch];
                norm_sq += val * val;
            }
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        float scale = g * inv_norm;
        for (int ic = 0; ic < in_ch; ic++) {
            for (int ki = 0; ki < ksize; ki++) {
                float vval = weight_v[ki + (size_t)ic * ksize + (size_t)oc * ksize * in_ch];
                w[ki + (size_t)ic * ksize + (size_t)oc * in_ch * ksize] = vval * scale;
            }
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Snake1d activation: x + (1/alpha) * sin(alpha * x)^2
// alpha is a per-channel learnable parameter [C].
// x_in: [C, T]  (in-place safe if x_in == x_out)
// ---------------------------------------------------------------------------
static void snake1d(const float* alpha, const float* x_in, float* x_out, int C, int T) {
    for (int c = 0; c < C; c++) {
        float a = alpha[c];
        float inv_a = (std::abs(a) > 1e-8f) ? 1.0f / a : 1.0f;
        for (int t = 0; t < T; t++) {
            float v = x_in[(size_t)c * T + t];
            float s = std::sin(a * v);
            x_out[(size_t)c * T + t] = v + inv_a * s * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d (CPU, arbitrary dilation, padding left only)
// weight: [out_ch, in_ch/groups, ksize]  (weight-norm already applied)
// bias:   [out_ch]  (may be nullptr)
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in (causal padding = (ksize-1)*dilation)
// stride: typically 1 for residual units
// groups: for depthwise conv use groups=in_ch=out_ch
// ---------------------------------------------------------------------------
static void causal_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int out_ch,
                          int in_ch, int ksize, int T_in, int stride, int dilation, int groups) {
    int T_out = T_in / stride; // causal: same length (with causal pad)
    // Causal left-padding: pad = (ksize-1)*dilation
    int pad = (ksize - 1) * dilation;
    int in_per_grp = in_ch / groups;
    int out_per_grp = out_ch / groups;

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_per_grp; oc++) {
            int oc_abs = g * out_per_grp + oc;
            float b_val = bias ? bias[oc_abs] : 0.0f;
            for (int ot = 0; ot < T_out; ot++) {
                float acc = b_val;
                int it_center = ot * stride;
                for (int k = 0; k < ksize; k++) {
                    int it = it_center - pad + k * dilation;
                    if (it < 0 || it >= T_in)
                        continue;
                    const float* w_row = weight + ((size_t)oc_abs * in_per_grp * ksize + (size_t)0 * ksize + k);
                    for (int ic = 0; ic < in_per_grp; ic++) {
                        int ic_abs = g * in_per_grp + ic;
                        float x_val = x_in[(size_t)ic_abs * T_in + it];
                        float w_val = weight[(size_t)oc_abs * in_per_grp * ksize + (size_t)ic * ksize + k];
                        acc += x_val * w_val;
                    }
                }
                x_out[(size_t)oc_abs * T_out + ot] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Causal TransposeConv1d (upsample by stride)
// Equivalent to: insert (stride-1) zeros between input samples, then causal conv.
// weight: [in_ch, out_ch, ksize]  — note transposed layout
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in * stride
// ---------------------------------------------------------------------------
static void causal_transposed_conv1d(const float* weight, const float* bias, const float* x_in, float* x_out, int in_ch,
                                     int out_ch, int ksize, int T_in, int stride) {
    int T_out = T_in * stride;
    // Causal transpose conv: output[t] sums weight[k] * x[floor((t-k)/stride)] for
    // valid positions. We implement via direct scatter-add.
    std::fill(x_out, x_out + (size_t)out_ch * T_out, 0.0f);

    // Causal padding: trim the first (ksize-1) output samples
    int trim = ksize - 1;

    for (int ic = 0; ic < in_ch; ic++) {
        for (int oc = 0; oc < out_ch; oc++) {
            // weight layout (transposed conv stored as [in_ch, out_ch, ksize]):
            const float* w_k = weight + ((size_t)ic * out_ch + oc) * ksize;
            for (int it = 0; it < T_in; it++) {
                float x_val = x_in[(size_t)ic * T_in + it];
                // Each input sample it maps to output positions it*stride + k
                for (int k = 0; k < ksize; k++) {
                    int ot_raw = it * stride + k;
                    int ot = ot_raw - trim; // causal: shift left
                    if (ot < 0 || ot >= T_out)
                        continue;
                    x_out[(size_t)oc * T_out + ot] += x_val * w_k[k];
                }
            }
        }
    }
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            float b_val = bias[oc];
            for (int t = 0; t < T_out; t++) {
                x_out[(size_t)oc * T_out + t] += b_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: get F32 tensor data from ctx->tensors by name; returns nullptr if absent
// ---------------------------------------------------------------------------
static const float* vae_tensor_f32(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
        return nullptr;
    return (const float*)it->second->data;
}

static int vae_tensor_dim(const std::map<std::string, ggml_tensor*>& tensors, const std::string& name, int dim_idx) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
        return 0;
    return (int)it->second->ne[dim_idx];
}

// ---------------------------------------------------------------------------
// CausalResidualUnit (GGUF sequential naming):
//   Tensor names follow the pattern:  prefix.0.alpha   (Snake before dilated conv)
//                                     prefix.1.{weight_g, weight_v, bias}  (dilated conv k=7)
//                                     prefix.2.alpha   (Snake before 1x1 conv)
//                                     prefix.3.{weight_g, weight_v, bias}  (1x1 conv)
//   y = x + 1x1_conv(snake(dilated_conv(snake(x))))
// ---------------------------------------------------------------------------
static void vae_residual_unit(const std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix,
                              const float* x_in, float* x_out, int C, int T, int dilation) {
    // snake0 (.0.alpha)
    std::vector<float> h1((size_t)C * T);
    const float* alpha0 = vae_tensor_f32(tensors, prefix + ".0.alpha");
    if (alpha0) {
        snake1d(alpha0, x_in, h1.data(), C, T);
    } else {
        std::memcpy(h1.data(), x_in, (size_t)C * T * sizeof(float));
    }

    // dilated causal conv: .1.{weight_g, weight_v, bias}  k=7, dil=dilation
    // This is DEPTHWISE (groups=C): weight_v shape [k, 1, C] in GGUF
    int k1 = 7;
    std::vector<float> h2((size_t)C * T, 0.0f);
    const float* g1 = vae_tensor_f32(tensors, prefix + ".1.weight_g");
    const float* v1 = vae_tensor_f32(tensors, prefix + ".1.weight_v");
    const float* b1 = vae_tensor_f32(tensors, prefix + ".1.bias");
    if (g1 && v1) {
        // Depthwise: out_ch=C, in_per_grp=1, groups=C
        auto w1 = wn_reconstruct(g1, v1, C, 1, k1);
        causal_conv1d(w1.data(), b1, h1.data(), h2.data(), C, C, k1, T, 1, dilation, C);
    } else {
        std::memcpy(h2.data(), h1.data(), (size_t)C * T * sizeof(float));
    }

    // snake2 (.2.alpha)
    std::vector<float> h3((size_t)C * T);
    const float* alpha2 = vae_tensor_f32(tensors, prefix + ".2.alpha");
    if (alpha2) {
        snake1d(alpha2, h2.data(), h3.data(), C, T);
    } else {
        std::memcpy(h3.data(), h2.data(), (size_t)C * T * sizeof(float));
    }

    // 1x1 conv: .3.{weight_g, weight_v, bias}  k=1
    std::vector<float> h4((size_t)C * T, 0.0f);
    const float* g2 = vae_tensor_f32(tensors, prefix + ".3.weight_g");
    const float* v2 = vae_tensor_f32(tensors, prefix + ".3.weight_v");
    const float* b2 = vae_tensor_f32(tensors, prefix + ".3.bias");
    if (g2 && v2) {
        auto w2 = wn_reconstruct(g2, v2, C, C, 1);
        causal_conv1d(w2.data(), b2, h3.data(), h4.data(), C, C, 1, T, 1, 1, 1);
    } else {
        std::memcpy(h4.data(), h3.data(), (size_t)C * T * sizeof(float));
    }

    // residual add
    for (size_t i = 0; i < (size_t)C * T; i++)
        x_out[i] = x_in[i] + h4[i];
}

// ---------------------------------------------------------------------------
// VAE decode: concatenated patches -> 48kHz PCM
//
// Each patch is [feat_dim * patch_frames = 64 * 4 = 256] float values.
// We concatenate them along the time axis to form latents [64, T_latent].
// The VAE decoder upsamples by 8*6*5*2*2*2 = 1920x (from 25 Hz to 48000 Hz).
//
// Tensor naming follows the GGUF sequential layer scheme:
//   layer.0  : depthwise in-conv (k=7, groups=64)
//   layer.1  : pointwise in-conv (k=1, 64->2048)
//   layer.2-7: upsample blocks (rates [8,6,5,2,2,2])
//   layer.8  : final Snake1d
//   layer.9  : final out-conv (k=7, last_ch->1)
//   sr_cond.{2-7}.scale_embed / bias_embed : [channels, 4], bucket=3 for 48kHz
//
// When VAE weights are absent, returns silence of the correct duration.
// ---------------------------------------------------------------------------

static std::vector<float> vae_decode(voxcpm2_context* ctx, const std::vector<std::vector<float>>& patches,
                                     ggml_backend_t /*cpu_be*/) {
    int n_patches = (int)patches.size();
    if (n_patches == 0)
        return {};

    int feat_dim = 64;
    int P = (int)ctx->hp.patch_frames; // 4

    // Total latent time frames = n_patches * P
    int T_lat = n_patches * P;

    // Upsampling ratios for layers 2-7: product = 8*6*5*2*2*2 = 1920
    // 25 Hz (latent rate) * 1920 = 48000 Hz output
    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    static const int n_up_blocks = 6;

    // Channel progression after each upsample block: 2048->1024->512->256->128->64->32
    static const int block_out_ch[] = {1024, 512, 256, 128, 64, 32};

    const auto& T = ctx->tensors;

    // Check if VAE weights exist -- look for the first input conv weight
    bool have_vae = (vae_tensor_f32(T, "vae.dec.layer.0.weight_g") != nullptr);

    if (!have_vae) {
        // Graceful degradation: return silence of computed duration
        // 25 Hz latent rate x 1920 upsample = 48000 Hz; P frames/patch -> P*1920 samples/patch
        std::vector<float> pcm((size_t)n_patches * (size_t)P * 1920, 0.0f);
        return pcm;
    }

    // --- Build latent tensor [feat_dim=64, T_lat] ---
    // Patches are [P * feat_dim] row-major = P frames of feat_dim each.
    // Latent tensor is [feat_dim, T_lat] = channels-first (C, T).
    std::vector<float> latents((size_t)feat_dim * T_lat, 0.0f);
    for (int n = 0; n < n_patches; n++) {
        const auto& patch = patches[n];
        for (int p = 0; p < P; p++) {
            int t = n * P + p;
            size_t patch_off = (size_t)p * feat_dim;
            if (patch_off >= patch.size())
                break;
            const float* src = patch.data() + patch_off;
            int avail = (int)std::min((size_t)feat_dim, patch.size() - patch_off);
            for (int c = 0; c < avail; c++) {
                latents[(size_t)c * T_lat + t] = src[c];
            }
        }
    }

    if (ctx->verbosity >= 2) {
        float mx = 0; for (auto v : latents) { float a = v < 0 ? -v : v; if (a > mx) mx = a; }
        fprintf(stderr, "voxcpm2 VAE: latent [%d, %d] max=%.4f\n", feat_dim, T_lat, mx);
    }

    // SR conditioning bucket for 48kHz
    int sr_bucket = 3;

    // --- Layer 0: depthwise in-conv (k=7, groups=64) ---
    // GGUF weight_v: [k=7, in_per_grp=1, out=64]  weight_g: [64]
    int Tc = T_lat;
    int Cc = feat_dim;
    std::vector<float> h;

    {
        const float* g0 = vae_tensor_f32(T, "vae.dec.layer.0.weight_g");
        const float* v0 = vae_tensor_f32(T, "vae.dec.layer.0.weight_v");
        const float* b0 = vae_tensor_f32(T, "vae.dec.layer.0.bias");
        if (g0 && v0) {
            // Depthwise: groups=feat_dim, in_per_grp=1
            auto w0 = wn_reconstruct(g0, v0, feat_dim, 1, 7);
            std::vector<float> h0((size_t)feat_dim * Tc, 0.0f);
            causal_conv1d(w0.data(), b0, latents.data(), h0.data(), feat_dim, feat_dim, 7, Tc, 1, 1, feat_dim);
            h = std::move(h0);
        } else {
            h = latents;
        }
    }
    if (ctx->verbosity >= 2) {
        float mx = 0; for (auto v : h) { float a = v < 0 ? -v : v; if (a > mx) mx = a; }
        fprintf(stderr, "voxcpm2 VAE: after layer0 (depthwise k=7): Cc=%d Tc=%d max=%.4f\n", Cc, Tc, mx);
    }

    // --- Layer 1: pointwise in-conv (k=1, 64->2048) ---
    {
        const float* g1 = vae_tensor_f32(T, "vae.dec.layer.1.weight_g");
        const float* v1 = vae_tensor_f32(T, "vae.dec.layer.1.weight_v");
        const float* b1 = vae_tensor_f32(T, "vae.dec.layer.1.bias");
        if (g1 && v1) {
            // weight_g is [out_ch] stored with varying GGUF shapes.
            // Total elements = out_ch regardless of ne[] layout.
            auto it_g1 = T.find("vae.dec.layer.1.weight_g");
            int out_ch1 = it_g1 != T.end() ? (int)ggml_nelements(it_g1->second) : 2048;
            auto w1 = wn_reconstruct(g1, v1, out_ch1, feat_dim, 1);
            std::vector<float> h1((size_t)out_ch1 * Tc, 0.0f);
            causal_conv1d(w1.data(), b1, h.data(), h1.data(), out_ch1, feat_dim, 1, Tc, 1, 1, 1);
            h = std::move(h1);
            Cc = out_ch1;
        }
    }
    if (ctx->verbosity >= 2) {
        float mx = 0; for (auto v : h) { float a = v < 0 ? -v : v; if (a > mx) mx = a; }
        fprintf(stderr, "voxcpm2 VAE: after layer1 (1x1 64->%d): Tc=%d max=%.4f\n", Cc, Tc, mx);
    }

    // --- Layers 2-7: upsample blocks ---
    for (int b = 0; b < n_up_blocks; b++) {
        int layer_idx = b + 2; // layers 2 through 7
        int up = up_rates[b];
        std::string lp = "vae.dec.layer." + std::to_string(layer_idx);

        // SR conditioning: scale_embed and bias_embed are [channels, 4]
        // GGUF layout [channels, 4] -> ne[0]=4, ne[1]=channels
        // scale_embed[c, bucket] = data[bucket + c*4]
        // Apply: x[c, t] = x[c, t] * scale[c] + bias[c]
        {
            std::string sr_pfx = "vae.dec.sr_cond." + std::to_string(layer_idx);
            const float* se = vae_tensor_f32(T, sr_pfx + ".scale_embed");
            const float* be = vae_tensor_f32(T, sr_pfx + ".bias_embed");
            if (se) {
                for (int c = 0; c < Cc; c++) {
                    float sc = se[(size_t)c * 4 + sr_bucket];
                    float bi = be ? be[(size_t)c * 4 + sr_bucket] : 0.0f;
                    for (int t = 0; t < Tc; t++) {
                        h[(size_t)c * Tc + t] = h[(size_t)c * Tc + t] * sc + bi;
                    }
                }
            }
        }

        // Snake1d before upsample: .block.0.alpha
        {
            const float* alpha_up = vae_tensor_f32(T, lp + ".block.0.alpha");
            if (alpha_up) {
                snake1d(alpha_up, h.data(), h.data(), Cc, Tc);
            }
        }

        // CausalTransposeConv1d upsample: .block.1.{weight_g, weight_v, bias}
        int out_ch_b = block_out_ch[b];
        int T_up = Tc * up;
        std::vector<float> h_up((size_t)out_ch_b * T_up, 0.0f);
        {
            const float* g_up = vae_tensor_f32(T, lp + ".block.1.weight_g");
            const float* v_up = vae_tensor_f32(T, lp + ".block.1.weight_v");
            const float* b_up = vae_tensor_f32(T, lp + ".block.1.bias");
            if (g_up && v_up) {
                // GGUF transposed conv weight_v: [k, out_ch, in_ch]
                // ne[0]=k, ne[1]=out_ch, ne[2]=in_ch
                // ksize from ne[0] of weight_v
                int ksize_up = vae_tensor_dim(T, lp + ".block.1.weight_v", 0);
                if (ksize_up <= 0)
                    ksize_up = 2 * up;
                // wn_reconstruct(g, v, out_ch=Cc, in_ch=out_ch_b, k)
                // treats weight_v as [k, in_ch=out_ch_b, out_ch=Cc]
                // output layout [Cc, out_ch_b, k] = [in_ch, out_ch, k] for causal_transposed_conv1d
                auto w_up = wn_reconstruct(g_up, v_up, Cc, out_ch_b, ksize_up);
                causal_transposed_conv1d(w_up.data(), b_up, h.data(), h_up.data(), Cc, out_ch_b, ksize_up, Tc, up);
            } else {
                // Repeat-interpolate fallback
                int copy_ch = std::min(Cc, out_ch_b);
                for (int c = 0; c < copy_ch; c++) {
                    for (int t = 0; t < Tc; t++) {
                        float val = h[(size_t)c * Tc + t];
                        for (int u = 0; u < up; u++) {
                            h_up[(size_t)c * T_up + t * up + u] = val;
                        }
                    }
                }
            }
        }

        Tc = T_up;
        Cc = out_ch_b;
        h = std::move(h_up);

        if (ctx->verbosity >= 2) {
            float mx = 0; for (size_t i = 0; i < (size_t)Cc*Tc; i++) { float a = h[i]<0?-h[i]:h[i]; if(a>mx)mx=a; }
            fprintf(stderr, "voxcpm2 VAE: block %d upsample(%d): Cc=%d Tc=%d max=%.4f\n", b, up, Cc, Tc, mx);
        }

        // 3x CausalResidualUnit: .block.{2,3,4} with dilations 1, 3, 9
        for (int r = 0; r < 3; r++) {
            int dil = (r == 0) ? 1 : (r == 1) ? 3 : 9;
            std::string rp = lp + ".block." + std::to_string(r + 2);
            std::vector<float> h_res((size_t)Cc * Tc);
            vae_residual_unit(T, rp, h.data(), h_res.data(), Cc, Tc, dil);
            h = std::move(h_res);
        }
    }

    if (ctx->verbosity >= 2) {
        float mx = 0; for (size_t i = 0; i < (size_t)Cc*Tc; i++) { float a = h[i] < 0 ? -h[i] : h[i]; if (a > mx) mx = a; }
        fprintf(stderr, "voxcpm2 VAE: after upsample blocks: Cc=%d Tc=%d max=%.4f\n", Cc, Tc, mx);
    }

    // --- Layer 8: final Snake1d ---
    {
        const float* alpha_f = vae_tensor_f32(T, "vae.dec.layer.8.alpha");
        if (alpha_f) {
            snake1d(alpha_f, h.data(), h.data(), Cc, Tc);
        }
    }

    // --- Layer 9: final out-conv (k=7, Cc->1) then Tanh ---
    std::vector<float> pcm(Tc, 0.0f);
    {
        const float* g9 = vae_tensor_f32(T, "vae.dec.layer.9.weight_g");
        const float* v9 = vae_tensor_f32(T, "vae.dec.layer.9.weight_v");
        const float* b9 = vae_tensor_f32(T, "vae.dec.layer.9.bias");
        if (g9 && v9) {
            auto w9 = wn_reconstruct(g9, v9, 1, Cc, 7);
            causal_conv1d(w9.data(), b9, h.data(), pcm.data(), 1, Cc, 7, Tc, 1, 1, 1);
            for (float& s : pcm)
                s = std::tanh(s);
        } else {
            // Mix-down fallback
            float inv_Cc = 1.0f / std::max(1, Cc);
            for (int t = 0; t < Tc; t++) {
                float mix = 0.0f;
                for (int c = 0; c < Cc; c++)
                    mix += h[(size_t)c * Tc + t];
                pcm[t] = std::tanh(mix * inv_Cc);
            }
        }
    }

    return pcm;
}
// ---------------------------------------------------------------------------
// GPT-2 byte encoder table (built lazily)
// ---------------------------------------------------------------------------

static const std::vector<int>& vox_byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool init = false;
    if (init)
        return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++)
        printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++)
        printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++)
        printable.push_back(b);
    int next_extra = 256;
    for (int b = 0; b < 256; b++) {
        bool is_p = false;
        for (int p : printable)
            if (p == b) {
                is_p = true;
                break;
            }
        bs[b] = is_p ? b : next_extra++;
    }
    init = true;
    return bs;
}

static void utf8_encode_cp(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static std::string vox_bytes_to_unicode(const char* bytes, size_t n) {
    auto& enc = vox_byte_encoder();
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        utf8_encode_cp((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// BPE merge for one byte-encoded pre-token
static void vox_bpe_one(const vox_tokenizer& tok, const std::string& word, std::vector<int32_t>& out) {
    if (word.empty())
        return;

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = (unsigned char)word[i];
        size_t len;
        if (c < 0x80)
            len = 1;
        else if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;
        else
            len = 1;
        if (i + len > word.size())
            len = 1;
        symbols.emplace_back(word, i, len);
        i += len;
    }

    if (!tok.merge_rank.empty()) {
        int max_iter = (int)symbols.size();
        for (int iter = 0; iter < max_iter && symbols.size() >= 2; iter++) {
            int best_i = -1, best_rank = INT_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); k++) {
                std::string pair = symbols[k] + " " + symbols[k + 1];
                auto it = tok.merge_rank.find(pair);
                if (it != tok.merge_rank.end() && (int)it->second < best_rank) {
                    best_rank = (int)it->second;
                    best_i = (int)k;
                }
            }
            if (best_i < 0)
                break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
    }

    for (const auto& s : symbols) {
        auto it = tok.token_to_id.find(s);
        if (it != tok.token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Per-byte fallback
            size_t j = 0;
            while (j < s.size()) {
                unsigned char c2 = (unsigned char)s[j];
                size_t len;
                if (c2 < 0x80)
                    len = 1;
                else if ((c2 & 0xE0) == 0xC0)
                    len = 2;
                else if ((c2 & 0xF0) == 0xE0)
                    len = 3;
                else if ((c2 & 0xF8) == 0xF0)
                    len = 4;
                else
                    len = 1;
                auto jt = tok.token_to_id.find(std::string(s, j, len));
                if (jt != tok.token_to_id.end())
                    out.push_back(jt->second);
                j += len;
            }
        }
    }
}

// CJK codepoint check for post-expansion
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

// Decode one UTF-8 codepoint from s at position i, advance i
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i += 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
        i += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F);
        i += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) |
                      (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
        i += 4;
        return cp;
    }
    i += 1;
    return 0xFFFD;
}

// Tokenize: SentencePiece BPE with ▁ word boundaries + CJK post-expansion
static std::vector<int32_t> vox_tokenize(const vox_tokenizer& tok, const std::string& text) {
    // Step 1: Normalize — prepend ▁, replace spaces with ▁
    // ▁ = U+2581 = \xe2\x96\x81
    std::string normalized = "\xe2\x96\x81";
    for (char c : text) {
        if (c == ' ')
            normalized += "\xe2\x96\x81";
        else
            normalized += c;
    }

    // Step 2: BPE encode using vocab ordering as merge priority
    // Start with UTF-8 codepoint symbols
    std::vector<std::string> symbols;
    {
        size_t pos = 0;
        while (pos < normalized.size()) {
            size_t prev = pos;
            utf8_next(normalized, pos);
            symbols.push_back(normalized.substr(prev, pos - prev));
        }
    }

    // Greedy BPE merge: repeatedly merge the pair with lowest merged-token ID
    while (symbols.size() > 1) {
        int best_id = INT_MAX, best_pos = -1;
        for (int k = 0; k + 1 < (int)symbols.size(); k++) {
            std::string merged = symbols[k] + symbols[k + 1];
            auto it = tok.token_to_id.find(merged);
            if (it != tok.token_to_id.end() && it->second < best_id) {
                best_id = it->second;
                best_pos = k;
            }
        }
        if (best_pos < 0)
            break;
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    // Convert symbols to IDs
    std::vector<int32_t> result;
    for (const auto& sym : symbols) {
        auto it = tok.token_to_id.find(sym);
        if (it != tok.token_to_id.end()) {
            result.push_back(it->second);
        } else {
            // Byte fallback: encode each byte as <0xNN>
            for (unsigned char c : sym) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", c);
                auto jt = tok.token_to_id.find(hex);
                if (jt != tok.token_to_id.end())
                    result.push_back(jt->second);
            }
        }
    }

    // Step 3: CJK post-expansion — split multi-char CJK tokens into individual chars
    std::vector<int32_t> expanded;
    for (int32_t id : result) {
        if (id < 0 || id >= (int32_t)tok.id_to_token.size()) {
            expanded.push_back(id);
            continue;
        }
        const std::string& ts = tok.id_to_token[id];
        // Check if token is multi-char CJK (remove ▁ prefix first)
        std::string clean = ts;
        while (clean.size() >= 3 && clean.substr(0, 3) == "\xe2\x96\x81")
            clean = clean.substr(3);
        if (clean.empty()) {
            expanded.push_back(id);
            continue;
        }
        // Count CJK codepoints
        int n_cjk = 0, n_total = 0;
        {
            size_t p = 0;
            while (p < clean.size()) {
                if (is_cjk(utf8_next(clean, p)))
                    n_cjk++;
                n_total++;
            }
        }
        if (n_cjk == n_total && n_total >= 2) {
            // Split into individual chars
            size_t p = 0;
            while (p < clean.size()) {
                size_t prev = p;
                utf8_next(clean, p);
                std::string ch = clean.substr(prev, p - prev);
                auto jt = tok.token_to_id.find(ch);
                if (jt != tok.token_to_id.end())
                    expanded.push_back(jt->second);
                else
                    expanded.push_back(id); // fallback: keep original
            }
        } else {
            expanded.push_back(id);
        }
    }
    result = expanded;

    if (tok.audio_start_token >= 0) {
        result.push_back(tok.audio_start_token);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Model loading — two-pass GGUF
// ---------------------------------------------------------------------------

static bool vox_load_weights(voxcpm2_context* ctx, const char* path) {
    using namespace core_gguf;

    // --- Pass 1: metadata ---
    gguf_context* meta = open_metadata(path);
    if (!meta)
        return false;

    vox_hparams& hp = ctx->hp;

    hp.tslm_n_layers = kv_u32(meta, "tslm.n_layers", hp.tslm_n_layers);
    hp.tslm_d_model = kv_u32(meta, "tslm.d_model", hp.tslm_d_model);
    hp.tslm_n_heads = kv_u32(meta, "tslm.n_heads", hp.tslm_n_heads);
    hp.tslm_n_kv = kv_u32(meta, "tslm.n_kv_heads", hp.tslm_n_kv);
    hp.tslm_head_dim = kv_u32(meta, "tslm.head_dim", hp.tslm_head_dim);
    hp.tslm_ff_dim = kv_u32(meta, "tslm.ff_dim", hp.tslm_ff_dim);
    hp.tslm_max_pos = kv_u32(meta, "tslm.max_pos", hp.tslm_max_pos);
    hp.tslm_rope_theta = kv_f32(meta, "tslm.rope_theta", hp.tslm_rope_theta);
    hp.rms_norm_eps = kv_f32(meta, "tslm.rms_norm_eps", hp.rms_norm_eps);

    hp.ralm_n_layers = kv_u32(meta, "ralm.n_layers", hp.ralm_n_layers);
    hp.ralm_d_model = kv_u32(meta, "ralm.d_model", hp.ralm_d_model);
    hp.ralm_n_heads = kv_u32(meta, "ralm.n_heads", hp.ralm_n_heads);
    hp.ralm_n_kv = kv_u32(meta, "ralm.n_kv_heads", hp.ralm_n_kv);
    hp.ralm_head_dim = kv_u32(meta, "ralm.head_dim", hp.ralm_head_dim);
    hp.ralm_ff_dim = kv_u32(meta, "ralm.ff_dim", hp.ralm_ff_dim);

    hp.locenc_n_layers = kv_u32(meta, "locenc.n_layers", hp.locenc_n_layers);
    hp.locenc_d_model = kv_u32(meta, "locenc.d_model", hp.locenc_d_model);
    hp.locenc_n_heads = kv_u32(meta, "locenc.n_heads", hp.locenc_n_heads);
    hp.locenc_n_kv = kv_u32(meta, "locenc.n_kv_heads", hp.locenc_n_kv);
    hp.locenc_head_dim = kv_u32(meta, "locenc.head_dim", hp.locenc_head_dim);
    hp.locenc_ff_dim = kv_u32(meta, "locenc.ff_dim", hp.locenc_ff_dim);

    hp.locdit_n_layers = kv_u32(meta, "locdit.n_layers", hp.locdit_n_layers);
    hp.locdit_d_model = kv_u32(meta, "locdit.d_model", hp.locdit_d_model);
    hp.locdit_n_heads = kv_u32(meta, "locdit.n_heads", hp.locdit_n_heads);
    hp.locdit_n_kv = kv_u32(meta, "locdit.n_kv_heads", hp.locdit_n_kv);
    hp.locdit_head_dim = kv_u32(meta, "locdit.head_dim", hp.locdit_head_dim);
    hp.locdit_ff_dim = kv_u32(meta, "locdit.ff_dim", hp.locdit_ff_dim);

    // Override n_vocab from GGUF metadata
    hp.n_vocab = kv_u32(meta, "tokenizer.n_vocab", hp.n_vocab);
    hp.audio_start_token = kv_u32(meta, "tokenizer.audio_start_id", 0);

    hp.patch_frames = kv_u32(meta, "vae.patch_frames", hp.patch_frames);
    hp.patch_dim = kv_u32(meta, "vae.patch_dim", hp.patch_dim);

    // Tokenizer: try GGUF string arrays first, then vocab blob tensor
    {
        auto tokens = kv_str_array(meta, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            ctx->tokenizer.id_to_token = tokens;
            for (size_t b = 0; b < tokens.size(); b++)
                ctx->tokenizer.token_to_id[tokens[b]] = (int32_t)b;
            auto merges = kv_str_array(meta, "tokenizer.ggml.merges");
            for (size_t b = 0; b < merges.size(); b++)
                ctx->tokenizer.merge_rank[merges[b]] = (int32_t)b;
        }
        // Note: vocab blob tensor is loaded after weights (see below)
    }

    free_metadata(meta);

    // --- Pass 2: weights ---
    ggml_backend_t cpu_be = get_cpu_backend();
    WeightLoad wl;
    if (!load_weights(path, cpu_be, "voxcpm2", wl))
        return false;

    ctx->ggml_ctx = wl.ctx;
    ctx->weight_buf = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    auto& T = ctx->tensors;
    vox_weights& W = ctx->weights;

    // Infer n_kv for LocEnc/LocDiT from K weight shapes when not in metadata.
    // K weight: ne[0]=d_model (input), ne[1]=n_kv*head_dim (output)
    {
        auto it = T.find("locenc.blk.0.attn_k.weight");
        if (it != T.end() && hp.locenc_head_dim > 0) {
            uint32_t kv_dim = (uint32_t)it->second->ne[1];
            uint32_t inferred = kv_dim / hp.locenc_head_dim;
            if (inferred > 0 && inferred != hp.locenc_n_kv) {
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "voxcpm2: locenc n_kv inferred from K weight: %u (was %u)\n", inferred,
                            hp.locenc_n_kv);
                hp.locenc_n_kv = inferred;
            }
        }
    }
    {
        auto it = T.find("locdit.blk.0.attn_k.weight");
        if (it != T.end() && hp.locdit_head_dim > 0) {
            uint32_t kv_dim = (uint32_t)it->second->ne[1];
            uint32_t inferred = kv_dim / hp.locdit_head_dim;
            if (inferred > 0 && inferred != hp.locdit_n_kv) {
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "voxcpm2: locdit n_kv inferred from K weight: %u (was %u)\n", inferred,
                            hp.locdit_n_kv);
                hp.locdit_n_kv = inferred;
            }
        }
    }

    // Check if vocab blob tensor exists and override tokenizer from it.
    // The GGUF stores each vocab entry as [uint16 len][bytes...] packed into
    // a 1-D F32 tensor where each float holds one byte value.
    {
        ggml_tensor* vocab_t = try_get(T, "tokenizer.vocab_tensor");
        if (vocab_t && ctx->tokenizer.id_to_token.empty()) {
            int n = (int)ggml_nelements(vocab_t);
            std::vector<float> fp_buf(n);
            ggml_backend_tensor_get(vocab_t, fp_buf.data(), 0, (size_t)n * sizeof(float));
            // Decode F32 -> uint8 raw bytes
            std::vector<uint8_t> raw(n);
            for (int i = 0; i < n; i++)
                raw[i] = (uint8_t)(int)fp_buf[i];
            // Parse [uint16 len][bytes len] entries
            ctx->tokenizer.id_to_token.clear();
            ctx->tokenizer.token_to_id.clear();
            int offset = 0;
            int n_vocab = (int)hp.n_vocab;
            for (int v = 0; v < n_vocab && offset + 2 <= n; v++) {
                uint16_t len = (uint16_t)(raw[offset] | (raw[offset + 1] << 8));
                offset += 2;
                if (offset + len > n)
                    break;
                std::string tok_str((const char*)(raw.data() + offset), len);
                offset += len;
                ctx->tokenizer.id_to_token.push_back(tok_str);
                ctx->tokenizer.token_to_id[tok_str] = (int32_t)v;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "voxcpm2: loaded %zu tokens from vocab blob\n", ctx->tokenizer.id_to_token.size());
            }
        }
    }

    ctx->tokenizer.audio_start_token = (int32_t)hp.audio_start_token;

    // Override vocab size from GGUF metadata
    if (hp.n_vocab > 0 && ctx->tokenizer.id_to_token.size() < (size_t)hp.n_vocab) {
        ctx->tokenizer.id_to_token.resize(hp.n_vocab);
    }

    // Bind TSLM (note: tslm.lm_head.weight is NOT present in the GGUF)
    W.tslm_token_embd = require(T, "tslm.token_embd.weight", "voxcpm2");
    W.tslm_output_norm = require(T, "tslm.output_norm.weight", "voxcpm2");
    W.tslm_rope_short = try_get(T, "tslm.rope_short_factors");
    W.tslm_rope_long = try_get(T, "tslm.rope_long_factors");

    W.tslm_layers.resize(hp.tslm_n_layers);
    for (uint32_t i = 0; i < hp.tslm_n_layers; i++) {
        auto& L = W.tslm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "tslm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // Bind RALM
    W.ralm_output_norm = require(T, "ralm.output_norm.weight", "voxcpm2");
    W.ralm_layers.resize(hp.ralm_n_layers);
    for (uint32_t i = 0; i < hp.ralm_n_layers; i++) {
        auto& L = W.ralm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "ralm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // FSQ
    W.fsq_in_proj_w = require(T, "fsq.in_proj.weight", "voxcpm2");
    W.fsq_in_proj_b = require(T, "fsq.in_proj.bias", "voxcpm2");
    W.fsq_out_proj_w = require(T, "fsq.out_proj.weight", "voxcpm2");
    W.fsq_out_proj_b = require(T, "fsq.out_proj.bias", "voxcpm2");

    // LocEnc
    W.locenc_cls_token = try_get(T, "locenc.cls_token");
    W.locenc_in_proj_w = require(T, "locenc.in_proj.weight", "voxcpm2");
    W.locenc_in_proj_b = require(T, "locenc.in_proj.bias", "voxcpm2");
    W.locenc_norm_w = try_get(T, "locenc.output_norm.weight");
    W.locenc_layers.resize(hp.locenc_n_layers);
    for (uint32_t i = 0; i < hp.locenc_n_layers; i++) {
        auto& L = W.locenc_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locenc.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.norm2_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // LocDiT
    W.locdit_in_proj_w = require(T, "locdit.in_proj.weight", "voxcpm2");
    W.locdit_in_proj_b = require(T, "locdit.in_proj.bias", "voxcpm2");
    W.locdit_cond_proj_w = require(T, "locdit.cond_proj.weight", "voxcpm2");
    W.locdit_cond_proj_b = require(T, "locdit.cond_proj.bias", "voxcpm2");
    W.locdit_time_mlp_0_w = require(T, "locdit.time_mlp.0.weight", "voxcpm2");
    W.locdit_time_mlp_0_b = require(T, "locdit.time_mlp.0.bias", "voxcpm2");
    W.locdit_time_mlp_1_w = require(T, "locdit.time_mlp.1.weight", "voxcpm2");
    W.locdit_time_mlp_1_b = require(T, "locdit.time_mlp.1.bias", "voxcpm2");
    W.locdit_dt_mlp_0_w = require(T, "locdit.dt_mlp.0.weight", "voxcpm2");
    W.locdit_dt_mlp_0_b = require(T, "locdit.dt_mlp.0.bias", "voxcpm2");
    W.locdit_dt_mlp_1_w = require(T, "locdit.dt_mlp.1.weight", "voxcpm2");
    W.locdit_dt_mlp_1_b = require(T, "locdit.dt_mlp.1.bias", "voxcpm2");
    W.locdit_norm_w = try_get(T, "locdit.output_norm.weight");
    W.locdit_out_proj_w = require(T, "locdit.out_proj.weight", "voxcpm2");
    W.locdit_out_proj_b = require(T, "locdit.out_proj.bias", "voxcpm2");
    W.locdit_layers.resize(hp.locdit_n_layers);
    for (uint32_t i = 0; i < hp.locdit_n_layers; i++) {
        auto& L = W.locdit_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locdit.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w = require(T, fn("attn_norm.weight"), "voxcpm2");
        L.norm2_w = require(T, fn("ffn_norm.weight"), "voxcpm2");
        L.attn_q_w = require(T, fn("attn_q.weight"), "voxcpm2");
        L.attn_k_w = require(T, fn("attn_k.weight"), "voxcpm2");
        L.attn_v_w = require(T, fn("attn_v.weight"), "voxcpm2");
        L.attn_o_w = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"), "voxcpm2");
        L.ffn_up_w = require(T, fn("ffn_up.weight"), "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"), "voxcpm2");
    }

    // Projection heads
    W.enc_to_lm_w = try_get(T, "proj.enc_to_lm.weight");
    W.enc_to_lm_b = try_get(T, "proj.enc_to_lm.bias");
    W.lm_to_dit_w = try_get(T, "proj.lm_to_dit.weight");
    W.lm_to_dit_b = try_get(T, "proj.lm_to_dit.bias");
    W.res_to_dit_w = try_get(T, "proj.res_to_dit.weight");
    W.res_to_dit_b = try_get(T, "proj.res_to_dit.bias");
    W.fusion_w = try_get(T, "proj.fusion.weight");
    W.fusion_b = try_get(T, "proj.fusion.bias");

    // Stop predictor
    W.stop_proj_w = try_get(T, "stop.proj.weight");
    W.stop_proj_b = try_get(T, "stop.proj.bias");
    W.stop_head_w = try_get(T, "stop.head.weight"); // [2048, 2], no bias

    // VAE decoder (graceful degradation when absent)
    // vae_decode() accesses ctx->tensors directly; these fields are kept for reference.
    W.vae_in_conv.w = try_get(T, "vae.dec.layer.0.weight_v");
    W.vae_in_conv.b = try_get(T, "vae.dec.layer.0.bias");
    W.vae_out_conv.w = try_get(T, "vae.dec.layer.9.weight_v");
    W.vae_out_conv.b = try_get(T, "vae.dec.layer.9.bias");
    W.vae_out_snake_a = try_get(T, "vae.dec.layer.8.alpha");
    W.vae_sr_cond_w = try_get(T, "vae.dec.sr_cond.2.scale_embed");
    W.vae_sr_cond_b = try_get(T, "vae.dec.sr_cond.2.bias_embed");

    // KV caches
    const int kv_max_ctx = 4096;
    ctx->tslm_kv.init((int)hp.tslm_n_layers, (int)hp.tslm_n_kv, (int)hp.tslm_head_dim, kv_max_ctx);
    ctx->ralm_kv.init((int)hp.ralm_n_layers, (int)hp.ralm_n_kv, (int)hp.ralm_head_dim, kv_max_ctx);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: loaded — TSLM %uL d=%u n_kv=%u, RALM %uL, LocEnc %uL d=%u, LocDiT %uL\n",
                hp.tslm_n_layers, hp.tslm_d_model, hp.tslm_n_kv, hp.ralm_n_layers, hp.locenc_n_layers,
                hp.locenc_d_model, hp.locdit_n_layers);
        fprintf(stderr, "voxcpm2: vocab %zu tokens, audio_start=%u\n", ctx->tokenizer.id_to_token.size(),
                hp.audio_start_token);
    }

    // Set ggml matmul thread count
    g_cpu_n_threads = 4; // TODO: fix ctx->params

    return true;
}

// ---------------------------------------------------------------------------
// Core synthesis pipeline
// ---------------------------------------------------------------------------

static float* vox_synthesize_internal(voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                      int ref_n_samples, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);

    // Seed RNG for CFM noise (use a fixed seed for reproducibility in testing,
    // or a time-based seed for varied output in production)
    mt19937_seed(ctx->rng, 42);

    double t0_total = vox_now_ms();
    (void)ref_samples;
    (void)ref_n_samples; // ref cloning: future work

    // 1. Tokenize
    std::vector<int32_t> token_ids = vox_tokenize(ctx->tokenizer, std::string(text));
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: tokenized '%s' -> %zu tokens\n", text, token_ids.size());
    }
    if (token_ids.empty()) {
        fprintf(stderr, "voxcpm2: empty token sequence\n");
        return nullptr;
    }

    // 2. TSLM prefill
    double t0_prefill = vox_now_ms();
    std::vector<float> tslm_hidden = tslm_prefill(ctx, token_ids, cpu_be);

    // Apply TSLM output norm
    {
        int d = (int)ctx->hp.tslm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(tslm_hidden.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                     ctx->hp.rms_norm_eps);
        tslm_hidden = normed;
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: TSLM prefill %.1f ms\n", vox_now_ms() - t0_prefill);
    }

    // 3. FSQ on TSLM hidden
    std::vector<float> fsq_out = fsq_forward(ctx, tslm_hidden.data(), cpu_be);

    // 4. fusion_concat_proj(cat(fsq_out, audio_feat_embed)) → RALM prefill
    // For zero-shot, audio_feat_embed is zeros (no reference audio)
    {
        int d = (int)ctx->hp.tslm_d_model;
        int in_dim = 2 * d; // 4096
        std::vector<float> cat_buf(in_dim, 0.0f);
        memcpy(cat_buf.data(), fsq_out.data(), d * sizeof(float));
        // Second half is zeros (no audio feat for text-only zero-shot)
        std::vector<float> fusion_out(d);
        matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim, fusion_out.data(),
                       d);
        fsq_out = fusion_out;
    }
    std::vector<float> ralm_hidden = ralm_prefill(ctx, fsq_out, cpu_be);
    // Apply RALM output norm
    {
        int d = (int)ctx->hp.ralm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(ralm_hidden.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), d,
                     ctx->hp.rms_norm_eps);
        ralm_hidden = normed;
    }

    // 5. Build mu [2048] for LocDiT: mu = cat(lm_to_dit(tslm), res_to_dit(ralm))
    // lm_to_dit:  [2048 → 1024] maps TSLM hidden to first half of mu
    // res_to_dit: [2048 → 1024] maps RALM hidden to second half of mu
    // mu.view(-1, 1024) = [2, 1024] = 2 conditioning tokens in the DiT sequence
    int d_lm = (int)ctx->hp.tslm_d_model;    // 2048
    int d_dit = (int)ctx->hp.locdit_d_model; // 1024 (half of mu)
    int d_ralm = (int)ctx->hp.ralm_d_model;  // 2048
    int d_mu = 2 * d_dit;                    // 2048 = full mu

    auto build_mu = [&](const std::vector<float>& lm_h, const std::vector<float>& ralm_h) -> std::vector<float> {
        std::vector<float> mu(d_mu, 0.0f);
        // First half: lm_to_dit projection [d_lm → d_dit]
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, lm_h.data(), d_lm, mu.data(),
                           d_dit);
        }
        // Second half: res_to_dit projection [d_ralm → d_dit]
        if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                           mu.data() + d_dit, d_dit);
        }
        return mu;
    };

    std::vector<float> mu = build_mu(tslm_hidden, ralm_hidden);
    // cond_raw for LocDiT: previous patch in feat_dim space [feat_dim * patch_frames]
    // Initialize to zeros for the first step
    int feat_dim_vae = 64;
    int P_frames = (int)ctx->hp.patch_frames;
    std::vector<float> prev_patch_raw(feat_dim_vae * P_frames, 0.0f);

    // 6. AR loop
    double t0_ar = vox_now_ms();
    std::vector<std::vector<float>> patches;
    float stop_thresh = 0.5f;
    int step = 0;

    while (step < ctx->max_len) {
        double t0_step = vox_now_ms();

        // a. CFM Euler solve (LocDiT)
        // patch is [feat_dim * patch_frames = 64 * 4 = 256] in latent space
        // Generate Gaussian noise as initial state for diffusion
        std::vector<float> noise(feat_dim_vae * P_frames);
        fill_gaussian_noise(noise.data(), (int)noise.size(), ctx->rng);
        std::vector<float> patch = cfm_euler_solve(ctx, mu.data(), prev_patch_raw.data(), ctx->inference_steps,
                                                   ctx->cfg_value, cpu_be, noise.data());
        patches.push_back(patch);

        // Update previous patch for next step
        prev_patch_raw = patch;
        // Resize to exactly feat_dim * P_frames if needed
        prev_patch_raw.resize(feat_dim_vae * P_frames, 0.0f);

        // b. LocEnc on predicted patch (uses in_proj to project feat_dim=64 → d_locenc=1024)
        // Pass the raw latent [feat_dim * patch_frames] — locenc_forward handles in_proj internally
        std::vector<float> patch_enc_in = patch;
        patch_enc_in.resize(feat_dim_vae * P_frames, 0.0f);
        std::vector<float> enc_out = locenc_forward(ctx, patch_enc_in.data(), cpu_be);

        // d. enc_to_lm_proj: locenc_d -> tslm_d
        std::vector<float> enc_lm(d_lm, 0.0f);
        if (ctx->weights.enc_to_lm_w && ctx->weights.enc_to_lm_b) {
            matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc_out.data(), d_dit,
                           enc_lm.data(), d_lm);
        } else {
            int copy_d = std::min(d_dit, d_lm);
            std::memcpy(enc_lm.data(), enc_out.data(), (size_t)copy_d * sizeof(float));
        }

        // e. TSLM step (single audio token position)
        {
            int tslm_pos = ctx->tslm_kv.n_past;
            std::vector<float> h = enc_lm;
            for (int l = 0; l < (int)ctx->hp.tslm_n_layers; l++) {
                tslm_layer_step(ctx, l, h.data(), tslm_pos, cpu_be);
            }
            ctx->tslm_kv.n_past++;

            // Output norm
            std::vector<float> normed(d_lm);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d_lm,
                         ctx->hp.rms_norm_eps);
            tslm_hidden = normed;
        }

        // c. Stop check (after TSLM step, using updated lm_hidden)
        {
            float sp = stop_score(ctx, tslm_hidden.data(), cpu_be);
            if (ctx->verbosity >= 2) {
                fprintf(stderr, "voxcpm2: step %d stop=%.3f (%.1f ms)\n", step, sp, vox_now_ms() - t0_step);
            }
            if (sp > stop_thresh && step > 0) {
                if (ctx->verbosity >= 1) {
                    fprintf(stderr, "voxcpm2: stopped at step %d (stop=%.3f)\n", step + 1, sp);
                }
                break;
            }
        }

        // f. FSQ
        fsq_out = fsq_forward(ctx, tslm_hidden.data(), cpu_be);

        // g. proj.fusion(cat(lm_hidden, enc_lm)) [4096 -> 2048]
        std::vector<float> fusion_in((size_t)(d_lm + d_lm));
        std::memcpy(fusion_in.data(), tslm_hidden.data(), (size_t)d_lm * sizeof(float));
        std::memcpy(fusion_in.data() + d_lm, enc_lm.data(), (size_t)d_lm * sizeof(float));

        std::vector<float> fusion_out(d_ralm, 0.0f);
        if (ctx->weights.fusion_w && ctx->weights.fusion_b) {
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, fusion_in.data(), 2 * d_lm,
                           fusion_out.data(), d_ralm);
        } else {
            // Fallback: pass fsq_out directly
            fusion_out = fsq_out;
        }

        // h. RALM step
        {
            std::vector<float> h = fusion_out;
            int ralm_pos = ctx->ralm_kv.n_past;
            (void)ralm_pos;
            for (int l = 0; l < (int)ctx->hp.ralm_n_layers; l++) {
                ralm_layer_step(ctx, l, h.data(), cpu_be);
            }
            ctx->ralm_kv.n_past++;

            std::vector<float> normed(d_ralm);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), d_ralm,
                         ctx->hp.rms_norm_eps);
            ralm_hidden = normed;
        }

        // Update mu conditioning for next step
        mu = build_mu(tslm_hidden, ralm_hidden);
        step++;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: AR loop %d steps, %.1f ms\n", step, vox_now_ms() - t0_ar);
    }

    // 7. VAE decode
    double t0_vae = vox_now_ms();
    std::vector<float> pcm = vae_decode(ctx, patches, cpu_be);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: VAE decode %.1f ms -> %zu samples @48kHz\n", vox_now_ms() - t0_vae, pcm.size());
        fprintf(stderr, "voxcpm2: total %.1f ms\n", vox_now_ms() - t0_total);
    }

    if (pcm.empty()) {
        fprintf(stderr, "voxcpm2: empty audio output\n");
        return nullptr;
    }

    *out_n_samples = (int)pcm.size();
    float* result = (float*)std::malloc(pcm.size() * sizeof(float));
    if (!result)
        return nullptr;
    std::memcpy(result, pcm.data(), pcm.size() * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

struct voxcpm2_context_params voxcpm2_context_default_params(void) {
    struct voxcpm2_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = true;
    p.inference_steps = 10;
    p.cfg_value = 2.0f;
    p.max_len = 2000;
    return p;
}

struct voxcpm2_context* voxcpm2_init_from_file(const char* path_model, struct voxcpm2_context_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new voxcpm2_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->flash_attn = params.flash_attn;
    ctx->inference_steps = params.inference_steps > 0 ? params.inference_steps : 10;
    ctx->cfg_value = params.cfg_value > 0.0f ? params.cfg_value : 2.0f;
    ctx->max_len = params.max_len > 0 ? params.max_len : 2000;

    if (!vox_load_weights(ctx, path_model)) {
        fprintf(stderr, "voxcpm2: failed to load '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void voxcpm2_free(struct voxcpm2_context* ctx) {
    if (!ctx)
        return;
    if (ctx->weight_buf) {
        ggml_backend_buffer_free(ctx->weight_buf);
        ctx->weight_buf = nullptr;
    }
    if (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }
    delete ctx;
}

void voxcpm2_set_n_threads(struct voxcpm2_context* ctx, int n_threads) {
    if (ctx)
        ctx->n_threads = n_threads > 0 ? n_threads : 1;
}

float* voxcpm2_synthesize(struct voxcpm2_context* ctx, const char* text, int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, nullptr, 0, out_n_samples);
}

float* voxcpm2_synthesize_clone(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                int ref_n_samples, int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n_samples);
}

struct voxcpm2_stream* voxcpm2_stream_open(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                           int ref_n_samples) {
    if (!ctx || !text)
        return nullptr;
    auto* s = new voxcpm2_stream();
    s->ctx = ctx;
    s->done = false;
    s->chunk_offset = 0;

    int n_samples = 0;
    float* pcm = vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, &n_samples);
    if (pcm && n_samples > 0) {
        s->all_pcm.assign(pcm, pcm + n_samples);
        std::free(pcm);
    } else {
        s->done = true;
    }
    return s;
}

const float* voxcpm2_stream_next(struct voxcpm2_stream* stream, int* out_n_samples) {
    if (!stream || stream->done || stream->all_pcm.empty()) {
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }

    const int chunk_size = 3840; // one VAE decode unit at 48kHz (~12.5fps)
    int offset = stream->chunk_offset;
    int remaining = (int)stream->all_pcm.size() - offset;
    if (remaining <= 0) {
        stream->done = true;
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }

    int n = std::min(remaining, chunk_size);
    stream->chunk_buf.assign(stream->all_pcm.begin() + offset, stream->all_pcm.begin() + offset + n);
    stream->chunk_offset += n;
    if (stream->chunk_offset >= (int)stream->all_pcm.size()) {
        stream->done = true;
    }

    if (out_n_samples)
        *out_n_samples = n;
    return stream->chunk_buf.data();
}

void voxcpm2_stream_close(struct voxcpm2_stream* stream) {
    delete stream;
}

void voxcpm2_pcm_free(float* pcm) {
    std::free(pcm);
}

float* voxcpm2_extract_stage(struct voxcpm2_context* ctx, const char* text, const float* ref_samples, int ref_n_samples,
                             const char* stage_name, int* out_n) {
    if (!ctx || !text || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);

    std::string stage(stage_name);
    std::vector<int32_t> token_ids = vox_tokenize(ctx->tokenizer, std::string(text));

    if (stage == "text_input_ids") {
        *out_n = (int)token_ids.size();
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        for (int i = 0; i < *out_n; i++)
            out[i] = (float)token_ids[i];
        return out;
    }

    if (stage == "tslm_prefill_out" || stage == "tslm_layer_0_out" || stage == "tslm_layer_27_out") {
        // Instrumented prefill: capture per-position + per-layer outputs
        int d = (int)ctx->hp.tslm_d_model;
        int n_layers = (int)ctx->hp.tslm_n_layers;
        const int N_CAP = 8; // first 8 positions (matching Python reference)

        std::vector<float> all_pos;
        std::vector<float> layer0_buf, layer_last_buf;

        tslm_prefill_hooks hooks;
        hooks.max_capture_positions = N_CAP;
        hooks.all_positions = &all_pos;
        hooks.layer0_capture = 0;
        hooks.layer0_out = &layer0_buf;
        hooks.layer_last_capture = n_layers - 1;
        hooks.layer_last_out = &layer_last_buf;

        tslm_prefill_ex(ctx, token_ids, cpu_be, hooks);

        if (stage == "tslm_prefill_out") {
            // Apply output norm to each captured position
            int N = (int)(all_pos.size() / d);
            std::vector<float> normed((size_t)N * d);
            for (int i = 0; i < N; i++) {
                rms_norm_cpu(all_pos.data() + (size_t)i * d, tensor_data_f32(ctx->weights.tslm_output_norm),
                             normed.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
            }
            *out_n = N * d;
            float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
            std::memcpy(out, normed.data(), (size_t)*out_n * sizeof(float));
            return out;
        } else if (stage == "tslm_layer_0_out") {
            *out_n = (int)layer0_buf.size();
            if (*out_n == 0)
                return nullptr;
            float* out = (float*)std::malloc(layer0_buf.size() * sizeof(float));
            std::memcpy(out, layer0_buf.data(), layer0_buf.size() * sizeof(float));
            return out;
        } else { // tslm_layer_27_out
            *out_n = (int)layer_last_buf.size();
            if (*out_n == 0)
                return nullptr;
            float* out = (float*)std::malloc(layer_last_buf.size() * sizeof(float));
            std::memcpy(out, layer_last_buf.data(), layer_last_buf.size() * sizeof(float));
            return out;
        }
    }

    if (stage == "ralm_prefill_out") {
        // Multi-position RALM prefill matching Python:
        // 1. TSLM prefill → normed output for all positions
        // 2. FSQ masking (text_mask selects raw TSLM out for text tokens)
        // 3. fusion_concat_proj(cat(enc_out, audio_mask * feat_embed))
        //    For zero-shot: audio_mask=0, so second half is zeros
        // 4. RALM prefill (causal, all T positions)
        // 5. Capture first 8 positions
        const int N_CAP = 8;
        int d = (int)ctx->hp.tslm_d_model;

        // Run instrumented TSLM to get first N_CAP positions
        std::vector<float> all_pos;
        tslm_prefill_hooks hooks;
        hooks.max_capture_positions = N_CAP;
        hooks.all_positions = &all_pos;
        tslm_prefill_ex(ctx, token_ids, cpu_be, hooks);

        int N = (int)(all_pos.size() / d);
        // Apply output norm to each position
        std::vector<float> normed((size_t)N * d);
        for (int i = 0; i < N; i++) {
            rms_norm_cpu(all_pos.data() + (size_t)i * d, tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
        }

        // For zero-shot, all positions are text (text_mask=1, audio_mask=0).
        // FSQ masking: text positions use raw TSLM output (no FSQ).
        // So normed output IS the FSQ-masked enc_outputs for text tokens.

        // fusion_concat_proj: input is cat(enc_outputs_t, zeros) for each position
        int in_dim = 2 * d; // 4096
        std::vector<float> ralm_input((size_t)N * d);
        for (int i = 0; i < N; i++) {
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), normed.data() + (size_t)i * d, (size_t)d * sizeof(float));
            // Second half stays zero (no audio feat for text-only zero-shot)
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           ralm_input.data() + (size_t)i * d, d);
        }

        // Run multi-position RALM prefill (returns pre-norm hidden states)
        std::vector<float> ralm_out = ralm_prefill_multi(ctx, ralm_input.data(), N, cpu_be);

        // Apply RALM output norm (Python's model.residual_lm() includes final norm)
        for (int i = 0; i < N; i++) {
            rms_norm_cpu(ralm_out.data() + (size_t)i * d, tensor_data_f32(ctx->weights.ralm_output_norm),
                         ralm_out.data() + (size_t)i * d, d, ctx->hp.rms_norm_eps);
        }

        *out_n = N * d;
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        std::memcpy(out, ralm_out.data(), (size_t)*out_n * sizeof(float));
        return out;
    }

    if (stage == "cfm_step0_result") {
        // Full pipeline: TSLM → norm → FSQ → fusion → RALM → norm → projections → CFM
        // Matches Python: dit_hidden = cat(lm_to_dit(lm_hidden), res_to_dit(residual_hidden))
        int d = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        int d_ralm = (int)ctx->hp.ralm_d_model;

        // 1. TSLM prefill → last token hidden → norm
        std::vector<float> tslm_h = tslm_prefill(ctx, token_ids, cpu_be);
        {
            std::vector<float> normed(d);
            rms_norm_cpu(tslm_h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                         ctx->hp.rms_norm_eps);
            tslm_h = normed;
        }

        // 2. FSQ masking: for zero-shot last token (text_mask=1), no FSQ applied
        // 3. fusion_concat_proj(cat(tslm_h, zeros)) for single-token RALM input
        int in_dim = 2 * d;
        std::vector<float> cat_buf(in_dim, 0.0f);
        std::memcpy(cat_buf.data(), tslm_h.data(), (size_t)d * sizeof(float));
        std::vector<float> ralm_input(d);
        matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim, ralm_input.data(),
                       d);

        // 4. RALM prefill (single token)
        std::vector<float> ralm_h = ralm_prefill(ctx, ralm_input, cpu_be);
        {
            std::vector<float> normed(d_ralm);
            rms_norm_cpu(ralm_h.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), d_ralm,
                         ctx->hp.rms_norm_eps);
            ralm_h = normed;
        }

        // 5. Build mu = cat(lm_to_dit(tslm_h), res_to_dit(ralm_h))
        std::vector<float> mu(2 * d_dit, 0.0f);
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, tslm_h.data(), d, mu.data(),
                           d_dit);
        }
        if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                           mu.data() + d_dit, d_dit);
        }

        // 6. CFM Euler solve (10 steps, seeded noise)
        int P_fr = (int)ctx->hp.patch_frames;
        int state_size = 64 * P_fr;
        std::vector<float> zero_cond(state_size, 0.0f);
        std::vector<float> noise(state_size);
        fill_gaussian_noise(noise.data(), state_size, 42);
        std::vector<float> patch =
            cfm_euler_solve(ctx, mu.data(), zero_cond.data(), 10, ctx->cfg_value, cpu_be, noise.data());
        *out_n = (int)patch.size();
        float* out = (float*)std::malloc(patch.size() * sizeof(float));
        std::memcpy(out, patch.data(), patch.size() * sizeof(float));
        return out;
    }

    if (stage == "locenc_out") {
        // Python: feat_embed = model.feat_encoder(audio_feat) → [B, T, d_enc]
        // For zero-shot: audio_feat is zeros [T, P, 64]. LocEnc processes each
        // patch independently. Output first 8 positions [8, d_enc].
        int d_enc = (int)ctx->hp.locenc_d_model; // 1024
        int P_fr = (int)ctx->hp.patch_frames;
        const int N_CAP = 8;
        std::vector<float> zero_patch(64 * P_fr, 0.0f);

        // All patches are zero → all outputs are identical. Compute once, replicate.
        std::vector<float> one_out = locenc_forward(ctx, zero_patch.data(), cpu_be);
        int total = N_CAP * d_enc;
        *out_n = total;
        float* out = (float*)std::malloc((size_t)total * sizeof(float));
        for (int i = 0; i < N_CAP; i++) {
            std::memcpy(out + (size_t)i * d_enc, one_out.data(), (size_t)d_enc * sizeof(float));
        }
        return out;
    }

    if (stage == "locenc_in") {
        // LocEnc input: audio_feat[0, :8] = [8, patch_frames, feat_dim] = [8, 4, 64]
        // For zero-shot synthesis, audio features are zeros (no reference audio).
        int P_fr = (int)ctx->hp.patch_frames; // 4
        int feat_dim = 64;
        int N_CAP = 8;
        int total = N_CAP * P_fr * feat_dim; // 8 * 4 * 64 = 2048
        *out_n = total;
        float* out = (float*)std::calloc(total, sizeof(float));
        return out;
    }

    if (stage == "enc_to_lm") {
        // Python: feat_embed = enc_to_lm_proj(feat_encoder(audio_feat)) → [B, T, d_lm]
        // Capture first 8 positions. For zero-shot, all patches are zeros → identical.
        int d_enc = (int)ctx->hp.locenc_d_model; // 1024
        int d_lm = (int)ctx->hp.tslm_d_model;    // 2048
        int P_fr = (int)ctx->hp.patch_frames;
        const int N_CAP = 8;
        std::vector<float> zero_patch(64 * P_fr, 0.0f);
        std::vector<float> enc_out = locenc_forward(ctx, zero_patch.data(), cpu_be);
        if (ctx->weights.enc_to_lm_w && ctx->weights.enc_to_lm_b) {
            std::vector<float> proj_one(d_lm);
            matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b, enc_out.data(), d_enc,
                           proj_one.data(), d_lm);
            int total = N_CAP * d_lm;
            *out_n = total;
            float* out = (float*)std::malloc((size_t)total * sizeof(float));
            for (int i = 0; i < N_CAP; i++) {
                std::memcpy(out + (size_t)i * d_lm, proj_one.data(), (size_t)d_lm * sizeof(float));
            }
            return out;
        }
        return nullptr;
    }

    if (stage == "lm_to_dit_hidden" || stage == "res_to_dit_hidden") {
        // Run full pipeline to get projections: TSLM → FSQ → RALM → project
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                         ctx->hp.rms_norm_eps);
            h = normed;
        }
        std::vector<float> tslm_out = h; // save for lm_to_dit

        std::vector<float> fsq_out = fsq_forward(ctx, h.data(), cpu_be);
        // fusion_concat_proj(cat(fsq_out, zeros))
        {
            int in_dim = 2 * d;
            std::vector<float> cat_buf(in_dim, 0.0f);
            std::memcpy(cat_buf.data(), fsq_out.data(), d * sizeof(float));
            std::vector<float> fusion_out(d);
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b, cat_buf.data(), in_dim,
                           fusion_out.data(), d);
            fsq_out = fusion_out;
        }
        std::vector<float> ralm_h = ralm_prefill(ctx, fsq_out, cpu_be);
        {
            int dr = (int)ctx->hp.ralm_d_model;
            std::vector<float> normed(dr);
            rms_norm_cpu(ralm_h.data(), tensor_data_f32(ctx->weights.ralm_output_norm), normed.data(), dr,
                         ctx->hp.rms_norm_eps);
            ralm_h = normed;
        }

        if (stage == "lm_to_dit_hidden") {
            if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
                std::vector<float> proj(d_dit);
                matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b, tslm_out.data(), d,
                               proj.data(), d_dit);
                *out_n = d_dit;
                float* out = (float*)std::malloc((size_t)d_dit * sizeof(float));
                std::memcpy(out, proj.data(), (size_t)d_dit * sizeof(float));
                return out;
            }
        } else { // res_to_dit_hidden
            if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
                int d_ralm = (int)ctx->hp.ralm_d_model;
                std::vector<float> proj(d_dit);
                matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b, ralm_h.data(), d_ralm,
                               proj.data(), d_dit);
                *out_n = d_dit;
                float* out = (float*)std::malloc((size_t)d_dit * sizeof(float));
                std::memcpy(out, proj.data(), (size_t)d_dit * sizeof(float));
                return out;
            }
        }
        return nullptr;
    }

    if (stage == "stop_logits_step0") {
        // Run TSLM prefill, then compute stop logits from the last hidden state
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm), normed.data(), d,
                         ctx->hp.rms_norm_eps);
            h = normed;
        }
        // stop_proj: d_lm → d_lm with bias, then SiLU, then stop_head: d_lm → 2
        const vox_weights& W = ctx->weights;
        if (!W.stop_proj_w || !W.stop_proj_b || !W.stop_head_w)
            return nullptr;

        std::vector<float> proj(d);
        matmul_mv_bias(cpu_be, W.stop_proj_w, W.stop_proj_b, h.data(), d, proj.data(), d);
        for (int i = 0; i < d; i++) {
            float v = proj[i];
            proj[i] = v / (1.0f + std::exp(-v)); // SiLU
        }
        float logits[2] = {0.0f, 0.0f};
        matmul_mv(cpu_be, W.stop_head_w, proj.data(), d, logits, 2);

        *out_n = 2;
        float* out = (float*)std::malloc(2 * sizeof(float));
        out[0] = logits[0];
        out[1] = logits[1];
        return out;
    }

    if (stage == "cfm_step0_z") {
        // CFM initial noise. The Python reference uses torch.randn with
        // dtype=bfloat16 and seed=42 via a separate Generator. The BF16 randn
        // path in PyTorch uses a different bit-extraction than F32 randn, making
        // exact matching non-trivial. For now, output seeded F32 Gaussian noise
        // (matching the MT19937 F32 path used in vibevoice/chatterbox).
        // TODO: implement PyTorch's BF16 randn path for exact match.
        int P_fr = (int)ctx->hp.patch_frames; // 4
        int in_ch = 64;
        int state_size = in_ch * P_fr; // 256
        *out_n = state_size;
        float* out = (float*)std::malloc((size_t)state_size * sizeof(float));
        fill_gaussian_noise(out, state_size, 42);
        return out;
    }

    if (stage == "decoded_audio") {
        return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n);
    }

    fprintf(stderr, "voxcpm2_extract_stage: unknown stage '%s'\n", stage_name);
    return nullptr;
}

} // extern "C"
