// kugelaudio.cpp — KugelAudio-0-Open TTS runtime.
//
// Architecture overview (see kugelaudio.h for full description):
//   Qwen2.5-7B LM → DiT diffusion head → acoustic VAE decoder → 24 kHz PCM.
//
// This implementation reuses core/ primitives:
//   core_attn  — Qwen2.5 GQA attention with KV cache
//   core_ffn   — SwiGLU FFN (LM + diffusion head)
//
// DPM-Solver++ SDE with v-prediction is implemented inline following the
// vibevoice.cpp pattern, extended for the SDE noise injection variant.

#include "kugelaudio.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/conv.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/torch_rng.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// ── Constants ───────────────────────────────────────────────────────────────

static constexpr int KUGELAUDIO_SPEECH_START_ID    = 151652;
static constexpr int KUGELAUDIO_SPEECH_END_ID      = 151653;
static constexpr int KUGELAUDIO_SPEECH_DIFFUSION_ID = 151654;
static constexpr int KUGELAUDIO_EOS_TOKEN_ID       = 151643;

// ── DPM-Solver++ SDE schedule ──────────────────────────────────────────────

struct dpm_sde_schedule {
    int num_train_steps;
    int num_inference_steps;
    std::vector<float> alphas_cumprod;
    std::vector<float> sigmas;    // sqrt((1-ac)/ac) — diffusers sigma convention
    std::vector<int>   timesteps;
};

static dpm_sde_schedule make_dpm_sde_schedule(int num_inference_steps, int num_train_steps = 1000) {
    dpm_sde_schedule s;
    s.num_train_steps = num_train_steps;
    s.num_inference_steps = num_inference_steps;

    // Cosine beta schedule
    float offset = 0.008f;
    auto alpha_bar_fn = [&](float t) -> float {
        float frac = (t + offset) / (1.0f + offset);
        float val = cosf(frac * (float)M_PI * 0.5f);
        return val * val;
    };

    s.alphas_cumprod.resize(num_train_steps);
    s.sigmas.resize(num_train_steps);
    float a_prod = 1.0f;
    for (int i = 0; i < num_train_steps; i++) {
        float t1 = (float)i / (float)num_train_steps;
        float t2 = (float)(i + 1) / (float)num_train_steps;
        float beta = std::min(1.0f - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999f);
        a_prod *= (1.0f - beta);
        s.alphas_cumprod[i] = a_prod;
        s.sigmas[i] = sqrtf((1.0f - a_prod) / a_prod);
    }

    // Linspace timesteps
    s.timesteps.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; i++) {
        float val = (float)(num_train_steps - 1) * (float)(num_inference_steps - i)
                    / (float)num_inference_steps;
        s.timesteps[i] = (int)roundf(val);
    }

    return s;
}

// sigma → (alpha_t, sigma_t) conversion
static void sigma_to_alpha_sigma(float sigma, float& out_alpha, float& out_sigma) {
    out_alpha = 1.0f / sqrtf(sigma * sigma + 1.0f);
    out_sigma = sigma * out_alpha;
}

// Convert v-prediction to x0
static void v_to_x0(const float* x_t, const float* v, float* x0, int n,
                    float alpha_t, float sigma_t) {
    for (int i = 0; i < n; i++)
        x0[i] = alpha_t * x_t[i] - sigma_t * v[i];
}

// DPM-Solver++ SDE 1st order
static void dpm_sde_first_order(const dpm_sde_schedule& sched, int step_idx,
                                float* x, const float* x0_pred, const float* noise, int n) {
    float sigma_cur = sched.sigmas[sched.timesteps[step_idx]];
    int t_next = (step_idx + 1 < sched.num_inference_steps)
                     ? sched.timesteps[step_idx + 1] : -1;
    float sigma_next = (t_next >= 0) ? sched.sigmas[t_next] : 0.0f;

    float alpha_cur, sig_cur, alpha_next, sig_next;
    sigma_to_alpha_sigma(sigma_cur, alpha_cur, sig_cur);
    if (t_next >= 0) sigma_to_alpha_sigma(sigma_next, alpha_next, sig_next);
    else { alpha_next = 1.0f; sig_next = 0.0f; }

    float lambda_cur = logf(alpha_cur / sig_cur);
    float lambda_next = (t_next >= 0) ? logf(alpha_next / sig_next) : 20.0f;
    float h = lambda_next - lambda_cur;

    float exp_neg_h = expf(-h);
    float coeff_x = (sig_next / sig_cur) * exp_neg_h;
    float coeff_x0 = alpha_next * (1.0f - expf(-2.0f * h));
    float coeff_noise = sig_next * sqrtf(std::max(0.0f, 1.0f - expf(-2.0f * h)));

    for (int i = 0; i < n; i++)
        x[i] = coeff_x * x[i] + coeff_x0 * x0_pred[i] + coeff_noise * noise[i];
}

// DPM-Solver++ SDE 2nd order midpoint
static void dpm_sde_second_order(const dpm_sde_schedule& sched, int step_idx,
                                 float* x, const float* x0_cur, const float* x0_prev,
                                 int prev_step_idx, const float* noise, int n) {
    float sigma_cur = sched.sigmas[sched.timesteps[step_idx]];
    int t_next = (step_idx + 1 < sched.num_inference_steps)
                     ? sched.timesteps[step_idx + 1] : -1;
    float sigma_next = (t_next >= 0) ? sched.sigmas[t_next] : 0.0f;
    float sigma_prev = sched.sigmas[sched.timesteps[prev_step_idx]];

    float alpha_cur, sig_cur, alpha_next, sig_next, alpha_prev, sig_prev;
    sigma_to_alpha_sigma(sigma_cur, alpha_cur, sig_cur);
    if (t_next >= 0) sigma_to_alpha_sigma(sigma_next, alpha_next, sig_next);
    else { alpha_next = 1.0f; sig_next = 0.0f; }
    sigma_to_alpha_sigma(sigma_prev, alpha_prev, sig_prev);

    float lambda_cur = logf(alpha_cur / sig_cur);
    float lambda_next = (t_next >= 0) ? logf(alpha_next / sig_next) : 20.0f;
    float lambda_prev = logf(alpha_prev / sig_prev);

    float h = lambda_next - lambda_cur;
    float h_0 = lambda_cur - lambda_prev;
    float r = h_0 / h;

    float exp_neg_h = expf(-h);
    float coeff_x = (sig_next / sig_cur) * exp_neg_h;
    float eh_term = 1.0f - expf(-2.0f * h);
    float coeff_noise = sig_next * sqrtf(std::max(0.0f, eh_term));

    for (int i = 0; i < n; i++) {
        float D0 = x0_cur[i];
        float D1 = (1.0f / r) * (x0_cur[i] - x0_prev[i]);
        x[i] = coeff_x * x[i]
             + alpha_next * eh_term * D0
             + 0.5f * alpha_next * eh_term * D1
             + coeff_noise * noise[i];
    }
}

// ── Sinusoidal timestep embedding ──────────────────────────────────────────

static void compute_sinusoidal_embed(float t, float* out, int dim = 256) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

// ── Model / hparams ────────────────────────────────────────────────────────

struct kugelaudio_hparams {
    int d_lm = 3584;
    int n_lm_layers = 28;
    int n_heads = 28;
    int n_kv_heads = 4;
    int d_ffn = 18944;
    int vocab_size = 152064;
    int head_dim = 128;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    int vae_dim_acoustic = 64;
    int vae_dim_semantic = 128;

    // Diffusion head
    int head_layers = 4;
    float head_ffn_ratio = 3.0f;
    int latent_size = 64;
    int ddpm_num_steps = 1000;
    int ddpm_inference_steps = 20;
    float diff_norm_eps = 1e-5f;

    // Decoder
    int n_decoder_stages = 7;
    int dec_n_filters = 32;
    int total_upsample = 3200;
    float at_norm_eps = 1e-5f;
    std::vector<int> decoder_ratios;
    std::vector<int> decoder_depths;
    std::vector<int> encoder_ratios;
    std::vector<int> encoder_depths;
    int has_encoders = 0;

    // Scaling
    float speech_scaling_factor = 1.0f;
    float speech_bias_factor = 0.0f;
};

struct kugelaudio_model {
    kugelaudio_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<std::string> vocab;
};

// ── Context ────────────────────────────────────────────────────────────────

struct kugelaudio_context {
    kugelaudio_model model;
    kugelaudio_context_params params;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;

    // KV cache for positive (main) LM path
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // KV cache for negative (CFG) path
    ggml_context* kv_neg_ctx = nullptr;
    ggml_backend_buffer_t kv_neg_buf = nullptr;
    ggml_tensor* kv_neg_k = nullptr;
    ggml_tensor* kv_neg_v = nullptr;

    std::vector<uint8_t> compute_meta;

    // Voice cache
    std::vector<float> voice_acoustic_mean; // [n_voice_frames * vae_dim]
    int n_voice_frames = 0;

    // RNG for diffusion noise
    crispasr::core::mt19937_state rng;
};

// ── Tensor lookup helpers ──────────────────────────────────────────────────

static ggml_tensor* T(kugelaudio_context* ctx, const char* name) {
    return core_gguf::require(ctx->model.tensors, name, "kugelaudio");
}

static ggml_tensor* Topt(kugelaudio_context* ctx, const char* name) {
    return core_gguf::try_get(ctx->model.tensors, name);
}

// ── Public API ─────────────────────────────────────────────────────────────

extern "C" struct kugelaudio_context_params kugelaudio_context_default_params(void) {
    kugelaudio_context_params p;
    p.n_threads = 4;
    p.max_new_tokens = 2048;
    p.verbosity = 1;
    p.use_gpu = true;
    p.tts_steps = 20;
    p.cfg_scale = 3.0f;
    p.seed = 0;
    p.flash_attn = true;
    return p;
}

extern "C" void kugelaudio_set_tts_steps(struct kugelaudio_context* ctx, int steps) {
    if (!ctx) return;
    ctx->params.tts_steps = std::clamp(steps, 4, 100);
}

extern "C" void kugelaudio_set_cfg_scale(struct kugelaudio_context* ctx, float scale) {
    if (ctx) ctx->params.cfg_scale = scale;
}

extern "C" void kugelaudio_set_seed(struct kugelaudio_context* ctx, uint32_t seed) {
    if (!ctx) return;
    ctx->params.seed = seed;
    if (seed != 0)
        crispasr::core::mt19937_seed(ctx->rng, seed);
}

// ── Init ───────────────────────────────────────────────────────────────────

extern "C" struct kugelaudio_context* kugelaudio_init_from_file(
    const char* path_model, struct kugelaudio_context_params params) {

    auto* ctx = new kugelaudio_context();
    ctx->params = params;
    auto& m = ctx->model;
    auto& hp = m.hp;

    // ── Pass 1: metadata ────────────────────────────────────────────────
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    hp.d_lm = core_gguf::kv_u32(gctx, "kugelaudio.d_lm", 3584);
    hp.n_lm_layers = core_gguf::kv_u32(gctx, "kugelaudio.n_lm_layers", 28);
    hp.n_heads = core_gguf::kv_u32(gctx, "kugelaudio.n_heads", 28);
    hp.n_kv_heads = core_gguf::kv_u32(gctx, "kugelaudio.n_kv_heads", 4);
    hp.d_ffn = core_gguf::kv_u32(gctx, "kugelaudio.d_ffn", 18944);
    hp.vocab_size = core_gguf::kv_u32(gctx, "kugelaudio.vocab_size", 152064);
    hp.head_dim = core_gguf::kv_u32(gctx, "kugelaudio.head_dim", 128);
    hp.rope_theta = core_gguf::kv_f32(gctx, "kugelaudio.rope_theta", 1000000.0f);
    hp.rms_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.rms_norm_eps", 1e-6f);
    hp.vae_dim_acoustic = core_gguf::kv_u32(gctx, "kugelaudio.vae_dim_acoustic", 64);
    hp.vae_dim_semantic = core_gguf::kv_u32(gctx, "kugelaudio.vae_dim_semantic", 128);
    hp.head_layers = core_gguf::kv_u32(gctx, "kugelaudio.head_layers", 4);
    hp.head_ffn_ratio = core_gguf::kv_f32(gctx, "kugelaudio.head_ffn_ratio", 3.0f);
    hp.latent_size = core_gguf::kv_u32(gctx, "kugelaudio.latent_size", 64);
    hp.ddpm_num_steps = core_gguf::kv_u32(gctx, "kugelaudio.ddpm_num_steps", 1000);
    hp.ddpm_inference_steps = core_gguf::kv_u32(gctx, "kugelaudio.ddpm_inference_steps", 20);
    hp.diff_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.diff_norm_eps", 1e-5f);
    hp.n_decoder_stages = core_gguf::kv_u32(gctx, "kugelaudio.n_decoder_stages", 7);
    hp.dec_n_filters = core_gguf::kv_u32(gctx, "kugelaudio.dec_n_filters", 32);
    hp.total_upsample = core_gguf::kv_u32(gctx, "kugelaudio.total_upsample", 3200);
    hp.at_norm_eps = core_gguf::kv_f32(gctx, "kugelaudio.at_norm_eps", 1e-5f);
    hp.has_encoders = core_gguf::kv_u32(gctx, "kugelaudio.has_encoders", 0);

    // Read arrays
    auto read_int_array = [&](const char* key, std::vector<int>& out, const std::vector<int>& defaults) {
        int k = gguf_find_key(gctx, key);
        if (k >= 0) {
            int n = gguf_get_arr_n(gctx, k);
            out.resize(n);
            for (int i = 0; i < n; i++)
                out[i] = ((const int32_t*)gguf_get_arr_data(gctx, k))[i];
        } else {
            out = defaults;
        }
    };

    read_int_array("kugelaudio.decoder_ratios", hp.decoder_ratios, {8, 5, 5, 4, 2, 2});
    read_int_array("kugelaudio.decoder_depths", hp.decoder_depths, {8, 3, 3, 3, 3, 3, 3});
    read_int_array("kugelaudio.encoder_ratios", hp.encoder_ratios, {8, 5, 5, 4, 2, 2});
    read_int_array("kugelaudio.encoder_depths", hp.encoder_depths, {3, 3, 3, 3, 3, 3, 8});

    // Load vocabulary
    const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        m.vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s) m.vocab[i] = s;
        }
    }

    gguf_free(gctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: d_lm=%d, layers=%d, heads=%d/%d, ffn=%d, vocab=%d\n",
                hp.d_lm, hp.n_lm_layers, hp.n_heads, hp.n_kv_heads, hp.d_ffn, hp.vocab_size);
        fprintf(stderr, "kugelaudio: vae_dim=%d, latent=%d, diff_steps=%d, head_layers=%d\n",
                hp.vae_dim_acoustic, hp.latent_size, hp.ddpm_inference_steps, hp.head_layers);
        fprintf(stderr, "kugelaudio: decoder stages=%d, upsample=%dx\n",
                hp.n_decoder_stages, hp.total_upsample);
    }

    // ── Backend init ────────────────────────────────────────────────────
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, std::max(1, params.n_threads));
    if (ctx->backend && ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, std::max(1, params.n_threads));
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    // ── Pass 2: weight loading ──────────────────────────────────────────
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "kugelaudio", wl)) {
        ggml_backend_free(ctx->backend);
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    ctx->buf_cpu = wl.buf_cpu;
    m.tensors = std::move(wl.tensors);

    // Create scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 65536, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false));

    // Read scaling factors from tensor data
    auto* sf = Topt(ctx, "model.speech_scaling_factor");
    auto* bf = Topt(ctx, "model.speech_bias_factor");
    if (sf && bf) {
        // These are scalar tensors; read their values
        ggml_backend_tensor_get(sf, &hp.speech_scaling_factor, 0, sizeof(float));
        ggml_backend_tensor_get(bf, &hp.speech_bias_factor, 0, sizeof(float));
    }

    // Init RNG
    crispasr::core::mt19937_seed(ctx->rng, params.seed != 0 ? params.seed : (uint32_t)time(nullptr));

    if (params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: loaded %zu tensors (backend: %s)\n",
                m.tensors.size(), ggml_backend_name(ctx->backend));
        fprintf(stderr, "kugelaudio: scaling=%.6f bias=%.6f\n",
                hp.speech_scaling_factor, hp.speech_bias_factor);
    }

    return ctx;
}

// ── Free ───────────────────────────────────────────────────────────────────

extern "C" void kugelaudio_free(struct kugelaudio_context* ctx) {
    if (!ctx) return;
    if (ctx->kv_neg_buf) ggml_backend_buffer_free(ctx->kv_neg_buf);
    if (ctx->kv_neg_ctx) ggml_free(ctx->kv_neg_ctx);
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->buf) ggml_backend_buffer_free(ctx->buf);
    if (ctx->buf_cpu) ggml_backend_buffer_free(ctx->buf_cpu);
    if (ctx->weight_ctx) ggml_free(ctx->weight_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ── Synthesize (scaffold — graph building to be implemented) ───────────────
//
// Full pipeline:
//   1. Tokenize text → token IDs (Qwen2.5 BPE)
//   2. Build prompt: system + voice + text + speech_start
//   3. Prefill LM (28-layer Qwen2.5-7B) with full prompt
//   4. AR decode loop:
//      a. Run LM step, get logits constrained to {start, end, diffusion, eos}
//      b. On speech_diffusion:
//         - Get hidden state (condition)
//         - Optionally: get negative condition (CFG)
//         - 20-step DPM-Solver++ SDE loop:
//           * Build sinusoidal timestep embed → t_embedder MLP
//           * Diffusion head: noisy_proj + cond_proj + t_emb → 4 AdaLN layers → final
//           * v-prediction → x0, then SDE update with noise
//         - Unscale: latent / scaling - bias
//         - Acoustic decoder: stem conv → 6 upsample stages → head conv → PCM
//         - Feed back: acoustic_connector(latent) → next embedding
//      c. On speech_end/eos: stop
//   5. Concat audio chunks → normalize → output

extern "C" float* kugelaudio_synthesize(struct kugelaudio_context* ctx,
                                        const char* text,
                                        int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) return nullptr;
    *out_n_samples = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kugelaudio: synthesizing: \"%s\"\n", text);
    }

    // TODO: implement full graph-building TTS pipeline
    // (follows vibevoice_synthesize pattern, ~800 LOC)
    fprintf(stderr, "kugelaudio: synthesis not yet implemented (scaffold)\n");
    return nullptr;
}

extern "C" float* kugelaudio_run_diffusion_step(struct kugelaudio_context* ctx,
                                                 const float* noisy_latent, int vae_dim,
                                                 int timestep,
                                                 const float* condition, int d_lm,
                                                 int* out_dim) {
    (void)ctx; (void)noisy_latent; (void)vae_dim; (void)timestep;
    (void)condition; (void)d_lm; (void)out_dim;
    return nullptr;
}

extern "C" float* kugelaudio_run_acoustic_decoder(struct kugelaudio_context* ctx,
                                                   const float* latent, int vae_dim,
                                                   int* out_n_samples) {
    (void)ctx; (void)latent; (void)vae_dim; (void)out_n_samples;
    return nullptr;
}

extern "C" int kugelaudio_load_voice(struct kugelaudio_context* ctx, const char* voice_path) {
    (void)ctx; (void)voice_path;
    return -1;
}
