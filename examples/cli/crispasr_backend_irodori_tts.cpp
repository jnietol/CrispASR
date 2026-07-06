// crispasr_backend_irodori_tts.cpp — adapter for Aratako/Irodori-TTS.
//
// Flow-matching RF-DiT TTS with zero-shot voice cloning via DAC-VAE latents.
// Japanese-focused, 48 kHz output. Requires DAC-VAE decoder companion GGUF.
//
// Usage:
//   crispasr --backend irodori-tts -m auto --tts "こんにちは" -o out.wav

#include "crispasr_backend.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "irodori_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Look for dacvae GGUF next to the model.
static std::string discover_dacvae(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto pos = p.find_last_of("/\\");
        return (pos != std::string::npos) ? p.substr(0, pos) : ".";
    };
    std::string dir = dir_of(model_path);
    for (const char* name : {"dacvae-ja-32dim-f16.gguf", "dacvae-ja-32dim-q8_0.gguf"}) {
        std::string candidate = dir + "/" + name;
        if (file_exists(candidate))
            return candidate;
    }
    return {};
}

class IrodoriTTSBackend : public CrispasrBackend {
public:
    IrodoriTTSBackend() = default;
    ~IrodoriTTSBackend() override { IrodoriTTSBackend::shutdown(); }

    const char* name() const override { return "irodori-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        return {}; // TTS-only backend
    }

    bool init(const whisper_params& p) override {
        irodori_tts_params fp = irodori_tts_default_params();
        fp.n_threads = p.n_threads;
        fp.verbosity = p.no_prints ? 0 : 1;
        fp.seed = p.seed;
        fp.use_gpu = p.use_gpu;

        ctx_ = irodori_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            std::fprintf(stderr, "crispasr[irodori-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // ── Codec companion loading (3-tier: --codec-model → sibling → registry auto-download) ──
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty()) {
            codec_path = discover_dacvae(p.model);
        }
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            std::string quant;
            if (crispasr_registry_lookup(p.backend, entry, quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            irodori_tts_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                std::fprintf(stderr, "crispasr[irodori-tts]: codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            std::fprintf(stderr, "crispasr[irodori-tts]: no DAC-VAE codec found. Pass --codec-model PATH or place "
                                 "dacvae-ja-32dim-f16.gguf next to the model. (output will be silent)\n");
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*p*/) override {
        if (!ctx_ || text.empty())
            return {};

        float* pcm = nullptr;
        int sr = 0;
        int n = irodori_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> out(pcm, pcm + n);
        std::free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? irodori_tts_sample_rate(ctx_) : 48000; }

    void shutdown() override {
        if (ctx_) {
            irodori_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    irodori_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_irodori_tts_backend() {
    return std::make_unique<IrodoriTTSBackend>();
}
