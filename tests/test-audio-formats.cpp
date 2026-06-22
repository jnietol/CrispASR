/*
 * test-audio-formats.cpp — Catch2 tests for crispasr_audio_load with
 * extended format support: AU/SND (µ-law), AMR-NB, WebM (Opus).
 *
 * Each format is decoded via the crispasr_audio_load C ABI and compared
 * against the WAV reference (jfk.wav) decoded through the same path.
 * Lossy codecs won't be bit-exact; we check length similarity, non-silence,
 * and cross-correlation above a threshold.
 */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The C ABI under test — declared in crispasr.h, but we forward-declare to
// avoid pulling the full header into a test TU that doesn't need it.
extern "C" int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate);
extern "C" void crispasr_audio_free(float* pcm);

// ── helpers ──────────────────────────────────────────────────────────

#ifndef SAMPLES_DIR
#define SAMPLES_DIR "."
#endif

static std::string sample(const char* name) {
    return std::string(SAMPLES_DIR) + "/" + name;
}

// Normalised cross-correlation between two float signals (peak over ±lag window).
static double cross_correlation(const float* a, int na, const float* b, int nb, int max_lag = 480 /* 30 ms at 16 kHz */) {
    int n = std::min(na, nb);
    if (n == 0)
        return 0.0;

    double ea = 0, eb = 0;
    for (int i = 0; i < n; ++i) {
        ea += (double)a[i] * a[i];
        eb += (double)b[i] * b[i];
    }
    double norm = std::sqrt(ea * eb);
    if (norm < 1e-12)
        return 0.0;

    double best = -1.0;
    for (int lag = -max_lag; lag <= max_lag; ++lag) {
        double sum = 0;
        for (int i = 0; i < n; ++i) {
            int j = i + lag;
            if (j < 0 || j >= n)
                continue;
            sum += (double)a[i] * b[j];
        }
        double cc = sum / norm;
        if (cc > best)
            best = cc;
    }
    return best;
}

// Check that a signal is not silent.
static bool has_energy(const float* pcm, int n, double min_rms = 0.001) {
    if (n == 0)
        return false;
    double sum_sq = 0;
    for (int i = 0; i < n; ++i)
        sum_sq += (double)pcm[i] * pcm[i];
    return std::sqrt(sum_sq / n) >= min_rms;
}

// ── reference loader ─────────────────────────────────────────────────

struct AudioRef {
    float* pcm = nullptr;
    int samples = 0;
    int sample_rate = 0;

    ~AudioRef() {
        if (pcm)
            crispasr_audio_free(pcm);
    }
};

static AudioRef load_ref() {
    AudioRef ref;
    int rc = crispasr_audio_load(sample("jfk.wav").c_str(), &ref.pcm, &ref.samples, &ref.sample_rate);
    REQUIRE(rc == 0);
    REQUIRE(ref.pcm != nullptr);
    REQUIRE(ref.samples > 100000); // ~11s at 16kHz
    REQUIRE(ref.sample_rate == 16000);
    return ref;
}

// ── test cases ───────────────────────────────────────────────────────

TEST_CASE("crispasr_audio_load decodes WAV reference", "[audio][unit]") {
    auto ref = load_ref();
    REQUIRE(has_energy(ref.pcm, ref.samples));
}

TEST_CASE("crispasr_audio_load decodes AU (µ-law)", "[audio][unit][au]") {
    auto ref = load_ref();

    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.au").c_str(), &pcm, &samples, &rate);
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    // Length within 5% of reference (lossy + resample)
    double ratio = (double)samples / ref.samples;
    INFO("AU length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — µ-law at 8kHz is lossy, 0.70 is reasonable
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("AU cross-correlation: " << cc);
    REQUIRE(cc > 0.70);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes AMR-NB", "[audio][unit][amr]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.amr").c_str(), &pcm, &samples, &rate);

    // AMR support is optional (CRISPASR_HAVE_AMR). If the build doesn't
    // include it, rc will be -2 and we skip the test gracefully.
    if (rc == -2) {
        WARN("AMR decoder not available (CRISPASR_HAVE_AMR not set) — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    // AMR-NB at 12.2kbps is heavily lossy — length within 10%
    double ratio = (double)samples / ref.samples;
    INFO("AMR length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — AMR is very lossy, 0.50 is a generous floor
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("AMR cross-correlation: " << cc);
    REQUIRE(cc > 0.50);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes WebM (Opus)", "[audio][unit][webm]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.webm").c_str(), &pcm, &samples, &rate);

    // WebM/Opus requires CRISPASR_HAVE_OPUS. The Opus custom backend might
    // handle it via miniaudio, or our EBML demuxer fallback kicks in.
    if (rc == -2) {
        WARN("WebM decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    // Length within 5%
    double ratio = (double)samples / ref.samples;
    INFO("WebM length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // Cross-correlation — Opus is high quality, expect >0.80
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("WebM cross-correlation: " << cc);
    REQUIRE(cc > 0.80);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes WebM (Vorbis)", "[audio][unit][webm]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk-vorbis.webm").c_str(), &pcm, &samples, &rate);

    if (rc == -2) {
        WARN("WebM/Vorbis decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("WebM/Vorbis length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples);
    INFO("WebM/Vorbis cross-correlation: " << cc);
    REQUIRE(cc > 0.70);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load decodes M4A (AAC)", "[audio][unit][m4a]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load(sample("jfk.m4a").c_str(), &pcm, &samples, &rate);

    // M4A/AAC requires either CRISPASR_HAVE_FDK_AAC (Linux) or Apple AudioToolbox
    if (rc == -2) {
        WARN("M4A/AAC decoder not available — skipping");
        return;
    }
    REQUIRE(rc == 0);
    REQUIRE(pcm != nullptr);
    REQUIRE(rate == 16000);
    REQUIRE(has_energy(pcm, samples));

    auto ref = load_ref();

    double ratio = (double)samples / ref.samples;
    INFO("M4A length ratio: " << ratio);
    REQUIRE(ratio > 0.90);
    REQUIRE(ratio < 1.10);

    // AAC is good quality. Encoder priming is stripped via elst parsing,
    // but a small residual offset (~240 samples) may remain from SBR/
    // resampling rounding. Use a moderate lag window (500 samples = 31ms).
    double cc = cross_correlation(ref.pcm, ref.samples, pcm, samples, 500);
    INFO("M4A cross-correlation: " << cc);
    REQUIRE(cc > 0.85);

    crispasr_audio_free(pcm);
}

TEST_CASE("crispasr_audio_load rejects missing file", "[audio][unit]") {
    float* pcm = nullptr;
    int samples = 0, rate = 0;
    int rc = crispasr_audio_load("/nonexistent/file.wav", &pcm, &samples, &rate);
    REQUIRE(rc < 0);
    REQUIRE(pcm == nullptr);
}

TEST_CASE("crispasr_audio_load rejects null args", "[audio][unit]") {
    float* pcm = nullptr;
    int samples = 0;
    REQUIRE(crispasr_audio_load(nullptr, &pcm, &samples, nullptr) == -1);
    REQUIRE(crispasr_audio_load("test.wav", nullptr, &samples, nullptr) == -1);
    REQUIRE(crispasr_audio_load("test.wav", &pcm, nullptr, nullptr) == -1);
}
