# CrispASR v0.8.4

> **Note:** v0.8.3 was tagged before the VERSION file was bumped (binary
> self-identified as 0.8.2). v0.8.4 supersedes it and fixes the version
> mismatch. All features described here were already in v0.8.3 binaries;
> this release adds the Qwen3-TTS one-breath fix, VibeVoice long-input GPU
> crash fix, and the release-process hardening.

---

## What's new

### Audio format expansion — no ffmpeg required

CrispASR now decodes a much wider range of audio containers and codecs
without any dependency on ffmpeg:

- **Opus** (`.opus`, bare Ogg-Opus): via libopus + libopusfile (FetchContent, auto-downloaded; opt-out with `-DCRISPASR_OPUS_FETCH=OFF`)
- **WebM**: Ogg-demuxed Opus stream via the same path
- **AU / µ-law / AIFF**: Sun Audio and classic broadcast formats
- **AMR-NB / AMR-WB** (`.amr`): via opencore-amr FetchContent fork (`-DCRISPASR_AMR_FETCH=ON`; decoder-only, MIT-compatible)
- **M4A / AAC / ALAC / CAF** on Apple platforms: via AudioToolbox — zero extra dependencies, native OS decode
- **M4A / AAC** on Linux/Windows: via libfdk-aac loaded at runtime via dlopen — binary stays MIT-clean; falls back gracefully if the library is absent

Go and Ruby bindings expose the same set via their FetchContent link flags.
WASM/Emscripten builds use Opus with x86 SIMD disabled for portability.

### Qwen3-TTS improvements

**GQA_NATIVE + chunked codec decode (#183)**
- Switched attention kernel to `GQA_NATIVE` mode — O(N²)→O(N) scaling on Vulkan for long texts
- Codec decode is now chunked: VRAM peak stays flat regardless of text length; RTF remains ~0.5×
- Scratch scheduler reset between requests prevents cross-request memory bloat

**1.7B small_to_mtp graph fold (#161)**
- The `small_to_mtp` projection that previously launched 16 separate GPU graphs per frame is now fused into the `code_pred` graph — ~12–15% faster on Metal, ~5–6% on CUDA P100
- Opt-out via `QWEN3_TTS_CP_MTP_NOFUSE=1`

**CUDA codec chunk cap**
- CUDA builds no longer OOM on very long inputs due to oversized codec batches; chunk size is now bounded

**One-breath synthesis fix**
- Restored single-forward-pass synthesis path that was accidentally broken; multi-sentence inputs no longer stutter at sentence boundaries

### VibeVoice TTS fixes

**Long-input GPU crash (#171)**
- Fixed illegal memory access on CUDA when synthesising long text inputs

**Vulkan / RDNA4 segfault (#184)**
- Fixed segfault on AMD RDNA4 GPUs caused by strided `AdaLN` views passed directly to Vulkan ops — `ggml_cont` now materialises contiguous copies before compute
- Applies to all Vulkan backends (the `ggml_cont` on strided `view_2d` fix is backend-agnostic)

**Bucket scheduler stabilisation (#184)**
- Bucket-cache LM step is re-enabled on GPU after fixing the gallocr state lifetime; bucket sched is always reset+alloc'd before use to prevent stale-pointer crashes

### Cohere ASR

- Added `CRISPASR_COHERE_LEGACY_SA` env flag to fall back to the pre-#161 self-attention path for users who hit the perf regression on non-GQA architectures

### Orpheus GGUF consolidation

All Orpheus 3B GGUF variants (F16, Q8_0, Q4_K) are now consolidated in a single canonical repo `cstr/orpheus-3b-0.1-ft-GGUF`; the old `cstr/orpheus-3b-base-GGUF` repo shows a deprecation notice pointing to the new location. Model registry, regression manifest, test harness, and all HF READMEs updated.

### iOS / visionOS / tvOS xcframework

- AudioToolbox and CoreAudio frameworks added to xcframework slices (required for AAC/M4A decode on-device)
- Opus and AMR disabled on iOS (FetchContent-built static libs are missing from `libtool -static` combined archives)
- visionOS and tvOS slices skip Opus explicitly to avoid link failures
- Per-slice failure markers added to `build-xcframework.sh` so partial failures surface individually in CI

### Release process hardening (fixes #189)

- **`validate-version` CI job**: `release.yml` now checks that the `VERSION` file matches the pushed tag before `publish` runs. A mismatched tag (the v0.8.3 mis-tagging root cause) now fails CI before any artifacts are published.
- **`scripts/bump-version.sh`**: one-step helper that writes `VERSION`, runs `sync-version.py` to propagate to `Cargo.toml` / `package.json` / `pyproject.toml` / `pubspec.yaml`, commits `release: bump VERSION to X.Y.Z`, and creates an annotated tag — eliminating the manual bump-and-forget pattern.

---

## Bug fixes

| Area | Fix |
|------|-----|
| VibeVoice | GPU crash on long inputs (#171) |
| VibeVoice | Vulkan/RDNA4 segfault — strided AdaLN views |
| VibeVoice | Stale gallocr bucket sched state |
| Qwen3-TTS | One-breath synthesis regression |
| Qwen3-TTS | CUDA OOM on long codec batches |
| Audio | Windows Media Foundation `mfapi.h` include order (MSVC) |
| Audio | MinGW `uint32_t` missing — `<cstdint>` before miniaudio |
| Build | libopusfile private dependency not linked transitively |
| Build | iOS xcframework FetchContent static lib path |
| CMake | AMR FetchContent URL corrected to CrispStrobe fork |
| CI | Go LDFLAGS sync script missing `OPUS_FETCH=ON` |
| CI | clang-format on `miniaudio_libopus.c` |

---

## Upgrading

No breaking changes to the C ABI, Python binding, or CLI flags.

Packages that vendor `cstr/orpheus-3b-base-GGUF` should switch to
`cstr/orpheus-3b-0.1-ft-GGUF`. The model files are identical; only the
HuggingFace repo name changed.

---

## Full changelog

`git log v0.8.2..v0.8.4 --oneline --no-merges`
