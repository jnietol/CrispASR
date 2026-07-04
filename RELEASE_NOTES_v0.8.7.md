# CrispASR v0.8.7

This release adds **two new backends** (BananaMind-TTS-V2.1, MOSS-Transcribe-preview-2B ASR
and M2M-100 translation), a **much faster Granite decode path** (up to ~12× RTFx via
CUDA-graph capture, in-graph argmax, and fused embedding lookup), a **full imatrix
quantization pipeline** (producer + consumer, IQ4_NL/XS, per-tensor overrides), and a large
batch of correctness fixes across **TADA, MOSS/Vulkan, Granite long-audio, Parakeet, and
VibeVoice TTS**.

Highlights: the VibeVoice TTS quantized GGUFs no longer produce a spurious "music"/hum onset
(the diffusion prediction head is now kept at full precision), native-Vulkan segfaults on the
MOSS/MOSS-Audio encoders are fixed, and the Mimi codec now defaults to causal + sliding-window
masking (a measured WER/quality win).

## New backends & models

- **BananaMind-TTS-V2.1** — new TTS backend (Tacotron-lite acoustic model + HiFi-GAN vocoder),
  wired end-to-end per `docs/contributing.md` (CLI, C-API, bindings, quantization, tests).
- **MOSS-Transcribe-preview-2B** — new ASR backend (Qwen3-Omni encoder + Qwen3-1.7B LM).
- **M2M-100** — `m2m100-f16` registered for exact HF translation parity.
- **Qwen3-ASR-1.7B-JA-Anime-Galgame** — auto-download registry entry (#212); the Qwen3-ASR
  converter now accepts both `-hf` and non-`-hf` model layouts.

## Performance — Granite decode

- **CUDA-graph-capture bucketed decode** — up to **6.4×** faster on CUDA.
- **In-graph argmax** — up to **12× RTFx**, **+32%** on Q4_K.
- **Fused embedding lookup** into the captured decode graph (F16); **raw-gallocr
  allocate-once** decode on non-capture backends (Metal).
- Measured finding: the Metal ICB graph-replay path is a **DUD** (decode is GPU-bound); probe
  left in behind `CRISPASR_METAL_PROFILE`.

## Quantization

- **imatrix (importance-matrix) pipeline** — producer + consumer, activation-weighted k-quants,
  **IQ4_NL / IQ4_XS**, and **re-quantization directly from an already-quantized GGUF**.
- **`--tensor-type <regex>=<type>`** per-tensor overrides, an **A/B CER metric**, and a
  published **CC0 calibration set**.
- **`CRISPASR_QUANT_LMHEAD`** + aligner `lm_head`-at-F16 option (q8-everything aligner is
  bit-identical, #192).
- **VibeVoice diffusion-head carve-out (#171)** — for any `vibevoice-*` arch the quantizer now
  keeps the diffusion prediction head (`pred.*`), connectors (`at_conn.*`, `se_conn.*`), EOS
  classifier (`tts_eos.*`) and speech-type table at source precision; only the Qwen2 backbone
  and the deterministic VAE decoder are quantized. This prevents quantization error in the
  CFG-driven diffusion head from compounding into a spurious non-speech "music"/hum onset
  before the voice. Overridable with `CRISPASR_VIBEVOICE_QUANT_ALL=1`; guarded by a regression
  test. The published `vibevoice-realtime-0.5b`, `vibevoice-1.5b` and `vibevoice-7b` GGUFs have
  been regenerated with this recipe — **re-download to pick up the fix**.

## Mimi codec

- **Causal + sliding-window masking is now the default** for `kyutai_stt` and `csm_tts`
  (Mimi decoder transformer) — a measured TTS→ASR / WER A/B win (e.g. csm_tts 9.3% vs 12.0%
  WER). Symmetric `CRISPASR_MIMI_CAUSAL` gate to A/B either direction.

## Fixes

**VibeVoice TTS (#171)**

- Voice-prompt KV per-head stride leak that corrupted the speaker prompt on server
  multi-request runs (1st/2nd request fine, later requests garbled).
- The quantization carve-out above (music/hum onset).

**MOSS / MOSS-Audio native Vulkan (#215)**

- Root-caused a segfault to a conv graph cached across slices (use-after-free) and to
  `flash_attn_ext` on the encoder; fixes drain the queue before `vkResetCommandPool`, use
  manual masked attention on the encoders, and restore GPU execution on native Vulkan.

**TADA (#192 / #201)**

- Query-time inline `.wav` voice cloning and config-parity fixes; `--make-ref` reachable and
  fails loudly on unresolved `--voice`; `--align` forced-alignment word timestamps;
  auto-download of `tada-encoder` / `tada-aligner`; generation-loop parity fixes (fixed-step
  loop, candidate scoring, default candidates) so words are no longer truncated.

**Granite long-audio & templates (#205)**

- Recovered dropped chunk-context slices and spaceless text rebuilds; byte-exact chat template
  (system turn); route the `-plus` model through the control-token template for real word
  timestamps; honour `--max-len` on text-only backends.

**Parakeet long audio (#208)**

- Overlapping-window merge recovers dropped sections; bounded session long-audio encode and
  fixed a repeated-call encoder-cache collapse.

**Other**

- VAD returns empty for silent audio instead of hallucinating (#213).
- Respect `--gpu-backend` preference across all backends (#214).
- Kokoro built-in G2P technical-token normalization (#216).
- GLM-ASR / FireRed-ASR warn on unsupported `-l` instead of silently ignoring (#199).
- PCS/aligner: dequantize FC-head weights (q4_k previously crashed on every inference).
- Central f16→f16 GQA-REPEAT guard for all `kv_self_attn` callers on Vulkan (#200); dots-tts
  F32 KV read on Vulkan (#200).

## Platform & bindings

- **Android**: native Opus decoding enabled (#26); Media NDK made optional for Termux builds
  (#210); CUDA 13 build variant.
- **Windows**: portable `setenv`/`unsetenv` shim for MSVC.
- **Go bindings**: cgo `LDFLAGS` synced (adds `-lbananamind-tts`) so the static link and the
  drift check stay green.
- **All bindings**: expose `transcribe_chunked` + long-form progress callbacks (#208).
