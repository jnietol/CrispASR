# CrispASR v0.8.5

This release makes **TADA multilingual TTS production-ready**, adds a new
**Japanese RNNT ASR backend (ReazonSpeech)**, moves **CosyVoice3 onto the GPU**
with batched CFG, and ships **C# bindings** for the Session C ABI.

---

## What's new

### TADA TTS — multilingual, reliable, GPU-accelerated

The bulk of this release. TADA (Llama-3.2 backbone + per-token flow-matching
duration/acoustic head + TADA codec → 24 kHz) went from "English demo" to a
dependable multilingual TTS backend.

**Reliable multilingual timing (`num_acoustic_candidates`)**
- TADA's per-token duration head is **noise-sensitive**: a single unlucky noise
  draw can collapse token durations into rushed, garbled speech (a property of
  the model — the PyTorch reference behaves identically with the same noise).
- Ported the reference's `num_acoustic_candidates` ranking: several
  flow-matching candidates are drawn per token and the best is kept by
  reconstruction likelihood. **Default 4** on the CLI and through the C ABI;
  override with `TADA_NUM_CANDIDATES=N`.
- All candidates for a step solve in **one batched flow-matching forward**
  (~13N small forwards → ~13), so raising the count adds little wall-clock.
- Validated by ASR roundtrip in German and French (single-candidate runs
  sometimes garble; candidates=4 is reliable).
- Wired through the **full session C ABI + every binding** (Python, Go, Rust,
  Dart, Java, C#, Ruby, JS/WASM) via the new
  `crispasr_session_set_tts_num_candidates(n)` setter — bindings and the HTTP
  server get robust timing out of the box, not just the CLI.

**Voice references — create your own with `--make-ref` (no Python)**
- New in-tree **TADA encoder runtime** + converters: build a voice-reference
  GGUF directly from a C++ pipeline with `--make-ref`, no PyTorch required.
- **Auto-download language voice refs** on `-l <lang>`: the multilingual 3B-ml
  model pulls a matching reference (ar/ch/de/es/fr/it/ja/pl/pt) automatically.
- Language-reference GGUFs published to `cstr/tada-tts-{1b,3b-ml}-GGUF`.

**GPU + quantization**
- TADA now runs on the **GPU runtime path** (Metal/CUDA/Vulkan), sharing the
  backend with the codec.
- Pos/neg CFG batched into a **single B=2 graph per Euler step**; prefill tokens
  batched and pos→neg KV copied.
- **Quantized FM Metal fallback** + a **mixed-precision Q4_K quantizer** that
  matches the reference's BF16 noise rounding.
- Vulkan: `ggml_cont` materialises contiguous copies before the B=2 FM ops.

**Timing-embedding correctness (#192)**
- Fixed time-embedding during prompt-phase prefill and the time-transition
  boundary; only generated acoustic frames (not prompt-phase frames) are passed
  to the codec. Together these removed spurious trailing silence and rushed
  starts.
- `1b` = English-only; `3b-ml` = multilingual — now documented, with the voice
  cloning + `-l` auto-download workflow.

**Diff harness fairness**
- `crispasr-diff` for TADA is now a fair C++-vs-Python comparison (reseed before
  generate, store the prompt transcript), confirming the C++ forward pass
  matches the reference (`time_before` cos=1.0, codec cos=0.99998).

### ReazonSpeech — Japanese RNNT ASR

- New backend for `reazon-research/reazonspeech-nemo-v2`, a Japanese RNNT
  (transducer) ASR model.

### CosyVoice3 — GPU generation + perf

- **GPU generation** enabled (downloads colocated).
- **Batched CFG** and **lazy-loaded cloning encoders** (only loaded when voice
  cloning is requested).
- **Right-sized autoregressive KV cache** — lower memory, lower latency.
- Full session wiring; latency-tuning + KV-sizing docs added.

### C# bindings

- New **C# bindings** for the CrispASR Session C ABI, with unit tests
  (`InternalsVisibleTo` wired so the test project can reach `NativeMethods`).

### Qwen3-TTS

- Encoder transformer is **chunked** to fix OOM on long reference audio (#187).

---

## Bug fixes

| Area | Fix |
|------|-----|
| TADA | Noise-sensitive duration collapse → candidate ranking (default 4) |
| TADA | Time embedding during prompt-phase prefill + time-transition boundary (#192) |
| TADA | Codec fed prompt-phase frames instead of only generated frames |
| TADA | PyTorch-compatible MT19937 noise + BF16 rounding parity |
| TADA | Skip FM solver during prefill to avoid RNG offset |
| TADA | Encoder Conv1d stride padding + tensor layout; float32 cast; transformers 5.x |
| TADA | Language-ref converter: gated Llama → unsloth mirror, HF-token pass, tokenizer dict→str |
| Windows | Guard `setenv` with `_WIN32`/`_putenv_s` in the crispasr-diff TADA path |
| CosyVoice3 | Enable GPU generation; session wiring + formatting |
| Qwen3-TTS | OOM on long reference audio — chunked encoder (#187) |
| Build | Fully fix building against system `libopusfile` (#191) |
| Registry | Add `qwen3-forced-aligner` to the model registry (#190) |
| CI | Repair HIP checkout + Android `mediandk` link |

---

## Upgrading

No breaking changes to the C ABI, bindings, or CLI flags — all additions.

- TADA now ranks **4 timing candidates** per token by default (CLI, bindings,
  and server). Set `TADA_NUM_CANDIDATES=1` (or `set_tts_num_candidates(1)`) to
  restore the previous single-draw behaviour.
- The new `crispasr_session_set_tts_num_candidates(n)` setter is additive; older
  bindings without the symbol soft-no-op.

---

## Full changelog

`git log v0.8.4..v0.8.5 --oneline --no-merges`
