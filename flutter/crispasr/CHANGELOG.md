# Changelog

## 0.5.13

- **Whisper alt-token capture (`--alt N` parity)** — per-token
  top-N alternative-candidate capture for whisper greedy
  decode. Closes the last open whisper-cli equivalent gap. New
  `Word.alts` list (with a new `AltToken` value class, default
  `const []` so existing call-sites stay source-compatible)
  on both the low-level `Segment` shape (from `transcribePcm`)
  and the unified session `SessionSegment` shape (from
  `CrispasrSession.transcribe`). New
  `TranscribeOptions.altN` (default 0 = off) plumbs through
  the low-level path; sticky
  `CrispasrSession.setAltN(int)` (matches the
  `setFallbackThresholds` / `setWhisperDecodeExtras` pattern)
  threads through the session path. Pre-0.5.13 dylibs raise
  `UnsupportedError` for the session setter and report
  `Word.alts == []` from both readers, so apps stay loadable
  and just hide the affordance.
- Capture happens inside `whisper_sample_token` so the alts
  are softmax probabilities at the same decode step as the
  chosen token. Beam search is excluded — siblings are
  beam-conditional rather than greedy alternatives. Chosen
  token is not included in the alt list; entries are sorted
  descending by probability.
- Whisper session-transcribe path now also populates
  `SessionSegment.words` (it previously returned only
  segment-level text) — closes a long-standing gap with the
  parakeet / canary backends as a side-benefit of needing
  per-token data for the word-alt mapping. Whisper sub-word
  BPE means a multi-token word's alts cover the first content
  token only (e.g. "kubectl" surfaces alts for "kub"); full
  word-level enumeration is deferred.
- C-ABI symbols: `crispasr_params_set_alt_n`,
  `crispasr_session_set_alt_n`, `crispasr_token_n_alts` /
  `_alt_id` / `_alt_p` / `_alt_text`,
  `crispasr_session_result_word_n_alts` / `_alt_text` /
  `_alt_p`. New whisper getters
  `whisper_full_get_token_n_alts` / `_alt_id` / `_alt_p`
  (plus `_from_state` variants). All nine symbols pinned in
  `bindings_smoke_test.dart`.

## 0.5.12

- **Audio enhancement (RNNoise pre-step)** — new top-level
  `enhanceAudioRnnoise(Float32List pcm)` runs xiph/rnnoise v0.1
  on a 16 kHz mono float32 buffer (upsample to 48 kHz →
  RNNoise frame loop → downsample back) and returns a fresh
  same-length `Float32List`. Backed by the new C-ABI
  `crispasr_enhance_audio_rnnoise`; pre-0.5.12 dylibs raise
  `UnsupportedError` so callers graceful-degrade. State is
  per-call so worker isolates can run enhancement concurrently
  without coordination.

## 0.4.9

- Initial pub.dev release.
- Dart FFI bindings for the CrispASR C ABI (`src/crispasr_c_api.cpp`).
- Supports all 17 backends: Whisper, Qwen3-ASR, FastConformer, Canary, Parakeet, Cohere, Granite-Speech, Voxtral (Mistral 1.0/4B), wav2vec2, GLM-ASR, Kyutai-STT, Moonshine, FireRed, OmniASR, VibeVoice-ASR, plus FireRedPunc post-processor.
- Unified `Session` API across all backends; legacy `CrispASR` Whisper-shaped API preserved.
- Word-level alignment, speaker diarization, and language ID helpers.
- Auto-download of registered models via the model registry.
