# CrispASR v0.8.8

This release closes out **issue #89** — Japanese long-form transcription with
`parakeet-tdt-0.6b-ja` no longer silently drops half the speech. Content
recall on the reporter's own clips goes from **56–64 % to 96–97 %** (measured
against a whisper-large-v3-turbo reference with reading normalization), which
is the inter-model agreement ceiling — an independent SenseVoice run scores
the same recall on the same audio. It also picks up the MOSS-Transcribe
long-audio fixes (#218).

## Japanese long-form ASR — issue #89 (parakeet-ja)

The JA FastConformer encoder is unstable past ~12 s of attention context on
real speech, and blanks whole utterances whenever enough context follows them
— upstream NeMo fails identically on the same audio (its plain,
local-attention, and buffered long-form modes score 1–51 % content recall on
the reference clip; our runtime is character-identical to NeMo in the
comparable modes). The new default pipeline works around the model:

- **VAD slice cap** — VAD/energy slices are capped at 12 s and re-split at
  energy minima (never mid-word); each slice decodes in one NeMo-exact pass.
- **Gap-fill second pass** — any span ≥ 1 s the first pass left empty inside
  a slice is re-transcribed in isolation and the recovered words merged back.
  `CRISPASR_GAP_FILL=0` disables; `CRISPASR_GAP_FILL_MIN_CS` tunes.
- Applied everywhere: **CLI**, **session ABI** (all language bindings), and
  the **OpenAI-compatible server** (`/v1/audio/transcriptions`) via a shared
  implementation.
- **Regression guard**: new live test (`test-parakeet-ja-longform`) pins the
  behavior on a triple-length fixture through the session ABI.
- New runtime gate `CRISPASR_PARAKEET_ATT_CONTEXT="L,R"` — the equivalent of
  NeMo's `change_attention_model("rel_pos_local_attn")`, verified
  char-identical to NeMo at [128,128].

Measured (phonetic char-bigram recall vs whisper-large-v3-turbo): 60 s clip
**97.2 %** (was 64.2 %), 120 s **96.9 %** (was 61.3 %), 300 s **95.9 %**;
precision 90–93 %. Verified on **Metal** and **Vulkan** (MoltenVK: 95.1 % on
the same clip — the reporter's backend class).

## parakeet-ja GGUFs refreshed (`cstr/parakeet-tdt-0.6b-ja-GGUF`)

- All files now include the hybrid model's **CTC head** —
  `--parakeet-decoder ctc` works (earlier GGUFs lacked the tensors and fell
  back to TDT silently).
- New **q8_0** (TDT output byte-identical to F16 at half the size) — now the
  auto-download default for `parakeet-ja`.
- **q4_k guidance**: TDT decode degrades to repetition loops at q4-class
  quantisation on this model (autoregressive decode compounds the quant
  noise; pre-existing) — but **CTC decode over the same q4_k file is clean**.
  The runtime now prints a load-time warning suggesting
  `--parakeet-decoder ctc` or the q8_0.

## MOSS-Transcribe (#218)

- Degenerate greedy n-gram repetition loops collapsed via the shared
  `core_ngram::fix_loops` pass.
- Overlap-save disabled on long-audio slices for this backend (seam
  duplicates); long-audio path is **3.3× faster**.

## Tools

- `tools/asr_coverage_score.py` — char-bigram recall/precision of ASR output
  vs a stronger-model reference; separates "silently drops speech" from plain
  WER. `--strip-latin`, `--reading` (kana/kanji-variant-erasing hiragana
  normalization), `--per-line` loss localization.
- `tools/nemo_parakeet_blueprint.py` — run upstream NeMo's inference modes
  (plain / CTC / local attention / buffered) on a `.nemo` checkpoint to
  establish blueprint parity before hunting port bugs.

## Docs

- `docs/cli.md` — parakeet long-form section rewritten (JA pipeline, env-var
  table incl. `CRISPASR_PARAKEET_VAD_SLICE_CAP`, gap-fill knobs).
- `--align-only` standalone CTC forced alignment documented (#217).
- `PERFORMANCE.md` — final #89 coverage table; `LEARNINGS.md` — four
  transferable lessons from the #89 audit.
