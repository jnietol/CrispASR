---
license: other
license_name: funasr-model-license-v1.1
license_link: https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE
language:
- zh
- yue
- en
- ja
- ko
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech-recognition
- transcription
- ggml
- gguf
- funasr
- sensevoice
- sanm
- multilingual
- language-identification
- emotion-recognition
- audio-event-detection
library_name: ggml
base_model: FunAudioLLM/SenseVoiceSmall
---

# SenseVoiceSmall — GGUF (ggml-quantised)

GGUF / ggml conversion of [`FunAudioLLM/SenseVoiceSmall`](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) for use with the `sensevoice` backend in **[CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR)**.

SenseVoiceSmall is Alibaba's **multi-task encoder-only ASR**: one forward pass through a 70-block SANM encoder emits the full transcript **plus** the spoken language ID, emotion, and audio-event tags through a single CTC head. Non-autoregressive design → **15× faster than Whisper-Large** (70 ms for 10 s of audio in upstream's measurements).

- **70-block SenseVoiceEncoderSmall** (1 entry block @ 560→512 + 49 main blocks + 20 tp blocks, all 512-dim, 4 heads, FSMN k=11 depthwise convolution branch — the same encoder body Fun-ASR-Nano-2512 ships, just here paired with a CTC head instead of an LLM decoder)
- **4 query embeddings** (language / event / emotion / textnorm) prepended to the LFR fbank features so the encoder can emit rich annotations at those positions
- **CTC head** (`ctc.ctc_lo`, 25055 SentencePiece pieces)
- **50+ languages** with native LID (no whisper-tiny pre-step needed)
- **F16 only** ship: 0.47 GB. Q8_0/Q4_K pending.

## What you get in the output

The transcript carries the 4-token rich-annotation prefix as readable
special tokens followed by the SentencePiece-detokenised transcript:

```text
<|en|><|HAPPY|><|Speech|><|withitn|>And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
<|zh|><|NEUTRAL|><|Speech|><|withitn|>开饭时间早上9点至下午5点。
<|yue|><|NEUTRAL|><|Speech|><|withitn|>呢几个字都表达唔到我想讲嘅意思。
<|ja|><|NEUTRAL|><|Speech|><|withitn|>うちの中学は弁当制で持っていけない場合は、50円の学校販売の パンを買う。
<|ko|><|NEUTRAL|><|Speech|><|withitn|>조금만 생각을 하면서 살면 훨씬 편할 거야.
```

Languages: `<|zh|>` / `<|en|>` / `<|yue|>` / `<|ja|>` / `<|ko|>` / `<|nospeech|>`.
Emotions: `<|HAPPY|>` / `<|SAD|>` / `<|ANGRY|>` / `<|NEUTRAL|>` / `<|EMO_UNKNOWN|>`.
Audio events: `<|Speech|>` / `<|Music|>` / `<|Applause|>` / `<|Laughter|>` / `<|Cry|>` / `<|BGM|>`.
Text norm: `<|withitn|>` (Arabic digits, punctuation) or `<|woitn|>` (raw).

## Files

| File | Size | Notes |
| --- | ---: | --- |
| `sensevoice-small-f16.gguf` | 0.47 GB | F16 reference weights |

## Quick Start

```bash
git clone https://github.com/CrispStrobe/CrispASR
cd CrispASR
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target crispasr-cli

./build/bin/crispasr \
    --backend sensevoice \
    -m /path/to/sensevoice-small-f16.gguf \
    -f samples/jfk.wav -l en
```

## Verification

`crispasr-diff sensevoice` is 76/76 PASS, byte-identical `generated_text`,
on Alibaba's own example `zh.mp3`; 75/76 PASS on `samples/jfk.wav` with
the single difference being the emotion-tag argmax flipping between
`<|ANGRY|>` and `<|EMO_UNKNOWN|>` (F16/op-order pushes that one slot
across a near-tied boundary; the transcript itself is byte-identical
in both runs). On Apple M1 Metal the runtime hits **15-22× realtime**.

## Licence + attribution

Upstream **FunAudioLLM/SenseVoiceSmall**:

- **Code** (the `funasr` Python package): Apache-2.0.
- **Model weights**: [**FunASR Model License v1.1**](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/blob/main/LICENSE) (Alibaba) — commercial use OK with attribution. Confirmed on the upstream-tracking discussion in [CrispStrobe/CrispASR#99](https://github.com/CrispStrobe/CrispASR/issues/99).

These GGUF files are a quantised / repackaged distribution of the upstream weights and inherit the FunASR Model License v1.1. Please attribute Alibaba / FunAudioLLM in downstream products.

> If you use this model, please also cite the upstream FunASR work.
> See the [upstream model card](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) for the canonical citation.
