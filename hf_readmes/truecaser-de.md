---
license: mit
language:
- de
tags:
- truecasing
- text-processing
- german
- nlp
- statistical
pipeline_tag: token-classification
---

# German Statistical Truecaser

Statistical truecasing model for German, trained on 2M lines of German Wikipedia (CC-BY-SA source, model released under MIT).

Restores proper capitalization of German nouns, proper names, and acronyms in lowercase ASR output.

## How It Works

For each word (lowercased), the model stores frequency counts of three casing variants:
- **lc**: all lowercase (e.g. "die")
- **u1**: first letter capitalized (e.g. "Katze")
- **uc**: all uppercase (e.g. "NATO")

At inference, the variant with the highest count is applied. Sentence-initial words are always capitalized.

## Stats

- **Entries**: 375,283 unique words
- **Training data**: German Wikipedia (3M lines, mid-sentence words only, min count 5)
- **File size**: 9.2 MB
- **Inference**: instant (hash table lookup, no neural network)

## Usage with CrispASR

```bash
# Auto-download German truecaser
crispasr --backend moonshine -m model.gguf --truecase-model auto -f audio.wav

# Combined with punctuation restoration
crispasr --backend wav2vec2-de -m model.gguf \
    --punc-model punctuate-all --truecase-model auto -f audio.wav
```

## Example

| Stage | Output |
|-------|--------|
| Raw ASR | `die schnelle braune katze springt über den faulen hund` |
| + punctuation | `die schnelle braune katze springt über den faulen hund.` |
| + truecasing | `Die schnelle Braune Katze springt über den faulen Hund.` |

## Limitations

- Statistical only — no context awareness (adjective "braune" vs surname "Braun" are ambiguous)
- German-specific (separate models needed for other languages)
- Does not handle mixed-case words like "mRNA" or "iPhone"

## License

MIT
