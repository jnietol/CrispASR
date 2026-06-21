# Performance — KV Cache & State Cache Survey

Comprehensive survey of caching strategies across all CrispASR and CrispEmbed backends.

## KV Cache Implementations

### CrispASR — Autoregressive ASR/LLM Decoders

All autoregressive decoders use `core_attn::kv_self_attn` with persistent F16 KV tensors
allocated via `ggml_backend_alloc_ctx_tensors`. The cache is a 4D tensor
`(head_dim, max_ctx, n_kv_heads, n_layers)` for both K and V.

| Backend | KV Cache | Conv/Other State | Allocation | Notes |
|---------|----------|-----------------|------------|-------|
| voxtral | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Mistral-based GQA 32/8 |
| voxtral4b | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Fused QKV |
| qwen3_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | GQA expansion, RoPE NEOX |
| granite_speech | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | µP residual_multiplier |
| granite_nle | `core_attn::kv_self_attn` | Conv1d cache | `ggml_backend_sched` | NAR with conv streaming |
| cohere | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Encoder-decoder cross-attn KV |
| canary | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | FastConformer enc + Transformer dec |
| glm_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | GLM decoder |
| mimo_asr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | MIMO encoder-decoder |
| gemma4_e2b | Dual cache (sliding+full) | — | `ggml_backend_sched` | Separate K/V for each attention type |
| funasr | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Q/K/V merged for efficiency |
| mini_omni2 | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | 8-stream multimodal |
| moss_audio | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Whisper enc + Qwen3 LM |

### CrispASR — TTS/Speech Generation

| Backend | KV Cache | Conv/Other State | Allocation | Notes |
|---------|----------|-----------------|------------|-------|
| csm_tts | Dual (backbone+depth) | — | `ggml_backend_sched` | 2 separate KV caches |
| qwen3_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Post-proj Q/K norm |
| cosyvoice3_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | AR speech-token LM |
| pocket_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Llama-1B backbone |
| tada_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Per-token flow matching |
| zonos_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | CFG-guided, 9-codebook DAC |
| kugelaudio | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | 7B Qwen2.5 + DiT diffusion |
| voxcpm2_tts | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | Flow-matching diffusion |
| vibevoice | `core_attn::kv_self_attn` | Conv cache | `ggml_backend_sched` | σ-VAE streaming |
| **lfm2_audio** | `core_attn::kv_self_attn` | **Conv state cache** | `ggml_gallocr` (direct, §206) | **Hybrid conv+attn backbone; backbone computes directly on ctx->backend — not the sched — to avoid the weight-less-first-op cross-backend-copy bug** |
| **nemotron** | cache_last_channel + cache_last_time | **Conv + attn cache** | `ggml_backend_sched` | **Cache-aware streaming FastConformer** |
| bark_tts | None (3× non-cached forward) | — | `ggml_backend_sched` | Could benefit from KV cache |
| chatterbox | `core_attn::kv_self_attn` | — | `ggml_backend_sched` | T3 AR + S3Gen flow |

### CrispASR — Non-Autoregressive (no KV cache needed)

| Backend | Reason no cache needed |
|---------|----------------------|
| parakeet | CTC/TDT encoder — single forward pass |
| fastconformer_ctc | CTC head — encoder-only |
| wav2vec2 | CTC head — encoder-only |
| sensevoice | CTC multi-task — encoder-only |
| paraformer | NAR decoder — single pass with CIF predictor |
| kokoro | StyleTTS2 — non-autoregressive (has LSTM but runs fresh each call) |
| f5_tts | DiT flow-matching — iterative denoising, not autoregressive |
| fastpitch | Non-autoregressive parallel TTS |
| melotts | VITS2 — flow-based, non-autoregressive |
| piper | VITS — same as melotts |
| speecht5 | Encoder-decoder but mel output is parallel |
| dia_tts | Single-forward with CFG |
| parler_tts | T5 encoder + MusicGen — greedy but short |

### CrispEmbed — Encoders Only

No KV caches needed. All models are encoder-only (BERT, ViT, CNN).
Three models compute position biases per-forward: `bttr_ocr`, `ppformulanet_ocr`, `posformer_ocr`.

## LFM2-Audio: Hybrid Conv + Attention Cache

The LFM2 backbone uses two distinct cache types:

### Attention KV Cache (6 of 16 layers)
- Standard `core_attn::kv_self_attn` with F16 persistent tensors
- Cache: `(head_dim=64, max_ctx=2048, n_kv_heads=8, n_attn_layers=6)`
- Written during prefill, read+extended during decode

### Conv State Cache (10 of 16 layers)
- CPU-side float arrays: `(hidden=2048, kernel-1=2)` per layer
- Total: 10 layers × 2048 × 2 × 4 bytes = **160 KB**
- Stores last K-1=2 Bx columns from the ShortConv computation
- During decode: cached columns + new Bx → depthwise conv → take last output
- Shifted left by 1 after each step

### Depthformer Cache
- 6-layer transformer generating 8 codebooks per audio frame
- Currently: O(codebooks²) = 36 layer passes per frame (no cache)
- Target: O(codebooks) = 8 passes with manual KV cache (in progress)

## Allocation Strategies

| Strategy | Used by | Pros | Cons |
|----------|---------|------|------|
| `ggml_backend_sched` | 73 backends | GPU offload, automatic tensor placement | More complex setup |
| `ggml_gallocr` | 11 backends | Simple, exact allocation | CPU-only, manual tensor_set/get |
| Fixed buffer (`no_alloc=false`) | Legacy / simple paths | Simplest code | Wasteful, page-fault overhead |

## Performance Benchmarks (LFM2-Audio, 4-core CPU)

| Configuration | ASR (11s JFK, 23 tok) | TTS (short) | Notes |
|--------------|----------------------|-------------|-------|
| F16 no cache | ~480s | N/A | Original |
| F16 KV only | ~200s | N/A | 2.4× |
| F16 KV+conv | ~47s | ~60s | 10× |
| Q4_K KV+conv | ~31s | ~45s | 15× |
| F16 gallocr | ~2m20s | — | Reduced sys overhead |
| Q4_K gallocr+256MB | ~1m8s | — | Best CPU perf |
| GPU (M1 Metal, §206) | ~15s | — | Correct now (gallocr direct compute); AR decode is dispatch-bound so currently ~slower than threaded CPU. GPU-decode graph caching is the perf follow-up. |
