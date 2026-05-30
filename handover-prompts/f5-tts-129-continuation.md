# F5-TTS (#129) — Continuation Handover

## Current state

Native C++ ggml runtime for SWivid/F5-TTS. 4 commits on `main`, not yet
pushed. All infrastructure is in place; the DiT transformer attention has
one remaining bug preventing per-layer cos ≥ 0.999.

## What's done

### Files created
```
src/f5_tts.h                              — C API header (seed, cfg, speed, ref audio)
src/f5_tts.cpp                            — ~1100 LOC runtime (CPU-only)
models/convert-f5-tts-to-gguf.py          — safetensors+vocos → single GGUF (1337 MB F32)
tools/reference_backends/f5_tts.py         — Python reference dumper (20 intermediate tensors)
tests/test-f5-tts.cpp                     — diff test harness (loads ref GGUF, compares stages)
```

### Files modified
```
src/CMakeLists.txt                        — add_library(f5-tts ...)
tests/CMakeLists.txt                      — add_executable(test-f5-tts ...)
tools/dump_reference.py                   — register "f5-tts" backend + env vars
```

### Models downloaded
```
/mnt/storage/f5-tts/F5TTS_v1_Base/model_1250000.safetensors  (1.3 GB)
/mnt/storage/f5-tts/vocos/config.yaml + pytorch_model.bin      (54 MB)
/mnt/storage/f5-tts/f5-tts-v1-base-f16.gguf                   (1337 MB, all F32)
/mnt/storage/f5-tts/f5-tts-ref.gguf                            (12 MB, reference dump)
/mnt/storage/f5-tts/ref_3s.wav                                 (3s JFK clip for testing)
```

### Diff validation results (against PyTorch reference)

| Stage | cos_min | max_abs | Status |
|-------|---------|---------|--------|
| text_embed | 1.000000 | 9.1e-04 | **PASS** |
| time_embed | 1.000000 | 7.2e-07 | **PASS** |
| input_embed | 1.000000 | 3.2e-05 | **PASS** |
| ode_step_0 | 1.000000 | 0.0 | **PASS** (injected ref noise) |
| dit0_adaln_norm | 1.000000 | 1.7e-05 | **PASS** |
| dit0_post_attn | 0.846 | 9.2e+00 | **FAIL** |
| dit_layer_0 | 0.932 | 9.5e+00 | **FAIL** |

## The one remaining bug

The divergence is **inside the self-attention** of DiT block 0. The AdaLN
output (attention input) matches perfectly, but after QKV projection +
RoPE + flash_attn_ext + O-proj + gating, the result diverges (cos=0.846).

### What's been ruled out
- **F16 weight precision**: all weights converted to F32, same divergence
- **Attention scale**: fixed from 1.0 to 1/sqrt(64)=0.125
- **GELU variant**: ggml_gelu IS tanh-approximate (verified source)
- **Weight loading**: weights match PyTorch exactly (max_abs=0.0)
- **Concatenation order**: cat(x, cond, text) matches perfectly
- **AdaLN broadcast**: norm + norm*scale + shift matches perfectly
- **dump guard for CFG**: fixed (unconditioned forward was overwriting conditioned dumps)

### Most likely causes (in order of probability)

1. **ggml_rope interleave convention mismatch**: x_transformers uses
   `rotate_half` which interleaves as `(-x1, x0, -x3, x2, ...)`.
   ggml_rope with mode=0 SHOULD do the same, but the tensor layout
   (head_dim, n_heads, T) may cause it to rotate wrong pairs.
   **Test**: compute RoPE manually on CPU using the model's `rotary_inv_freq`,
   compare Q values with Python's `apply_rotary_pos_emb` output.
   The Python Q dump is at `/mnt/storage/f5-tts/py_dit0_q_after_rope.bin`
   (in ggml flat layout: `d + h*64 + t*1024`).

2. **ggml_flash_attn_ext output permutation**: after flash_attn, the
   output is permuted back from (head_dim, T, n_heads) to (head_dim,
   n_heads, T) then reshaped to (dim, T). The head interleave in the
   reshape might not match PyTorch's `transpose(1,2).reshape(B,-1,dim)`.
   **Test**: implement CPU-based softmax attention (no flash), compare.

3. **V permutation into flash_attn**: V gets the same RoPE + permute as
   Q and K, but V should NOT have RoPE applied. Looking at the code,
   V DOES get `ggml_reshape_3d` and `ggml_permute` but NOT `ggml_rope`.
   The permute for V might be wrong since V doesn't go through RoPE.
   Actually V is not roped — this is correct. But the permute still
   applies. Check that V's permutation into flash_attn format is correct.

### How to debug

The fastest path to fixing this:

```bash
# 1. Split the DiT block 0 ggml graph into two parts:
#    Part A: AdaLN + QKV projections + RoPE → read back Q, K, V
#    Part B: attention + O-proj + gating + FFN → read back output
#    Compare Part A's Q/K against Python's (already dumped).

# 2. If Q/K match after RoPE, implement attention on CPU:
#    for each head h:
#      scores[t1, t2] = sum_d Q[h,t1,d] * K[h,t2,d] / sqrt(64)
#      weights = softmax(scores)
#      out[h,t1,d] = sum_t2 weights[t1,t2] * V[h,t2,d]
#    Compare against Python's attn_output.

# 3. If Q/K DON'T match, the RoPE is wrong. Implement CPU RoPE:
#    for each head h, position t, dim pair (d0, d1):
#      angle = t * inv_freq[d0/2]
#      q'[d0] = q[d0]*cos(angle) - q[d1]*sin(angle)
#      q'[d1] = q[d0]*sin(angle) + q[d1]*cos(angle)
```

## Architecture reference

F5-TTS v1 Base config:
- dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2
- text_dim=512, text_num_embeds=2546 (2545 vocab + 1 filler)
- 4 ConvNeXtV2 text encoder blocks (text_dim=512, intermediate=1024)
- ConvPositionEmbedding: 2× Conv1d(1024, 1024, k=31, groups=16, pad=15) + Mish
- InputEmbedding: Linear(712→1024) where 712 = 100 (x) + 100 (cond) + 512 (text)
- TimestepEmbedding: SinusoidalPosEmb(256, scale=1000) → Linear(256,1024) → SiLU → Linear(1024,1024)
- DiT block: AdaLN-Zero(6 modulations) → self-attn(RoPE, bidirectional) → gated residual → FFN(GELU_tanh, 2048 inner) → gated residual
- ODE: Euler, 32 steps, EPSS schedule, sway=-1.0, CFG=2.0
- Vocos vocoder: Conv1d(100,512,k7) → LN → 8×ConvNeXt(512,1536,layer_scale) → LN → Linear(512,1026) → split mag/phase → iSTFT(1024,256)

## Test commands

```bash
# Generate reference dump (takes ~2 min on CPU)
F5_TTS_SYN_TEXT="Hello world." F5_TTS_REF_TEXT="Ask not what your country" F5_TTS_SEED=42 \
  python3 tools/dump_reference.py --backend f5-tts \
    --model-dir /mnt/storage/f5-tts \
    --audio /mnt/storage/f5-tts/ref_3s.wav \
    --output /mnt/storage/f5-tts/f5-tts-ref.gguf

# Convert model
python3 models/convert-f5-tts-to-gguf.py \
    --model-dir /mnt/storage/f5-tts \
    --output /mnt/storage/f5-tts/f5-tts-v1-base-f16.gguf

# Build
cd /mnt/storage/whisper.cpp
cmake -B build && cmake --build build -j$(nproc) --target test-f5-tts

# Run diff test (slow: ~30s per ODE step due to CPU ConvPosEmbed)
mkdir -p /mnt/storage/f5-tts/cpp_stages
build/bin/test-f5-tts \
    /mnt/storage/f5-tts/f5-tts-v1-base-f16.gguf \
    "Hello world." /mnt/storage/f5-tts/test_out.wav \
    --ref-gguf /mnt/storage/f5-tts/f5-tts-ref.gguf \
    --ref-text "Ask not what your country" \
    --dump /mnt/storage/f5-tts/cpp_stages --seed 42

# Compare stages (quick Python check)
python3 -c "
import numpy as np, gguf, os
from numpy.linalg import norm
reader = gguf.GGUFReader('/mnt/storage/f5-tts/f5-tts-ref.gguf')
tensors = {t.name: t.data.copy().view(np.float32).flatten() for t in reader.tensors}
for name, D in [('text_embed',512),('time_embed',1024),('input_embed',1024),
    ('dit_layer_0',1024),('dit_layer_21',1024),('dit_output',100),('ode_step_0',100)]:
    ref = tensors.get(name)
    path = f'/mnt/storage/f5-tts/cpp_stages/{name}.bin'
    if not os.path.exists(path) or ref is None: print(f'{name:25s}  SKIP'); continue
    cpp = np.fromfile(path, dtype=np.float32)
    n = min(len(ref), len(cpp)); T = n // D
    arr_r, arr_c = ref[:n].reshape(T,D), cpp[:n].reshape(T,D)
    cos_rows = [np.dot(arr_r[r],arr_c[r])/(norm(arr_r[r])*norm(arr_c[r])+1e-30)
                for r in range(T) if norm(arr_r[r])>1e-6]
    s = 'PASS' if min(cos_rows)>=0.999 else 'FAIL'
    print(f'{name:25s}  cos_min={min(cos_rows):.6f} max_abs={np.max(np.abs(ref[:n]-cpp[:n])):.2e}  {s}')
"
```

## What still needs to be done (in order)

1. **Fix DiT attention** — the one bug (RoPE or flash_attn permutation)
2. **Validate full DiT** — all 22 layers should pass cos ≥ 0.999
3. **Validate ODE solver** — EPSS schedule + Euler + CFG
4. **Implement Vocos vocoder** — Conv1d embed + 8 ConvNeXt + ISTFTHead
5. **Implement mel spectrogram** — vocos-type (torchaudio MelSpectrogram equivalent)
6. **Performance**: move ConvNeXtV2 and ConvPosEmbed from CPU loops to ggml ops
7. **Wire CLI adapter** — `examples/cli/crispasr_backend_f5_tts.cpp`
8. **Wire C API + model registry** — `crispasr_c_api.cpp`, `crispasr_model_registry.cpp`
9. **ASR roundtrip test** — synthesize → whisper transcribe → verify
10. **Push to cohere remote**

## Bugs found and fixed (for reference)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | CFG dump overwrite | unconditioned forward overwrote conditioned dumps | Add `!drop_audio_cond` guard to all step_idx==0 dumps |
| 2 | Attention scale | ggml_flash_attn_ext scale=1.0 instead of 1/sqrt(d_k) | Changed to 1/sqrt(64)=0.125 |
| 3 | F16 input_proj weight | input_proj had F16 weight causing precision loss | Added input_proj to keep_f32 list in converter |
| 4 | ggml_add1 deprecated | scalar add via ggml_add1 deprecated | Replaced with algebraic equivalent: norm + norm*scale |
| 5 | ggml_conv_1d groups | ggml_conv_1d doesn't support groups>1 | Implemented ConvPosEmbed as CPU loops |
| 6 | Text encoder ConvNeXtV2 | ggml shape issues with depthwise conv | Implemented full ConvNeXtV2 on CPU (correct, slow) |
| 7 | std::string::ends_with | C++20 feature not available | Implemented as lambda |
