#!/usr/bin/env python3
"""KugelAudio-0-Open TTS reference dump.

Runs the full KugelAudio TTS pipeline (text -> Qwen2.5-7B LM -> DPM-Solver++
diffusion -> acoustic VAE decoder -> 24kHz audio) and dumps intermediate tensors
to binary files for comparison with the C++ ggml implementation.

Dumped stages (all little-endian float32 unless noted):

  text_token_ids.bin          int32   [T_text]          tokenized text IDs
  lm_embed.bin                float32 [T_text, d_lm]    after token embedding
  lm_hidden_states.bin        float32 [T_text, d_lm]    LM last hidden states
  diffusion_condition.bin     float32 [d_lm]            LM hidden at diffusion pos
  pred_t_emb_step0.bin        float32 [d_lm]            timestep embedding at step 0
  pred_c_step0.bin            float32 [d_lm]            combined c = cond + t_emb
  pred_eps_step0.bin          float32 [vae_dim]          diffusion head output at step 0
  latent_after_diffusion.bin  float32 [vae_dim]          final denoised latent
  scaled_latent.bin           float32 [vae_dim]          after unscaling
  acoustic_decoder_input.bin  float32 [vae_dim, T_lat]   input to decoder (channels-first)
  audio_output.bin            float32 [N_samples]        raw audio PCM
  audio_output.wav            24kHz mono WAV

Key architecture facts (7B TTS model):
  - Qwen2.5-7B LM: d_lm=3584, 28 layers, vocab=152064, GQA 28/4
  - Diffusion head: 4x AdaLN + SwiGLU layers, vae_dim=64
  - Timestep embedding: sinusoidal[256] -> MLP(256->3584->3584)
  - DPM-Solver++ SDE: cosine beta, v-prediction, 20 steps, order 2
  - Acoustic decoder: ConvNeXt transposed, 3200x upsample -> 24kHz
  - Latent unscaling: z_raw = z / scaling_factor - bias_factor

Usage (on Kaggle with T4 GPU):
  python tools/reference_backends/kugelaudio.py \\
      --text "Hello, this is a test." \\
      --output-dir /kaggle/working/kugelaudio_ref \\
      --model kugelaudio/kugelaudio-0-open \\
      --num-steps 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np


# ── Binary dump helpers ─────────────────────────────────────────────────────

def dump_f32(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} float32 ({arr.nbytes} bytes)")


def dump_i32(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr.astype(np.int32))
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} int32 ({arr.nbytes} bytes)")


def dump_wav(path: Path, audio: np.ndarray, sr: int = 24000):
    audio_i16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    print(f"  wrote {path.name}: {len(audio)} samples, {len(audio)/sr:.2f}s @ {sr}Hz")


# ── DPM-Solver++ cosine beta schedule ─────────────────────────────────────

def make_cosine_schedule(num_train_steps: int = 1000):
    """Cosine beta schedule matching diffusers/DPM-Solver++.

    Returns (alphas_cumprod, alpha_t, sigma_t) all shape [num_train_steps].
    """
    def alpha_bar_fn(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_train_steps):
        t1 = i / num_train_steps
        t2 = (i + 1) / num_train_steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
    betas = np.array(betas, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alpha_t = np.sqrt(alphas_cumprod).astype(np.float32)
    sigma_t = np.sqrt(1.0 - alphas_cumprod).astype(np.float32)

    return alphas_cumprod.astype(np.float32), alpha_t, sigma_t


# ── Sinusoidal timestep embedding ─────────────────────────────────────────

def sinusoidal_embed(t: float, dim: int = 256) -> np.ndarray:
    """Sinusoidal embedding matching diffusion_head.py TimestepEmbedder."""
    half = dim // 2
    freqs = np.exp(
        -math.log(10000.0) * np.arange(half, dtype=np.float32) / half
    )
    args = t * freqs
    out = np.concatenate([np.cos(args), np.sin(args)])
    return out.astype(np.float32)


# ── Main pipeline (uses kugelaudio_open package for model loading) ─────────

def run_tts_pipeline(
    model_id: str,
    text: str,
    output_dir: str,
    num_steps: int = 20,
    seed: int = 42,
    voice: str = "default",
    cfg_scale: float = 3.0,
    use_hooks: bool = True,
):
    """Run KugelAudio TTS and dump intermediates.

    If use_hooks=True (default), uses forward hooks on the actual model to
    capture internal tensors. This is more reliable than manual reimplementation
    but requires the full model to be loaded.
    """
    import torch
    from transformers import AutoTokenizer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"=== KugelAudio TTS Reference Dump ===")
    print(f"model:  {model_id}")
    print(f"text:   {text!r}")
    print(f"device: {device}, dtype: {dtype}")
    print(f"steps:  {num_steps}, seed: {seed}, cfg: {cfg_scale}")
    print()

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("loading model...")

    # Try kugelaudio_open package first
    try:
        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor

        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_id, torch_dtype=dtype,
        ).to(device)
        model.eval()

        processor = KugelAudioProcessor.from_pretrained(model_id)

        print(f"  loaded via kugelaudio_open")
    except ImportError:
        raise RuntimeError(
            "kugelaudio_open package required. Install with: "
            "pip install kugelaudio-open"
        )

    # Get config info
    config = model.config
    d_lm = config.decoder_config.hidden_size          # 3584
    n_layers = config.decoder_config.num_hidden_layers  # 28
    vae_dim = config.acoustic_vae_dim                   # 64
    print(f"  config: d_lm={d_lm}, n_layers={n_layers}, vae_dim={vae_dim}")

    # ── 2. Save config summary ──────────────────────────────────────────────
    config_summary = {
        "model_id": model_id,
        "text": text,
        "seed": seed,
        "num_steps": num_steps,
        "cfg_scale": cfg_scale,
        "voice": voice,
        "d_lm": d_lm,
        "n_layers": n_layers,
        "vae_dim": vae_dim,
        "n_heads": config.decoder_config.num_attention_heads,
        "n_kv_heads": config.decoder_config.num_key_value_heads,
        "vocab_size": config.decoder_config.vocab_size,
        "rope_theta": config.decoder_config.rope_theta,
        "rms_norm_eps": config.decoder_config.rms_norm_eps,
        "diffusion_steps": config.diffusion_head_config.ddpm_num_inference_steps,
        "diffusion_train_steps": config.diffusion_head_config.ddpm_num_steps,
        "head_layers": config.diffusion_head_config.head_layers,
        "head_ffn_ratio": config.diffusion_head_config.head_ffn_ratio,
        "prediction_type": config.diffusion_head_config.prediction_type,
        "algorithm_type": config.diffusion_head_config.ddpm_algorithm_type,
        "beta_schedule": config.diffusion_head_config.ddpm_beta_schedule,
        "speech_scaling_factor": float(model.speech_scaling_factor.cpu()),
        "speech_bias_factor": float(model.speech_bias_factor.cpu()),
    }

    with open(out_path / "config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    print(f"  wrote config.json")

    # ── 3. Prepare inputs ──────────────────────────────────────────────────
    print("preparing inputs...")
    inputs = processor(text=text, voice=voice, return_tensors="pt")
    text_ids = inputs["text_ids"].to(device)
    speech_input_mask = inputs.get("speech_input_mask")
    if speech_input_mask is not None:
        speech_input_mask = speech_input_mask.to(device)
    voice_cache = inputs.get("voice_cache")

    token_ids_np = text_ids[0].cpu().numpy().astype(np.int32)
    print(f"  {len(token_ids_np)} tokens")
    dump_i32(out_path / "text_token_ids.bin", token_ids_np)

    if speech_input_mask is not None:
        mask_np = speech_input_mask[0].cpu().numpy().astype(np.int32)
        dump_i32(out_path / "speech_input_mask.bin", mask_np)
        print(f"  voice tokens: {mask_np.sum()}")

    # ── 4. Register forward hooks for intermediate capture ─────────────────
    captures = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captures[name] = output[0].detach().cpu().float()
            elif hasattr(output, "last_hidden_state"):
                captures[name] = output.last_hidden_state.detach().cpu().float()
            else:
                captures[name] = output.detach().cpu().float()
        return hook_fn

    hooks = []

    # Hook: LM embedding output
    embed_module = model.model.get_input_embeddings()
    hooks.append(embed_module.register_forward_hook(make_hook("lm_embed")))

    # Hook: LM final hidden states (language_model output)
    hooks.append(
        model.model.language_model.register_forward_hook(make_hook("lm_output"))
    )

    # Hook: Diffusion head output
    hooks.append(
        model.model.prediction_head.register_forward_hook(make_hook("diffusion_head_output"))
    )

    # Hook: Timestep embedder
    hooks.append(
        model.model.prediction_head.t_embedder.register_forward_hook(make_hook("t_embedder"))
    )

    # Hook: Diffusion head noisy_images_proj
    hooks.append(
        model.model.prediction_head.noisy_images_proj.register_forward_hook(
            make_hook("noisy_images_proj")
        )
    )

    # Hook: Diffusion head cond_proj
    hooks.append(
        model.model.prediction_head.cond_proj.register_forward_hook(
            make_hook("cond_proj")
        )
    )

    # Hook: Each diffusion head layer
    for i, layer in enumerate(model.model.prediction_head.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"diffusion_layer_{i}")))

    # Hook: Final diffusion layer
    hooks.append(
        model.model.prediction_head.final_layer.register_forward_hook(
            make_hook("diffusion_final_layer")
        )
    )

    # Hook: Acoustic connector
    hooks.append(
        model.model.acoustic_connector.register_forward_hook(make_hook("acoustic_connector"))
    )

    # Hook: Acoustic decoder
    if hasattr(model.model.acoustic_tokenizer, "decoder"):
        hooks.append(
            model.model.acoustic_tokenizer.decoder.register_forward_hook(
                make_hook("acoustic_decoder_output")
            )
        )

    # ── 5. Run generation ─────────────────────────────────────────────────
    print("running generation...")
    model.set_ddpm_inference_steps(num_steps)

    # We need to intercept the diffusion loop. Override sample_speech_tokens
    # to capture per-step intermediates.
    original_sample = model.sample_speech_tokens
    diffusion_steps_captured = []

    @torch.no_grad()
    def sample_speech_tokens_instrumented(condition, neg_condition, cfg_scale_=3.0):
        """Instrumented version that captures per-step diffusion state."""
        scheduler = model.model.noise_scheduler
        scheduler.set_timesteps(model.ddpm_inference_steps)

        captures["diffusion_condition"] = condition.detach().cpu().float()
        captures["diffusion_neg_condition"] = neg_condition.detach().cpu().float()

        if cfg_scale_ == 1.0:
            speech = torch.randn(condition.shape[0], config.acoustic_vae_dim).to(condition)
            captures["diffusion_noise_init"] = speech.detach().cpu().float()

            for step_idx, t in enumerate(scheduler.timesteps):
                # Clear previous hook captures for this step
                for k in list(captures.keys()):
                    if k.startswith("step_"):
                        del captures[k]

                eps = model.model.prediction_head(
                    speech, t.repeat(speech.shape[0]).to(speech), condition=condition
                )

                step_data = {
                    "timestep": int(t),
                    "noisy_input": speech.detach().cpu().float().numpy(),
                    "model_output": eps.detach().cpu().float().numpy(),
                }

                result = scheduler.step(eps, t, speech)
                speech = result.prev_sample

                step_data["denoised"] = speech.detach().cpu().float().numpy()
                diffusion_steps_captured.append(step_data)

            captures["latent_after_diffusion"] = speech.detach().cpu().float()
            return speech

        # With CFG
        combined_condition = torch.cat([condition, neg_condition], dim=0).to(
            model.model.prediction_head.device
        )
        speech = torch.randn(combined_condition.shape[0], config.acoustic_vae_dim).to(
            combined_condition
        )
        captures["diffusion_noise_init"] = speech[:len(speech)//2].detach().cpu().float()

        for step_idx, t in enumerate(scheduler.timesteps):
            half = speech[:len(speech) // 2]
            combined = torch.cat([half, half], dim=0)

            eps = model.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined),
                condition=combined_condition
            )

            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale_ * (cond_eps - uncond_eps)
            eps_combined = torch.cat([half_eps, half_eps], dim=0)

            step_data = {
                "timestep": int(t),
                "noisy_input": half.detach().cpu().float().numpy(),
                "model_output_cond": cond_eps.detach().cpu().float().numpy(),
                "model_output_uncond": uncond_eps.detach().cpu().float().numpy(),
                "model_output_cfg": half_eps.detach().cpu().float().numpy(),
            }

            result = scheduler.step(eps_combined, t, speech)
            speech = result.prev_sample

            step_data["denoised"] = speech[:len(speech)//2].detach().cpu().float().numpy()
            diffusion_steps_captured.append(step_data)

        final = speech[:len(speech) // 2]
        captures["latent_after_diffusion"] = final.detach().cpu().float()
        return final

    model.sample_speech_tokens = sample_speech_tokens_instrumented

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=2048,
            do_sample=False,
            show_progress=True,
        )

    # Restore original
    model.sample_speech_tokens = original_sample

    # ── 6. Dump captured intermediates ────────────────────────────────────
    print("\ndumping intermediates...")

    # Token embeddings
    if "lm_embed" in captures:
        dump_f32(out_path / "lm_embed.bin", captures["lm_embed"][0].numpy())

    # LM hidden states
    if "lm_output" in captures:
        dump_f32(out_path / "lm_hidden_states.bin", captures["lm_output"][0].numpy())

    # Diffusion condition
    if "diffusion_condition" in captures:
        dump_f32(out_path / "diffusion_condition.bin",
                 captures["diffusion_condition"].numpy())
    if "diffusion_neg_condition" in captures:
        dump_f32(out_path / "diffusion_neg_condition.bin",
                 captures["diffusion_neg_condition"].numpy())

    # Initial noise
    if "diffusion_noise_init" in captures:
        dump_f32(out_path / "diffusion_noise_init.bin",
                 captures["diffusion_noise_init"].numpy())

    # Per-step diffusion intermediates
    for step_data in diffusion_steps_captured:
        step_t = step_data["timestep"]
        prefix = f"diffusion_step_t{step_t}"
        dump_f32(out_path / f"{prefix}_noisy_input.bin",
                 step_data["noisy_input"])
        if "model_output" in step_data:
            dump_f32(out_path / f"{prefix}_model_output.bin",
                     step_data["model_output"])
        if "model_output_cfg" in step_data:
            dump_f32(out_path / f"{prefix}_model_output_cond.bin",
                     step_data["model_output_cond"])
            dump_f32(out_path / f"{prefix}_model_output_uncond.bin",
                     step_data["model_output_uncond"])
            dump_f32(out_path / f"{prefix}_model_output_cfg.bin",
                     step_data["model_output_cfg"])
        dump_f32(out_path / f"{prefix}_denoised.bin",
                 step_data["denoised"])

    # Final latent
    if "latent_after_diffusion" in captures:
        dump_f32(out_path / "latent_after_diffusion.bin",
                 captures["latent_after_diffusion"].numpy())

    # Acoustic connector output
    if "acoustic_connector" in captures:
        dump_f32(out_path / "acoustic_connector_out.bin",
                 captures["acoustic_connector"].numpy())

    # Acoustic decoder output (raw PCM)
    if "acoustic_decoder_output" in captures:
        dump_f32(out_path / "acoustic_decoder_raw.bin",
                 captures["acoustic_decoder_output"].numpy())

    # Timestep/condition embeddings from first diffusion call
    if "t_embedder" in captures:
        dump_f32(out_path / "pred_t_emb.bin", captures["t_embedder"].numpy())
    if "cond_proj" in captures:
        dump_f32(out_path / "pred_cond_proj.bin", captures["cond_proj"].numpy())
    if "noisy_images_proj" in captures:
        dump_f32(out_path / "pred_noisy_proj.bin",
                 captures["noisy_images_proj"].numpy())
    if "diffusion_head_output" in captures:
        dump_f32(out_path / "pred_output.bin",
                 captures["diffusion_head_output"].numpy())

    # Per-layer diffusion head outputs
    for i in range(config.diffusion_head_config.head_layers):
        key = f"diffusion_layer_{i}"
        if key in captures:
            dump_f32(out_path / f"diffusion_layer_{i}_out.bin",
                     captures[key].numpy())

    # ── 7. Save audio output ──────────────────────────────────────────────
    print("\nsaving audio...")
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0].cpu().float().numpy()
        dump_f32(out_path / "audio_output.bin", audio)
        dump_wav(out_path / "audio_output.wav", audio, sr=24000)
    else:
        print("  WARNING: no audio output generated!")

    # ── 8. Save generated token sequence ──────────────────────────────────
    if outputs.sequences is not None:
        gen_ids = outputs.sequences[0].cpu().numpy().astype(np.int32)
        dump_i32(out_path / "generated_token_ids.bin", gen_ids)
        print(f"  generated {len(gen_ids)} tokens total")

    # ── 9. Save scaling factors ──────────────────────────────────────────
    scaling = model.speech_scaling_factor.cpu().float().item()
    bias = model.speech_bias_factor.cpu().float().item()
    print(f"\n  speech_scaling_factor = {scaling}")
    print(f"  speech_bias_factor   = {bias}")

    # Save DPM-Solver schedule for validation
    alphas_cumprod, alpha_t_arr, sigma_t_arr = make_cosine_schedule(
        config.diffusion_head_config.ddpm_num_steps
    )
    dump_f32(out_path / "alphas_cumprod.bin", alphas_cumprod)
    dump_f32(out_path / "alpha_t.bin", alpha_t_arr)
    dump_f32(out_path / "sigma_t.bin", sigma_t_arr)

    # ── Cleanup hooks ─────────────────────────────────────────────────────
    for h in hooks:
        h.remove()

    print(f"\n=== Done. Output in {out_path} ===")
    return out_path


# ── Tensor-name mapping dump (for GGUF converter) ────────────────────────────

def dump_tensor_map(model_id: str, output_dir: str):
    """Dump the complete tensor name mapping for GGUF conversion."""
    from safetensors import safe_open

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the index to get tensor names
    from huggingface_hub import hf_hub_download
    index_file = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    # Group tensors by component
    components = {}
    for name, shard in index["weight_map"].items():
        parts = name.split(".")
        if parts[0] == "model" and len(parts) > 1:
            comp = parts[1]
        elif parts[0] == "lm_head":
            comp = "lm_head"
        else:
            comp = parts[0]

        if comp not in components:
            components[comp] = []
        components[comp].append(name)

    # Also get shapes from one shard
    shards = sorted(set(index["weight_map"].values()))
    shapes = {}
    for shard_name in shards:
        shard_file = hf_hub_download(model_id, shard_name)
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                shapes[key] = list(f.get_tensor(key).shape)

    # Write mapping
    mapping = {
        "total_tensors": len(index["weight_map"]),
        "total_size_bytes": index["metadata"]["total_size"],
        "components": {},
    }
    for comp in sorted(components.keys()):
        names = sorted(components[comp])
        mapping["components"][comp] = {
            "count": len(names),
            "tensors": {n: {"shape": shapes.get(n, []), "shard": index["weight_map"][n]}
                        for n in names},
        }

    with open(out_path / "tensor_map.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"wrote tensor_map.json: {len(index['weight_map'])} tensors in {len(components)} components")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KugelAudio TTS reference dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--text", default="Hello, this is a test of the speech synthesis system.",
                        help="Text to synthesize")
    parser.add_argument("--output-dir", default="/tmp/kugelaudio_ref",
                        help="Directory for output files")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of diffusion inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--voice", default="default",
                        help="Voice name from voices.json")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--dump-tensor-map", action="store_true",
                        help="Only dump tensor name mapping (no inference)")

    args = parser.parse_args()

    if args.dump_tensor_map:
        dump_tensor_map(args.model, args.output_dir)
    else:
        run_tts_pipeline(
            model_id=args.model,
            text=args.text,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            seed=args.seed,
            voice=args.voice,
            cfg_scale=args.cfg_scale,
        )


if __name__ == "__main__":
    main()
