#!/usr/bin/env python3
"""
Parakeet-Unified-EN-0.6B: convert NeMo 2.x .nemo (zip format) to GGUF.

NeMo 2.x uses zip (not tar) with model_weights/data.pkl + numbered storage files.
No model_config.yaml — hparams must be inferred from tensor shapes.
"""
import json, os, subprocess, sys, time, traceback, shutil, zipfile, pickle, io
from pathlib import Path

WORK = Path("/kaggle/working")
results = {}

def log(msg):
    print(msg, flush=True)
    try:
        with open(WORK / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

def save():
    try:
        (WORK / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception:
        pass

def main():
    global results
    save()
    log("=== Parakeet-Unified — direct zip extraction ===")

    # HF token
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            tok = open(p).read().strip()
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
            break

    # Clone CrispASR
    cdir = Path("/tmp/CrispASR")
    if not cdir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(cdir)])

    # Build
    log("Building CrispASR...")
    subprocess.run("apt-get update -qq && apt-get install -y cmake ninja-build g++ 2>/dev/null || true",
                   shell=True, capture_output=True)
    bdir = cdir / "build"
    subprocess.check_call(["cmake", "-G", "Ninja", "-B", str(bdir),
                          "-DCMAKE_BUILD_TYPE=Release"], cwd=str(cdir))
    subprocess.check_call(["cmake", "--build", str(bdir), "-j2"], cwd=str(cdir))
    log("Build OK")

    # Download .nemo
    log("Downloading .nemo...")
    from huggingface_hub import hf_hub_download
    nemo_path = hf_hub_download("nvidia/parakeet-unified-en-0.6b",
                                 "parakeet-unified-en-0.6b.nemo",
                                 cache_dir=str(WORK / ".hf"))
    log(f"Downloaded: {os.path.getsize(nemo_path)/(1024*1024):.0f} MB")

    # Step 1: Extract state dict from NeMo 2.x zip format
    log("Extracting state dict from NeMo 2.x zip...")
    import torch

    class NeMo2Unpickler(pickle.Unpickler):
        """Load NeMo 2.x data.pkl with lazy storage resolution."""
        def __init__(self, *args, zf=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._zf = zf

        def persistent_load(self, saved_id):
            if isinstance(saved_id, tuple) and len(saved_id) >= 4:
                if len(saved_id) == 5:
                    _, stype, key, device, numel = saved_id
                else:
                    stype, key, device, numel = saved_id
                # Load actual data from zip
                data_path = f"model_weights/data/{key}"
                raw = self._zf.read(data_path)
                dtype = torch.float32
                if 'Half' in str(stype) or 'float16' in str(stype):
                    dtype = torch.float16
                elif 'BFloat16' in str(stype) or 'bfloat16' in str(stype):
                    dtype = torch.bfloat16
                nbytes = numel * torch.tensor([], dtype=dtype).element_size()
                buf = torch.frombuffer(bytearray(raw[:nbytes]), dtype=dtype)
                return buf.storage()
            return saved_id

    with zipfile.ZipFile(nemo_path) as zf:
        pkl_data = zf.read("model_weights/data.pkl")
        buf = io.BytesIO(pkl_data)
        sd = NeMo2Unpickler(buf, zf=zf).load()

    log(f"State dict: {len(sd)} keys")
    results["n_keys"] = len(sd)

    # Show key tensors
    for k in sorted(sd.keys()):
        if any(x in k for x in ['embed', 'joint', 'pre_encode.out', 'pre_encode.conv.0']):
            log(f"  {k}: {list(sd[k].shape)} {sd[k].dtype}")
    save()

    # Step 2: Extract tokenizer from .nemo
    log("Extracting tokenizer...")
    spm_bytes = None
    with zipfile.ZipFile(nemo_path) as zf:
        for n in zf.namelist():
            if n.endswith("_tokenizer.model") or n.endswith("tokenizer.model"):
                spm_bytes = zf.read(n)
                log(f"  Tokenizer: {n} ({len(spm_bytes)} bytes)")
                break

    if not spm_bytes:
        log("  No tokenizer found in .nemo — downloading from HF...")
        # parakeet-unified might have tokenizer separately
        try:
            tok_path = hf_hub_download("nvidia/parakeet-unified-en-0.6b",
                                       "tokenizer.model",
                                       cache_dir=str(WORK / ".hf"))
            spm_bytes = open(tok_path, "rb").read()
            log(f"  Downloaded tokenizer: {len(spm_bytes)} bytes")
        except Exception as e:
            log(f"  Tokenizer download failed: {e}")

    # Step 3: Infer hparams from tensor shapes
    log("Inferring hparams...")
    d_model = sd["encoder.layers.0.self_attn.linear_q.weight"].shape[0]
    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.") and k.split(".")[2].isdigit()) + 1
    vocab_size = sd["decoder.prediction.embed.weight"].shape[0]
    pred_hidden = sd["decoder.prediction.dec_rnn.lstm.weight_ih_l0"].shape[1]
    joint_hidden = sd["joint.joint_net.2.weight"].shape[1]
    n_heads = d_model // (sd["encoder.layers.0.self_attn.linear_q.weight"].shape[1] // (d_model // 8))  # estimate

    results["hparams"] = {
        "d_model": d_model, "n_layers": n_layers, "vocab_size": vocab_size,
        "pred_hidden": pred_hidden, "joint_hidden": joint_hidden,
    }
    log(f"  d_model={d_model} n_layers={n_layers} vocab={vocab_size}")
    log(f"  pred_hidden={pred_hidden} joint_hidden={joint_hidden}")
    save()

    # Step 4: Try the converter with the extracted state dict
    # Save as a standard torch checkpoint that the converter can read
    log("Saving as standard checkpoint...")
    ckpt_path = WORK / "parakeet-unified-weights.pt"
    torch.save(sd, str(ckpt_path))
    log(f"  Checkpoint: {os.path.getsize(ckpt_path)/(1024*1024):.0f} MB")

    # Free the state dict from memory
    del sd
    import gc; gc.collect()

    # Step 5: Run converter with --weights-only flag
    log("Running converter...")
    converter = str(cdir / "models" / "convert-parakeet-to-gguf.py")
    out_f16 = str(WORK / "parakeet-unified-en-0.6b-f16.gguf")

    r = subprocess.run([sys.executable, converter,
                       "--weights", str(ckpt_path),
                       "--output", out_f16],
                      capture_output=True, text=True, timeout=600)
    results["converter_rc"] = r.returncode
    results["converter_stderr"] = r.stderr[-1000:]
    log(f"  Converter rc={r.returncode}")
    if r.returncode != 0:
        log(f"  stderr: {r.stderr[-500:]}")
        # The converter may not have --weights flag. Let me check.
        results["note"] = "Converter may need patching for standalone weights file"
    else:
        results["gguf_size_mb"] = round(os.path.getsize(out_f16)/(1024*1024), 1)
        log(f"  GGUF: {results['gguf_size_mb']} MB")

    save()
    log("\nDONE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        results["_crash"] = str(e)
        results["_tb"] = traceback.format_exc()
        save()
        log(f"CRASH: {e}")
        traceback.print_exc()

# Cleanup
import shutil
for p in [Path("/tmp/CrispASR"), WORK / "models", WORK / ".hf"]:
    shutil.rmtree(str(p), ignore_errors=True)
for f in WORK.glob("*.pt"):
    f.unlink(missing_ok=True)
for f in WORK.glob("*.gguf"):
    f.unlink(missing_ok=True)
