#!/usr/bin/env python3
"""
CrispASR regression transcript capture — Kaggle GPU kernel.

Builds CrispASR, downloads each backend's GGUF from HF, runs on JFK 11s,
captures the transcript, and writes results to /kaggle/working/transcripts.json.

Usage: push via `python -m kaggle kernels push -p .`
"""

import json, os, subprocess, sys, time, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"

# ── Harness bootstrap ──
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()

progress_file = WORK / "progress.txt"

def log(msg):
    print(msg, flush=True)
    with open(progress_file, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

# ── HF auth ──
kh.setup_hf_token()

# ── Build ──
print("=== Building CrispASR ===", flush=True)
# Write an early marker so we know the script started
with open(WORK / "transcripts.json", "w") as f:
    json.dump({"_status": "building"}, f)

try:
    kh.install_build_toolchain()
except Exception as e:
    print(f"install_build_toolchain failed: {e}", flush=True)

# Fallback: ensure cmake + ninja are available via pip
if not shutil.which("cmake"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cmake"], check=False)
if not shutil.which("ninja"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)

build_dir = _CRISPASR_DIR / "build"
cmake_args = ["-DCMAKE_BUILD_TYPE=Release"]
if shutil.which("ninja"):
    cmake_args += ["-G", "Ninja"]
if shutil.which("ccache"):
    cmake_args += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                   "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]

try:
    log("Configure...")
    subprocess.check_call(["cmake", "-B", str(build_dir)] + cmake_args,
                          cwd=str(_CRISPASR_DIR))
    log("Build (j2)...")
    subprocess.check_call(["cmake", "--build", str(build_dir), "-j2"],
                          cwd=str(_CRISPASR_DIR))
    log("Build OK")
except Exception as e:
    log(f"Build failed: {e}")
    with open(WORK / "transcripts.json", "w") as f:
        json.dump({"_build_error": str(e)}, f)
    sys.exit(1)

CRISPASR = str(build_dir / "bin" / "crispasr")
JFK_WAV = str(_CRISPASR_DIR / "samples" / "jfk.wav")
MODEL_DIR = WORK / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Backends to capture ──
# Each: (name, backend_id, hf_repo, hf_file, extra_args, needs_gpu)
# Sorted by size so we can bail early if disk fills up.
BACKENDS = [
    # Tiny / small — CPU OK
    # LID / speaker-verification backends are skipped here — they need
    # special invocation (--lid-backend, --diarize-embedder) not -f.
    # titanet, silero-lid, firered-lid, ecapa-lid, lid-cld3, lid-fasttext
    # are captured separately via the crispasr-lid CLI.
    # Medium ASR — may need GPU for speed
    ("qwen3-asr-0.6b", "qwen3", "cstr/qwen3-asr-0.6b-GGUF", "qwen3-asr-0.6b-q4_k.gguf", [], False),
    ("omniasr-ctc-1b-v2", "omniasr", "cstr/omniASR-CTC-1B-v2-GGUF", "omniasr-ctc-1b-v2-q4_k.gguf", [], False),
    ("kyutai-stt-1b", "kyutai-stt", "cstr/kyutai-stt-1b-GGUF", "kyutai-stt-1b-q4_k.gguf", [], False),
    ("funasr-nano", "funasr", "cstr/funasr-nano-GGUF", "funasr-nano-2512-q8_0.gguf", [], False),
    ("mini-omni2", "mini-omni2", "cstr/mini-omni2-GGUF", "mini-omni2-q4_k.gguf", [], False),
    ("omniasr-llm-300m", "omniasr", "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf", [], False),
    ("parakeet-rnnt-0.6b", "parakeet", "cstr/parakeet-rnnt-0.6b-GGUF", "parakeet-rnnt-0.6b-q4_k.gguf", [], False),
    ("parakeet-tdt-1.1b", "parakeet", "cstr/parakeet-tdt-1.1b-GGUF", "parakeet-tdt-1.1b-q8_0.gguf", [], False),
    ("parakeet-tdt_ctc-1.1b", "parakeet", "cstr/parakeet-tdt_ctc-1.1b-GGUF", "parakeet-tdt_ctc-1.1b-q8_0.gguf", [], False),
    ("funasr-mlt-nano", "fun-asr-mlt-nano", "cstr/funasr-mlt-nano-GGUF", "funasr-mlt-nano-2512-f16.gguf", [], False),
    ("parakeet-rnnt-1.1b", "parakeet", "cstr/parakeet-rnnt-1.1b-GGUF", "parakeet-rnnt-1.1b-q4_k.gguf", [], False),
    # Large — skip for CPU kernel (need separate GPU kernel)
    # voxtral, voxtral4b, gemma4, granite-plus, granite-nar,
    # moss-audio, mimo-asr, vibevoice, kugelaudio
]

# TTS backends (capture via ASR roundtrip is different — skip for now)
# voxcpm2, indextts, chatterbox-turbo need voice ref + codec companion files

results = {}

def download_model(repo, filename):
    """Download a single file from HF, return local path."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=filename,
                           cache_dir=str(MODEL_DIR / ".hf_cache"),
                           local_dir=str(MODEL_DIR))
    return str(MODEL_DIR / filename)

def run_backend(name, backend_id, model_path, extra_args, timeout=300):
    """Run crispasr on JFK and return the transcript."""
    cmd = [CRISPASR, "--backend", backend_id, "-m", model_path,
           "-f", JFK_WAV, "-np"] + extra_args
    log(f"  CMD: {' '.join(cmd[:8])}...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            log(f"  FAIL rc={r.returncode}: {r.stderr[-200:]}")
            return None
        # Transcript is last non-empty line of stdout
        lines = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
        if lines:
            return lines[-1]
        return ""
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        log(f"  ERROR: {e}")
        return None

# ── Main loop ──
log(f"=== Regression capture: {len(BACKENDS)} backends ===")
for i, (name, backend_id, hf_repo, hf_file, extra_args, needs_gpu) in enumerate(BACKENDS):
    log(f"\n[{i+1}/{len(BACKENDS)}] {name} ({hf_file})")

    # Check disk space
    disk_free_gb = shutil.disk_usage(str(WORK)).free / (1024**3)
    if disk_free_gb < 2.0:
        log(f"  SKIP — only {disk_free_gb:.1f} GB free")
        results[name] = {"status": "skip_disk", "transcript": None}
        continue

    # Download
    try:
        model_path = download_model(hf_repo, hf_file)
        log(f"  Downloaded: {hf_file}")
    except Exception as e:
        log(f"  DOWNLOAD FAILED: {e}")
        results[name] = {"status": "download_failed", "transcript": None}
        continue

    # Run
    t0 = time.time()
    transcript = run_backend(name, backend_id, model_path, extra_args,
                             timeout=600 if needs_gpu else 300)
    elapsed = time.time() - t0

    if transcript is not None:
        results[name] = {
            "status": "ok",
            "transcript": transcript,
            "elapsed_s": round(elapsed, 1),
            "backend_id": backend_id,
        }
        log(f"  OK ({elapsed:.1f}s): {transcript[:80]}...")
    else:
        results[name] = {"status": "failed", "transcript": None, "elapsed_s": round(elapsed, 1)}

    # Clean up model to free disk for next backend
    model_file = MODEL_DIR / hf_file
    if model_file.exists():
        model_file.unlink()
        log(f"  Cleaned: {hf_file}")

    # Write incremental results
    with open(WORK / "transcripts.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ── Summary ──
log(f"\n=== SUMMARY ===")
ok = sum(1 for r in results.values() if r["status"] == "ok")
fail = sum(1 for r in results.values() if r["status"] == "failed")
skip = sum(1 for r in results.values() if r["status"].startswith("skip"))
dl_fail = sum(1 for r in results.values() if r["status"] == "download_failed")
log(f"OK: {ok}  FAILED: {fail}  SKIP: {skip}  DL_FAIL: {dl_fail}  TOTAL: {len(results)}")

# Write final results
with open(WORK / "transcripts.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Wrote transcripts.json")
