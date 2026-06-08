"""
CrispASR — Zonos TTS GPU end-to-end test (PLAN #130)

Tests the full Zonos pipeline on GPU:
  1. CUDA build of crispasr-cli
  2. Download Zonos AR GGUF + DAC codec GGUF + parakeet ASR
  3. Generate a pre-computed speaker embedding (from jfk.wav via Python)
  4. Synthesize with Zonos -> WAV
  5. ASR roundtrip the WAV through parakeet
  6. Report: audio produced, ASR text, timing

Two GGUFs:
  - cstr/zonos-v0.1-transformer-GGUF/zonos-v0.1-transformer-q4_k.gguf (~900 MB)
  - cstr/dac-44khz-GGUF/dac-44khz-f16.gguf (~104 MB)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
RESULTS = WORK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git"
)
TTS_TEXT = "Please call Stella. Ask her to bring these things with her from the store."


def run(cmd, check=True, env=None, timeout=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r


# ── Clone + CUDA build ──────────────────────────────────────────────
print(f"[start] ref={CRISPASR_REF}", flush=True)
if REPO.exists():
    import shutil
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
     "--recursive", CRISPASR_REPO, str(REPO)])

sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh

kh.init_progress()
kh.resolve_hf_token()

# espeak-ng is required for proper IPA phonemization
run(["apt-get", "update", "-qq"], check=False)
run(["apt-get", "install", "-y", "--no-install-recommends", "espeak-ng"], check=False)

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
kh.step("cloned", sha=sha, ref=CRISPASR_REF)

run(["nvidia-smi", "-L"])
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
kh.step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = (
    ["cmake", "-S", str(REPO), "-B", str(BUILD),
     "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
     "-DCRISPASR_BUILD_TESTS=OFF"]
    + kh.cuda_build_flags(arch)
    + kh.cache_and_link_flags()
)
run(cmake_args)
kh.step("cmake_done")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli"
        f" -j{kh.safe_build_jobs(gpu=True)}"
    )

CLI = BUILD / "bin" / "crispasr"
if not CLI.exists():
    cands = [c for c in BUILD.rglob("crispasr")
             if c.is_file() and os.access(c, os.X_OK)]
    assert cands, "crispasr binary not found after build"
    CLI = cands[0]
os.environ["LD_LIBRARY_PATH"] = (
    f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
kh.step("build_done", cli=str(CLI))

# ── Download models ─────────────────────────────────────────────────
kh.step("downloading_models")
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
    )
    from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN")
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

zonos_model = Path(hf_hub_download(
    "cstr/zonos-v0.1-transformer-GGUF",
    "zonos-v0.1-transformer-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
dac_codec = Path(hf_hub_download(
    "cstr/dac-44khz-GGUF",
    "dac-44khz-f16.gguf",
    cache_dir=str(MODELS), token=token,
))
asr_model = Path(hf_hub_download(
    "cstr/parakeet-tdt-0.6b-v2-GGUF",
    "parakeet-tdt-0.6b-v2-q4_k.gguf",
    cache_dir=str(MODELS), token=token,
))
kh.step("models_downloaded")

# ── Download pre-computed speaker embedding ─────────────────────────
# Real 128-d LDA speaker embedding extracted from JFK audio via Zonos's
# ResNet293 encoder. Random embeddings produce garbage output.
kh.step("downloading_speaker_embedding")
spk_emb_path = Path(hf_hub_download(
    "cstr/zonos-v0.1-transformer-GGUF",
    "jfk_speaker_emb.bin",
    cache_dir=str(MODELS), token=token,
))
kh.step("speaker_embedding_done")

# ── Synthesize ──────────────────────────────────────────────────────
kh.step("synthesize.start")
out_wav = WORK / "zonos_output.wav"
if out_wav.exists():
    out_wav.unlink()

# Set speaker embedding path via env var
env = {
    "ZONOS_SPEAKER_EMB_PATH": str(spk_emb_path),
}

cmd = [
    str(CLI), "--backend", "zonos-tts",
    "-m", str(zonos_model),
    "--codec-model", str(dac_codec),
    "--tts", TTS_TEXT,
    "--tts-output", str(out_wav),
    "--seed", "42",
    "-v",
]
t0 = time.time()
try:
    r = subprocess.run(
        cmd, env={**os.environ, **env},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=300,
    )
    rc, stdout, stderr = r.returncode, r.stdout, r.stderr
except subprocess.TimeoutExpired as ex:
    rc = -1
    stdout = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    stderr = (ex.stderr or b"").decode(errors="replace") if isinstance(ex.stderr, bytes) else (ex.stderr or "")
elapsed = round(time.time() - t0, 1)
combined = stdout + "\n" + stderr
(RESULTS / "zonos_log.txt").write_text(combined)

wav_exists = out_wav.exists() and out_wav.stat().st_size > 1000
wav_size = out_wav.stat().st_size if out_wav.exists() else 0

print(f"\n{'='*64}", flush=True)
print(f"Zonos TTS: rc={rc}  elapsed={elapsed}s  wav={'OK' if wav_exists else 'MISSING'}  size={wav_size}", flush=True)

# Always print Zonos diagnostic lines (phoneme count, AR steps, DAC decode)
print("--- zonos diagnostics ---", flush=True)
for ln in combined.splitlines():
    if any(k in ln for k in ["zonos_tts:", "crispasr[zonos", "CONSENT", "espeak"]):
        print(f"  {ln.strip()}", flush=True)

if rc != 0:
    print("--- stderr tail ---", flush=True)
    for ln in combined.splitlines()[-30:]:
        print(f"  {ln}", flush=True)

kh.step("synthesize.done", rc=rc, elapsed=elapsed, wav_ok=wav_exists, wav_size=wav_size)

# ── ASR roundtrip ───────────────────────────────────────────────────
asr_text = ""
if wav_exists:
    kh.step("asr.start")
    out_stem = WORK / "asr_zonos"
    asr_cmd = [
        str(CLI), "--backend", "parakeet",
        "-m", str(asr_model),
        "-f", str(out_wav),
        "-of", str(out_stem), "-otxt",
        "--no-prints",
    ]
    try:
        r = subprocess.run(asr_cmd, env=os.environ, capture_output=True, text=True, timeout=120)
        txt_path = out_stem.with_suffix(".txt")
        asr_text = txt_path.read_text().strip() if txt_path.exists() and txt_path.stat().st_size > 0 else ""
    except subprocess.TimeoutExpired:
        pass
    kh.step("asr.done", chars=len(asr_text))

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*64}", flush=True)
print(f"SUMMARY — Zonos TTS GPU test — {sha[:8]}", flush=True)
print(f"{'='*64}", flush=True)
print(f"  Synthesis: rc={rc}  wav={'OK' if wav_exists else 'FAIL'}  {wav_size} bytes  {elapsed}s", flush=True)
print(f"  ASR roundtrip: {asr_text[:150]!r}", flush=True)

if wav_exists and asr_text:
    print(f"\n  VERDICT: Zonos TTS produces audio and ASR roundtrip yields text.", flush=True)
elif wav_exists:
    print(f"\n  VERDICT: Zonos TTS produces audio but ASR roundtrip empty.", flush=True)
elif rc == 0:
    print(f"\n  VERDICT: Zonos TTS ran but produced no audio.", flush=True)
else:
    print(f"\n  VERDICT: Zonos TTS failed (rc={rc}).", flush=True)

kh.step("summary", wav_ok=wav_exists, asr_len=len(asr_text), sha=sha)
kh._push_progress_to_hf(force=True)
kh.step("script.end")
