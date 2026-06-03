# %% [markdown]
# # CrispASR — MOSS-Audio-4B-Instruct F16 GGUF conversion + reference dump
#
# Converts `OpenMOSS-Team/MOSS-Audio-4B-Instruct` (3-shard safetensors,
# ~10 GB bf16) to F16 GGUF on Kaggle GPU notebook and uploads to
# `cstr/MOSS-Audio-4B-Instruct-GGUF`.
#
# Also runs the Python reference dump (tools/dump_reference.py --backend
# moss-audio) on samples/jfk.wav to produce the stage-capture GGUF that
# crispasr-diff validates the C++ runtime against.
#
# GPU notebook: T4 16 GB VRAM, 30 GB system RAM — enough for the F16
# model in memory and the reference forward pass in bf16.

# %% [code]
# ── Cell 1: HF_TOKEN from Kaggle Secrets
from kaggle_secrets import UserSecretsClient

try:
    hf_token_secret = UserSecretsClient().get_secret("HF_TOKEN")
    print("[cell 1] HF_TOKEN read OK from Kaggle Secrets")
except Exception as exc:
    print(f"[cell 1] HF_TOKEN unreadable ({type(exc).__name__}: {exc}). "
          "Will stage to /kaggle/working/ for local pickup.")
    hf_token_secret = None

# %% [code]
# ── Cell 2: env, clone CrispASR, install deps
import os
import subprocess
import sys
import shutil
from pathlib import Path

if hf_token_secret:
    os.environ["HF_TOKEN"] = hf_token_secret
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT_F16 = WORK / "moss-audio-4b-instruct-f16.gguf"
OUT_REF = WORK / "moss-audio-4b-instruct-ref.gguf"

CRISPASR_REF = os.environ.get("CRISPASR_REF", "feature/moss-audio")
CRISPASR_REPO = os.environ.get(
    "CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", CRISPASR_REF,
    CRISPASR_REPO, str(REPO),
])
sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print(f"[cell 2] cloned CrispASR at {sha[:12]} (branch {CRISPASR_REF})")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers>=4.57", "safetensors", "gguf",
    "huggingface_hub", "hf_transfer",
])
print("[cell 2] deps installed")

# %% [code]
# ── Cell 3: download MOSS-Audio source weights
from huggingface_hub import snapshot_download

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "moss-audio-src"
        break
scratch.mkdir(parents=True, exist_ok=True)
print(f"[cell 3] source cache: {scratch}  "
      f"(free: {shutil.disk_usage(scratch).free / (1024**3):.1f} GiB)")

src = snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    cache_dir=str(scratch),
)
print(f"[cell 3] source dir: {src}")

# %% [code]
# ── Cell 4: run the converter (F16)
subprocess.check_call([
    sys.executable, "models/convert-moss-audio-to-gguf.py",
    "--input", src,
    "--output", str(OUT_F16),
    "--outtype", "f16",
], cwd=str(REPO))

print(f"[cell 4] wrote {OUT_F16} ({OUT_F16.stat().st_size / (1024**3):.1f} GiB)")

# %% [code]
# ── Cell 5: run reference dump on jfk.wav
#    Clone the MOSS-Audio GitHub repo for modeling code
MOSS_GITHUB = WORK / "MOSS-Audio-github"
if not MOSS_GITHUB.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/OpenMOSS/MOSS-Audio.git",
        str(MOSS_GITHUB),
    ])

os.environ["MOSS_AUDIO_DIR"] = src
os.environ["MOSS_AUDIO_GITHUB"] = str(MOSS_GITHUB)
os.environ["MOSS_AUDIO_PROMPT"] = "Transcribe this audio."
os.environ["MOSS_AUDIO_MAX_NEW"] = "128"

subprocess.check_call([
    sys.executable, "tools/dump_reference.py",
    "--backend", "moss-audio",
    "--model-dir", src,
    "--audio", str(REPO / "samples" / "jfk.wav"),
    "--output", str(OUT_REF),
], cwd=str(REPO))

print(f"[cell 5] wrote reference {OUT_REF} ({OUT_REF.stat().st_size / (1024**3):.2f} GiB)")

# %% [code]
# ── Cell 6: upload to HF
HF_REPO = "cstr/MOSS-Audio-4B-Instruct-GGUF"

if hf_token_secret:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_secret)

    # Create repo if needed
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[cell 6] repo creation: {e}")

    for path, name in [(OUT_F16, "moss-audio-4b-instruct-f16.gguf"),
                        (OUT_REF, "moss-audio-4b-instruct-ref.gguf")]:
        if path.exists():
            print(f"[cell 6] uploading {name} ({path.stat().st_size / (1024**3):.1f} GiB) …")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=HF_REPO,
                repo_type="model",
                commit_message=f"Add {name} (PLAN #58 MOSS-Audio-4B-Instruct)",
            )
            print(f"[cell 6] uploaded {name}")
else:
    print("[cell 6] no HF_TOKEN — files staged in /kaggle/working/ for local pickup:")
    for p in [OUT_F16, OUT_REF]:
        if p.exists():
            print(f"  {p} ({p.stat().st_size / (1024**3):.1f} GiB)")

print("[cell 6] done")
