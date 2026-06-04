# %% [markdown]
# # CrispASR — MOSS-Audio-4B-Instruct F16+Q4K GGUF conversion
#
# Convert `OpenMOSS-Team/MOSS-Audio-4B-Instruct` (3-shard safetensors,
# ~10 GB bf16) to F16 GGUF then quantize to Q4_K on Kaggle GPU notebook.
# Upload both to `cstr/MOSS-Audio-4B-Instruct-GGUF`.
#
# VPS can't fit it (7.6 GB RAM, OOMs on quantize). Kaggle GPU has
# 30 GB RAM + T4 16 GB VRAM — plenty of headroom.
#
# If HF_TOKEN reads OK → auto-uploads at the end.
# If not → stages to /kaggle/working/ for local pickup.

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
from pathlib import Path

if hf_token_secret:
    os.environ["HF_TOKEN"] = hf_token_secret
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token_secret

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
OUT_F16 = WORK / "moss-audio-4b-instruct-f16.gguf"
OUT_Q4K = WORK / "moss-audio-4b-instruct-q4_k.gguf"

# Clone from feature branch (converter lives there)
BRANCH = os.environ.get("CRISPASR_REF", "feature/moss-audio")
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", BRANCH,
        "--filter=blob:none", "--sparse",
        "https://github.com/CrispStrobe/CrispASR.git", str(REPO),
    ])
subprocess.check_call(["git", "-C", str(REPO), "sparse-checkout", "set",
                       "models", "tools", "examples/crispasr-quantize",
                       "src", "ggml", "cmake", "CMakeLists.txt"])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "safetensors", "gguf", "huggingface_hub", "hf_transfer",
])
print("[cell 2] deps installed")

# %% [code]
# ── Cell 3: download MOSS-Audio source weights
from huggingface_hub import snapshot_download
import shutil

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
# ── Cell 4: run the F16 converter
subprocess.check_call([
    sys.executable, "models/convert-moss-audio-to-gguf.py",
    "--input", src,
    "--output", str(OUT_F16),
    "--outtype", "f16",
], cwd=str(REPO))

print(f"[cell 4] wrote F16: {OUT_F16} ({OUT_F16.stat().st_size / (1024**3):.1f} GiB)")

# %% [code]
# ── Cell 5: build crispasr-quantize and quantize F16 → Q4_K
BUILD = WORK / "build"
BUILD.mkdir(exist_ok=True)

subprocess.check_call([
    "cmake", "-G", "Ninja", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
])
subprocess.check_call([
    "cmake", "--build", str(BUILD), "--target", "crispasr-quantize", "-j4",
])

QUANTIZE = BUILD / "bin" / "crispasr-quantize"
subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q4K), "q4_k"])

print(f"[cell 5] wrote Q4K: {OUT_Q4K} ({OUT_Q4K.stat().st_size / (1024**3):.1f} GiB)")

# %% [code]
# ── Cell 6: upload to HF
HF_REPO = "cstr/MOSS-Audio-4B-Instruct-GGUF"

if hf_token_secret:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_secret)

    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[cell 6] repo creation: {e}")

    for path, name in [(OUT_F16, "moss-audio-4b-instruct-f16.gguf"),
                        (OUT_Q4K, "moss-audio-4b-instruct-q4_k.gguf")]:
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
    for p in [OUT_F16, OUT_Q4K]:
        if p.exists():
            print(f"  {p} ({p.stat().st_size / (1024**3):.1f} GiB)")

print("[cell 6] done")
