#!/usr/bin/env python3
"""
Kaggle kernel: end-to-end dots.tts synthesis test on GPU.

Builds CrispASR with CUDA, downloads Q4_K GGUFs, runs synthesis,
validates output WAV.
"""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(str(WORK))

PROGRESS = WORK / "progress.txt"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")

log("Kernel started")

try:
    # ── Clone CrispASR ──
    _CRISPASR_DIR = Path("/tmp/CrispASR")
    if not _CRISPASR_DIR.exists():
        log("Cloning CrispASR...")
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(_CRISPASR_DIR)])
    sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))

    import kaggle_harness as kh
    kh.init_progress()
    log("kaggle_harness imported OK")

    # ── Build CrispASR with CUDA ──
    log("Installing build toolchain...")
    kh.install_build_toolchain()

    log("Detecting CUDA arch...")
    cuda_arch = kh.detect_cuda_arch()
    log(f"CUDA arch: {cuda_arch}")

    cmake_flags = kh.cuda_build_flags(cuda_arch)
    cache_flags = kh.cache_and_link_flags()
    n_jobs = kh.safe_build_jobs(gpu=True)

    build_dir = _CRISPASR_DIR / "build"
    cmake_env = os.environ.copy()
    cmake_env["CCACHE_DIR"] = "/kaggle/working/.ccache"

    # Configure
    log("CMake configure...")
    cmake_args = [
        "cmake", "-G", "Ninja", "-B", str(build_dir), "-S", str(_CRISPASR_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    # Add cache/link flags
    if isinstance(cache_flags, list):
        cmake_args.extend(cache_flags)
    elif isinstance(cache_flags, str) and cache_flags.strip():
        cmake_args.extend(cache_flags.strip().split())
    # Add CUDA flags
    if isinstance(cmake_flags, list):
        cmake_args.extend(cmake_flags)
    elif isinstance(cmake_flags, str) and cmake_flags.strip():
        cmake_args.extend(cmake_flags.strip().split())

    log(f"cmake args: {cmake_args}")
    r = subprocess.run(cmake_args, capture_output=True, text=True, env=cmake_env,
                       cwd=str(_CRISPASR_DIR), timeout=120)
    log(f"CMake rc={r.returncode}")
    if r.returncode != 0:
        log(f"stderr: {r.stderr[-800:]}")

    # Build CLI
    log(f"Building with {n_jobs} jobs...")
    with kh.build_heartbeat("dots-tts CUDA build"):
        r2 = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "crispasr-cli", f"-j{n_jobs}"],
            capture_output=True, text=True, env=cmake_env, cwd=str(_CRISPASR_DIR), timeout=1200)
    log(f"Build rc={r2.returncode}")
    if r2.returncode != 0:
        log(f"Build stderr (last 500): {r2.stderr[-500:]}")

    cli_bin = build_dir / "bin" / "crispasr"
    if not cli_bin.exists():
        log(f"ERROR: CLI not found at {cli_bin}")
        sys.exit(1)
    log("CLI built OK")

    # ── Download GGUFs ──
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    hf_token = kh.resolve_hf_token()

    from huggingface_hub import hf_hub_download

    model_dir = WORK / "models"
    model_dir.mkdir(exist_ok=True)

    files = [
        "dots-tts-soar-q4_k.gguf",
        "dots-tts-soar-vocoder-f16.gguf",
    ]
    for fname in files:
        log(f"Downloading {fname}...")
        hf_hub_download("cstr/dots-tts-soar-GGUF", fname,
                       local_dir=str(model_dir), token=hf_token if hf_token else None)
        log(f"  {fname}: {(model_dir / fname).stat().st_size / (1024*1024):.1f} MB")

    # ── Check GPU ──
    log("GPU check:")
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")

    # ── Run synthesis ──
    core_model = model_dir / "dots-tts-soar-q4_k.gguf"
    output_wav = WORK / "dots_tts_output.wav"

    test_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for i, text in enumerate(test_texts):
        log(f"\n=== Test {i+1}: '{text}' ===")
        t0 = time.time()

        cmd = [
            str(cli_bin),
            "--backend", "dots-tts",
            "-m", str(core_model),
            "--tts", text,
            "--tts-output", str(output_wav),
            "--seed", "42",
            "-t", "4",
        ]
        log(f"CMD: {' '.join(cmd)}")

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          env={**os.environ, "DOTS_TTS_BENCH": "1", "CRISPASR_DOTS_TTS_DEBUG": "1"})

        elapsed = time.time() - t0
        log(f"Exit code: {r.returncode} ({elapsed:.1f}s)")
        log(f"stdout: {r.stdout[-500:]}")
        if r.stderr:
            log(f"stderr:\n{r.stderr[-2000:]}")

        if output_wav.exists():
            sz = output_wav.stat().st_size
            log(f"Output WAV: {sz} bytes ({sz/1024:.1f} KB)")

            # Check if WAV has actual audio content
            if sz > 44:  # WAV header is 44 bytes
                log("WAV file produced with audio content!")

                # Quick audio stats
                try:
                    import struct
                    with open(output_wav, "rb") as wf:
                        wf.seek(40)
                        data_size = struct.unpack("<I", wf.read(4))[0]
                        n_samples = data_size // 4  # float32
                        wf.seek(24)
                        sample_rate = struct.unpack("<I", wf.read(4))[0]
                        duration = n_samples / sample_rate if sample_rate > 0 else 0
                        log(f"  Samples: {n_samples}, Rate: {sample_rate} Hz, Duration: {duration:.2f}s")

                        # Read a few samples to check they're not all zero
                        wf.seek(44)
                        raw = wf.read(min(4000, data_size))
                        samples = struct.unpack(f"<{len(raw)//4}f", raw)
                        max_val = max(abs(s) for s in samples)
                        log(f"  Max amplitude (first 1000 samples): {max_val:.6f}")
                        if max_val > 0.001:
                            log("  ✓ Audio has non-trivial content")
                        else:
                            log("  ✗ Audio appears silent (all near-zero)")
                except Exception as e:
                    log(f"  WAV analysis failed: {e}")
            else:
                log("WAV file is too small (empty)")
        else:
            log("No output WAV produced")

        # Break after first test if it took too long
        if elapsed > 120:
            log("Skipping remaining tests (too slow)")
            break

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Python reference dump (dots.tts upstream)
    # ══════════════════════════════════════════════════════════════════
    log("\n=== Phase 2: Python reference dump ===")
    try:
        log("Installing dots_tts Python package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/rednote-hilab/dots.tts.git",
                               "gguf"], timeout=300)

        ref_dir = WORK / "ref_dump"
        ref_dir.mkdir(exist_ok=True)

        # Run the reference backend
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "reference_backends"))

        ref_env = os.environ.copy()
        ref_env["DOTS_TEXT"] = "Hello world, this is a test."
        ref_env["DOTS_MAX_PATCHES"] = "2"
        ref_env["DOTS_SEED"] = "42"
        ref_env["DOTS_ODE_STEPS"] = "4"

        log("Running dots_tts reference dump...")
        r = subprocess.run([
            sys.executable, "-c",
            f"""
import os, sys, numpy as np
from pathlib import Path
os.environ['DOTS_TEXT'] = '{ref_env["DOTS_TEXT"]}'
os.environ['DOTS_MAX_PATCHES'] = '{ref_env["DOTS_MAX_PATCHES"]}'
os.environ['DOTS_SEED'] = '{ref_env["DOTS_SEED"]}'
os.environ['DOTS_ODE_STEPS'] = '{ref_env["DOTS_ODE_STEPS"]}'
sys.path.insert(0, '{_CRISPASR_DIR / "tools" / "reference_backends"}')
import dots_tts_reference as ref
results = ref.run(None, 0, Path('{ref_dir}'))
for k, v in results.items():
    print(f'  {{k}}: {{v.shape}} min={{v.min():.4f}} max={{v.max():.4f}}')
"""
        ], capture_output=True, text=True, timeout=600, env=ref_env)

        log(f"Reference dump rc={r.returncode}")
        log(f"stdout:\n{r.stdout[-2000:]}")
        if r.stderr:
            log(f"stderr:\n{r.stderr[-1000:]}")

        # List dumped files
        log("Reference files:")
        for f in sorted(ref_dir.iterdir()):
            log(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        log(f"Reference dump failed: {e}")
        log(traceback.format_exc())

    log("\nDone!")

except Exception as e:
    log(f"\nFATAL ERROR: {e}")
    log(traceback.format_exc())
