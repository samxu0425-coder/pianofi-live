"""Run the C++ engine over a grid of settings and score each run.

    .venv\\Scripts\\python.exe eval\\sweep.py test_audio\\16k\\song.wav
    .venv\\Scripts\\python.exe eval\\sweep.py test_audio\\16k\\song.wav --seconds 30 --quick

Each run is real time, so it takes as long as the clip (default: first 60 s).
Every run is scored by eval/score_engine.py, which appends a row to results.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# (label, extra args). Ordered as a story: baseline -> GPU -> caching -> stride -> A heads.
FULL = [
    ("cpu-nocache-512ms", ["--cpu", "--no-cache", "--stride", "32"]),
    ("cpu-cache-512ms", ["--cpu", "--stride", "32"]),
    ("gpu-nocache-512ms", ["--no-cache", "--stride", "32"]),
    ("gpu-cache-512ms", ["--stride", "32"]),
    ("gpu-cache-256ms", ["--stride", "16"]),
    ("gpu-cache-128ms", ["--stride", "8"]),
    ("gpu-cache-64ms", ["--stride", "4"]),
    ("gpu-a-heads-128ms", ["--heads", "a", "--stride", "8"]),
    ("gpu-batch-windows", ["--stride", "512"]),  # must reproduce batch exactly
]
QUICK = [FULL[1], FULL[3], FULL[5], FULL[8]]


def find_exe() -> Path:
    for d in (ROOT / "cpp" / "build" / "Release", ROOT / "cpp" / "build"):
        for cand in (d / "pianofi_live.exe", d / "pianofi_live"):
            if cand.exists():
                return cand
    raise SystemExit("pianofi_live not found under cpp/build. Build the C++ project first.")


def engine_env() -> dict[str, str]:
    """Put the CUDA 12 and cuDNN 9 DLLs that ship with CUDA torch on PATH."""
    torch_lib = Path(importlib.util.find_spec("torch").origin).parent / "lib"
    return {**os.environ, "PATH": f"{torch_lib}{os.pathsep}{os.environ['PATH']}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=Path, help="16 kHz wav from py/prep_audio.py")
    ap.add_argument("--seconds", type=float, default=60.0, help="clip length to use (0 = whole song)")
    ap.add_argument("--quick", action="store_true", help="4 runs instead of 9")
    args = ap.parse_args()

    wav = args.wav
    if args.seconds > 0:
        data, sr = sf.read(wav, dtype="float32")
        if sr != 16000:
            raise SystemExit(f"{wav} is {sr} Hz: run py/prep_audio.py first")
        clip = wav.parent / f"{wav.stem}_first{int(args.seconds)}s.wav"
        sf.write(clip, np.asarray(data[: int(args.seconds * sr)], np.float32), sr, subtype="FLOAT")
        wav = clip

    exe = find_exe()
    env = engine_env()
    failed = []
    for label, extra in (QUICK if args.quick else FULL):
        out = ROOT / "runs" / f"{wav.stem}-{label}"
        print(f"\n=== {label} ===", flush=True)
        cmd = [str(exe), "--wav", str(wav), "--models", str(ROOT / "models"), "--out", str(out), "--dump", *extra]
        if subprocess.call(cmd, env=env) != 0:
            failed.append(label)
            continue
        # The engine falls back to CPU if CUDA won't load. Don't report that as a GPU run.
        provider = json.loads((out / "metrics.json").read_text())["provider"]
        if label.startswith("gpu") and provider != "CUDA":
            print(f"{label} ran on {provider}, not scoring it", file=sys.stderr)
            failed.append(label)
            continue
        if subprocess.call([PY, str(ROOT / "eval" / "score_engine.py"), str(out), str(wav)]) != 0:
            failed.append(label)
    print("\nAll runs done." if not failed else f"\nProblems in: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
