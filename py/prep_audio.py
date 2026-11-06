"""Convert wav files to 16 kHz mono float32 wavs for the C++ engine.

Does exactly what Pipeline.wav2feature does before the mel step (average the
channels, then torchaudio Resample to 16 kHz), so the engine and batch mode see
the same samples. Running the batch pipeline on the converted file gives the
same features as on the original.

Usage (from the repo root):
    .venv\\Scripts\\python.exe py\\prep_audio.py test_audio\\song.wav          -> test_audio\\16k\\song.wav
    .venv\\Scripts\\python.exe py\\prep_audio.py test_audio\\*.wav
"""

from __future__ import annotations

import glob
import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from stream_sim import read_wav  # same decoder the stream simulation uses

TARGET_SR = 16000


def convert(src: Path) -> Path:
    wave, sr = read_wav(src)
    mono = torch.mean(wave, dim=0)
    mono = torchaudio.transforms.Resample(sr, TARGET_SR)(mono)
    dst = src.parent / "16k" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, mono.numpy().astype(np.float32), TARGET_SR, subtype="FLOAT")
    return dst


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    paths = [Path(p) for arg in sys.argv[1:] for p in (glob.glob(arg) or [arg])]
    for p in paths:
        if p.parent.name == "16k":
            continue
        dst = convert(p)
        print(f"{p} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
