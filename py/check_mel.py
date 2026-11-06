"""Check the C++ mel spectrogram against a float64 run of the batch transform.

The reference uses the same audio, parameters and filterbank as
Pipeline.wav2feature, but computes in float64. In near-silent cells, float32
rounding in wav2feature alone can reach 1.2e-3, more than the limit below.
The difference from wav2feature itself is printed for information.

Usage (from the repo root, after building):
    .venv\\Scripts\\python.exe py\\check_mel.py test_audio\\16k\\song.wav
Passes if the max difference (log-mel) from the float64 reference is below 1e-3.
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))

import numpy as np
import torch
import torchaudio

import stream_sim  # also patches torchaudio.load
from models import Pipeline

EXE_DIRS = [ROOT / "cpp" / "build" / "Release", ROOT / "cpp" / "build"]


def find_exe(name: str) -> Path:
    for d in EXE_DIRS:
        for cand in (d / f"{name}.exe", d / name):
            if cand.exists():
                return cand
    raise SystemExit(f"{name} not found under cpp/build. Build the C++ project first.")


def float64_feature(pipe: Pipeline, wav: Path) -> np.ndarray:
    """AMT.wav2feature in float64, for a wav that is already 16 kHz."""
    feature = pipe.config["feature"]
    wave, sr = stream_sim.read_wav(wav)
    if sr != feature["sr"]:
        raise SystemExit(f"{wav} is {sr} Hz. Convert it first with py/prep_audio.py.")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=feature["sr"],
        n_fft=feature["fft_bins"],
        win_length=feature["window_length"],
        hop_length=feature["hop_sample"],
        pad_mode=feature["pad_mode"],
        n_mels=feature["mel_bins"],
        norm="slaney",
    ).to(torch.float64)
    spec = mel(torch.mean(wave.to(torch.float64), dim=0))
    return torch.log(spec + feature["log_offset"]).T.numpy()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    wav = Path(sys.argv[1])
    pipe = Pipeline(no_model=True)
    ref = float64_feature(pipe, wav)
    batch = pipe.wav2feature(str(wav)).numpy().astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mel.f32"
        subprocess.run([str(find_exe("mel_dump")), str(wav), str(out)], check=True)
        got = np.fromfile(out, dtype="<f4").reshape(-1, ref.shape[1])
    print(f"frames: python {len(ref)}, C++ {len(got)}")
    if got.shape != ref.shape:
        print("FAILED: frame counts differ")
        return 1
    diff = np.abs(got - ref)
    print(f"C++ vs float64 reference: max diff {diff.max():.2e}, mean diff {diff.mean():.2e}")
    print(f"C++ vs wav2feature (info): max diff {np.abs(got - batch).max():.2e}")
    print(f"float64 reference vs wav2feature (info): max diff {np.abs(ref - batch).max():.2e}")
    ok = diff.max() < 1e-3
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
