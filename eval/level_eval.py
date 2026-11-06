"""Check the automatic input level (cpp/include/pf/auto_gain.h) on whole songs.

Live audio arrives at whatever volume the PC is set to, and the model finds fewer
notes in quiet audio. This feeds each song to the engine as is, quieter, with a
tone tilt, and with a volume dip in the middle, with and without --normalize.
Notes are scored against batch notes of the ORIGINAL file, so "file as is"
without auto level is the best any row can do.

    .venv\\Scripts\\python.exe eval\\level_eval.py

Appends one table per song to results.md. About 20 minutes on a GPU.
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import csv
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "eval"))

import mir_eval
import numpy as np
import soundfile as sf
import torch

import stream_sim
from models import Pipeline
from sweep import engine_env, find_exe

SR = 16000
ENGINE_ARGS = ["--stride", "16", "--delay", "16", "--fast"]
# Tone tilt measured on Sam's PC (Realtek speakers, loopback vs file), relative to the bass.
TILT_HZ = [0, 400, 800, 1600, 3200, 6400, 8000]
TILT_GAIN = [1.0, 1.0, 1.11, 1.245, 1.40, 1.51, 1.60]


def tilt(x: np.ndarray) -> np.ndarray:
    """Boost the treble the way that PC's sound output does."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(spectrum * np.interp(freqs, TILT_HZ, TILT_GAIN), len(x)).astype(np.float32)


def dip(x: np.ndarray) -> np.ndarray:
    """x0.4 volume, dropping to x0.03 for 20 s from 40% of the way in."""
    y = x * 0.4
    start = int(len(x) * 0.4)
    y[start: start + 20 * SR] *= 0.03 / 0.4
    return y


def engine_notes(samples: np.ndarray, normalize: bool, work: Path) -> tuple[list[dict], str]:
    """Run the engine on samples. Returns its notes and its median input gain."""
    wav = work / "in.wav"
    sf.write(wav, samples.astype(np.float32), SR, subtype="FLOAT")
    args = [str(find_exe()), "--wav", str(wav), "--out", str(work / "out"), *ENGINE_ARGS]
    result = subprocess.run(args + (["--normalize"] if normalize else []), env=engine_env(), cwd=ROOT,
                            capture_output=True, text=True, check=True)
    # The engine prints "input gain p50 x2.41, lowest ...": keep the "x2.41".
    gain = next((line.split()[3].rstrip(",") for line in result.stdout.splitlines() if line.startswith("input gain")), "")
    with open(work / "out" / "notes.csv", newline="") as f:
        notes = [{"pitch": int(r["pitch"]), "onset": float(r["onset"]), "offset": float(r["offset"])}
                 for r in csv.DictReader(f)]
    return notes, gain


def main() -> int:
    wavs = sorted(p for p in (ROOT / "test_audio" / "16k").glob("*.wav") if "_first" not in p.stem)
    if not wavs:
        print("No songs in test_audio/16k. Run py/prep_audio.py first.", file=sys.stderr)
        return 2
    pipe = Pipeline(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pipe.model.eval()
    sv = np.fromfile(ROOT / "models" / "sv_level2.bin", dtype="<f4")

    text = (
        f"\n## Automatic input level, {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Command: `python eval\\level_eval.py`\n\n"
        f"Machine: {stream_sim.machine_description()}\n\n"
        "Whole songs through the engine (file input, stride 256 ms + delay 256 ms), changed as the first column "
        "says. P, R and F1 are against batch notes of the original file (50 ms onsets, offsets ignored). "
        "\"Tone tilt\" is the treble boost measured on this PC's sound output. "
        "\"Dip\" is x0.4 volume with 20 s at x0.03 in the middle.\n"
    )
    cases = [
        ("file as is", lambda x: x),
        ("x0.4", lambda x: x * 0.4),
        ("x0.1", lambda x: x * 0.1),
        ("x0.03", lambda x: x * 0.03),
        ("tone tilt, x0.4", lambda x: tilt(x) * 0.4),
        ("dip", dip),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for wav in wavs:
            x, _ = sf.read(wav, dtype="float32")
            reference, _ = stream_sim.batch_reference(pipe, wav, sv)
            ref = stream_sim.notes_to_arrays(reference)
            print(f"{wav.name}: {len(reference)} batch notes", flush=True)
            text += (
                f"\n### {wav.name}\n\n{len(x) / SR:.1f} s, file peak {np.abs(x).max():.3f}, {len(reference)} batch notes.\n\n"
                "| input | auto level | notes | P | R | F1 | median gain |\n|---|---|---|---|---|---|---|\n"
            )
            for label, change in cases:
                for normalize in (False, True):
                    notes, gain = engine_notes(change(x.copy()), normalize, Path(tmp))
                    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                        *ref, *stream_sim.notes_to_arrays(notes),
                        onset_tolerance=stream_sim.ONSET_TOLERANCE_SEC, offset_ratio=None)
                    row = f"| {label} | {'on' if normalize else 'off'} | {len(notes)} | {p:.4f} | {r:.4f} | {f:.4f} | {gain or '-'} |"
                    print("  " + row, flush=True)
                    text += row + "\n"
    with open(ROOT / "results.md", "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Appended to {ROOT / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
