"""Check that a long session's memory stays bounded, and that bounding it changes no notes.

Live input drops mel and decoder frames once nothing needs them. --bounded-memory
makes a file run do the same, so it can be compared with a normal run:

  1. Same notes: each song, with and without --bounded-memory. notes.csv must be identical.
  2. Bounded: 5 and 20 minutes of music (the test songs repeated). The frames held in
     memory must not grow with the length when history is dropped.

    .venv\\Scripts\\python.exe eval\\memory_check.py

Appends two tables to results.md. About 20 minutes on a GPU. Exit code 1 if any notes differ.
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import json
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np
import soundfile as sf

import stream_sim
from sweep import engine_env, find_exe

SR = 16000
SETTINGS = [["--stride", "16", "--delay", "16"], ["--stride", "4"]]


def run(wav: Path, out: Path, extra: list[str]) -> tuple[bytes, dict]:
    """Run the engine. Returns notes.csv as bytes and the metrics."""
    subprocess.run([str(find_exe()), "--wav", str(wav), "--fast", "--out", str(out), *extra],
                   env=engine_env(), cwd=ROOT, check=True, capture_output=True)
    return (out / "notes.csv").read_bytes(), json.loads((out / "metrics.json").read_text())


def main() -> int:
    wavs = sorted(p for p in (ROOT / "test_audio" / "16k").glob("*.wav") if "_first" not in p.stem)
    if not wavs:
        print("No songs in test_audio/16k. Run py/prep_audio.py first.", file=sys.stderr)
        return 2
    text = (
        f"\n## Long-session memory, {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Command: `python eval\\memory_check.py`\n\n"
        f"Machine: {stream_sim.machine_description()}\n\n"
        "File input, `--fast`. \"Dropped\" runs use `--bounded-memory`, which is what live input always does.\n\n"
        "| song | settings | notes | notes identical with history dropped |\n|---|---|---|---|\n"
    )
    all_same = True
    with tempfile.TemporaryDirectory() as tmp:
        for wav in wavs:
            for settings in SETTINGS:
                plain, metrics = run(wav, Path(tmp) / "plain", settings)
                dropped, _ = run(wav, Path(tmp) / "dropped", settings + ["--bounded-memory"])
                same = plain == dropped
                all_same &= same
                row = f"| {wav.name} | `{' '.join(settings)}` | {metrics['notes']} | {'yes' if same else 'NO'} |"
                print(row, flush=True)
                text += row + "\n"

        text += (
            "\nThe test songs repeated to the given length, stride 256 ms + delay 256 ms. Frames held = the most "
            "kept in memory at once. Old mel frames are dropped in every run without `--dump`, so only the "
            "decoder column differs. Peak memory is the whole process, which includes about 1 GB of CUDA "
            "libraries and the wav file itself (3.7 MB per minute).\n\n"
            "| length | history | decoder frames held | mel frames held | peak memory MB |\n|---|---|---|---|---|\n"
        )
        loop = np.concatenate([sf.read(w, dtype="float32")[0] for w in wavs])
        for minutes in (5, 20):
            n = minutes * 60 * SR
            long_wav = Path(tmp) / "long.wav"
            sf.write(long_wav, np.tile(loop, n // len(loop) + 1)[:n], SR, subtype="FLOAT")
            for label, extra in (("kept", []), ("dropped", ["--bounded-memory"])):
                _, m = run(long_wav, Path(tmp) / "long", SETTINGS[0] + extra)
                row = (f"| {minutes} min | {label} | {m['max_decoder_frames']} | {m['max_mel_frames']} "
                       f"| {m['peak_memory_mb']:.0f} |")
                print(row, flush=True)
                text += row + "\n"
    with open(ROOT / "results.md", "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Appended to {ROOT / 'results.md'}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
