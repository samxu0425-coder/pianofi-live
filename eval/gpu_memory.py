"""Measure how much GPU memory one engine run takes, by sampling nvidia-smi while it runs.

    .venv\\Scripts\\python.exe eval\\gpu_memory.py test_audio\\16k\\song.wav

Run it with nothing else using the GPU: the number is everything that changed on the
card during the run. Runs one engine at a time and appends a table to results.md.
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "eval"))

import stream_sim
from sweep import engine_env, find_exe

SETTINGS = [
    ("stride 256 ms + delay 256 ms, cached", ["--stride", "16", "--delay", "16"]),
    ("stride 256 ms + delay 256 ms, cached, FP16", ["--stride", "16", "--delay", "16", "--fp16"]),
    ("stride 64 ms, cached", ["--stride", "4"]),
    ("stride 256 ms, no cache", ["--stride", "16", "--no-cache"]),
    ("stride 8192 ms (batch windows)", ["--stride", "512"]),
]


def gpu_used_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def measure(wav: Path, extra: list[str], out: Path) -> tuple[int, dict]:
    """Run the engine once. Returns (peak GPU MB above the starting level, metrics)."""
    before = gpu_used_mb()
    peak, done = [before], threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], gpu_used_mb())
            time.sleep(0.5)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        subprocess.run([str(find_exe()), "--wav", str(wav), "--fast", "--out", str(out), *extra],
                       env=engine_env(), cwd=ROOT, check=True, capture_output=True)
    finally:
        done.set()
        watcher.join()
    return peak[0] - before, json.loads((out / "metrics.json").read_text())


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    wav = Path(sys.argv[1])
    text = (
        f"\n## GPU memory per engine, {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Command: `python eval\\gpu_memory.py {wav}`\n\n"
        f"Machine: {stream_sim.machine_description()}\n\n"
        f"{wav.name}, file input, `--fast`, one engine at a time. GPU memory is the peak rise in "
        "`nvidia-smi` memory.used during the run, sampled every 0.5 s, so short spikes can be missed.\n\n"
        "| settings | GPU memory MB | pass p50 ms | pass max ms |\n|---|---|---|---|\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        for label, extra in SETTINGS:
            used, m = measure(wav, extra, Path(tmp) / "run")
            row = f"| {label} | {used} | {m['pass_ms']['p50']:.1f} | {m['pass_ms']['max']:.0f} |"
            print(row, flush=True)
            text += row + "\n"
    with open(ROOT / "results.md", "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Appended to {ROOT / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
