"""Score a C++ engine run against batch AMT-APC, and check the C++ decoder.

Needs a run made with --dump (raw outputs) on a 16 kHz wav from py/prep_audio.py.

    .venv\\Scripts\\python.exe eval\\score_engine.py runs\\s32 test_audio\\16k\\song.wav

Reports:
  - accuracy: C++ notes.csv vs Pipeline.wav2midi notes (mir_eval note F1,
    50 ms onsets, offsets ignored; same as py/stream_sim.py)
  - decoder check: Pipeline.mpe2note on the dumped arrays vs notes.csv
    (must match: the C++ decoder is a port of it). Times may differ by 1e-5 s,
    or by one float32 step for notes past 128 s, where that step is larger.
  - array check: at stride 512 with heads b, the dumped arrays must match batch
Appends one row to results.md and stores the run and score in runs/results.db (eval/results_db.py).
"""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import csv
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "eval"))

import mir_eval
import numpy as np
import torch

import results_db
import stream_sim
from models import Pipeline
from utils import config

N_NOTE = config.data.midi.num_note


def load_notes(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [
            {"pitch": int(r["pitch"]), "onset": float(r["onset"]), "offset": float(r["offset"]), "velocity": int(r["velocity"])}
            for r in csv.DictReader(f)
        ]


def f1(reference: list[dict], estimate: list[dict]) -> tuple[float, float, float]:
    ref = stream_sim.notes_to_arrays(reference)
    est = stream_sim.notes_to_arrays(estimate)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        *ref, *est, onset_tolerance=stream_sim.ONSET_TOLERANCE_SEC, offset_ratio=None
    )
    return p, r, f


def same_notes(a: list[dict], b: list[dict], tol: float = 1e-5) -> bool:
    if len(a) != len(b):
        return False
    # Compare per pitch: notes that start together can sort either way, because
    # Python and C++ round their onset times differently.
    key = lambda n: (n["pitch"], n["onset"])
    a, b = sorted(a, key=key), sorted(b, key=key)

    # mpe2note keeps note times as float32, and past 128 s one float32 step is
    # bigger than tol (1.5e-5 s). Allow one step there; before that, tol decides.
    def close(s: float, t: float) -> bool:
        return abs(s - t) <= max(tol, float(np.spacing(np.float32(max(s, t)))))

    return all(
        x["pitch"] == y["pitch"] and x["velocity"] == y["velocity"]
        and close(x["onset"], y["onset"]) and close(x["offset"], y["offset"])
        for x, y in zip(a, b)
    )


def add_row(text: str, header: str, row: str) -> str:
    """Put a row at the end of the engine table. Other sections may come after it."""
    title = "## C++ engine runs"
    if title not in text:
        text += header
    table = text.index("|---", text.index(title))
    end = text.find("\n\n", table)
    if end == -1:
        end = len(text.rstrip("\n"))
    return f"{text[:end]}\n{row}{text[end:]}".rstrip("\n") + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    run, wav = Path(sys.argv[1]), Path(sys.argv[2])
    metrics = json.loads((run / "metrics.json").read_text())
    style = "level2"
    sv = np.fromfile(ROOT / "models" / f"sv_{style}.bin", dtype="<f4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Pipeline(device=device)
    pipe.model.eval()
    reference, batch_arrays = stream_sim.batch_reference(pipe, wav, sv)
    cpp_notes = load_notes(run / "notes.csv")
    p, r, f = f1(reference, cpp_notes)
    print(f"accuracy vs batch: P={p:.4f} R={r:.4f} F1={f:.4f} ({len(cpp_notes)} notes, batch {len(reference)})")

    # Decoder check: real mpe2note on the arrays the engine produced.
    arrays = [
        np.fromfile(run / "onset.f32", dtype="<f4").reshape(-1, N_NOTE),
        np.fromfile(run / "offset.f32", dtype="<f4").reshape(-1, N_NOTE),
        np.fromfile(run / "mpe.f32", dtype="<f4").reshape(-1, N_NOTE),
        np.fromfile(run / "velocity.i64", dtype="<i8").reshape(-1, N_NOTE),
    ]
    th = config.infer.threshold
    py_notes = pipe.mpe2note(*arrays, thred_onset=th.onset, thred_offset=th.offset, thred_mpe=th.frame)
    py_notes = stream_sim.drop_short_notes(py_notes, config.infer.min_duration)
    dec_ok = same_notes(py_notes, cpp_notes)
    print(f"decoder check (C++ vs mpe2note on the same arrays): {'PASSED' if dec_ok else 'FAILED'}"
          f" ({len(cpp_notes)} vs {len(py_notes)} notes)")

    arr_ok = None  # only checked at stride 512
    if metrics["stride_frames"] == 512 and metrics["heads"] == "b":
        d = max(float(np.abs(a - b).max()) for a, b in zip(arrays[:3], batch_arrays[:3]))
        vel = float((arrays[3] != batch_arrays[3]).mean())
        arr_ok = d < 1e-3 and vel < 1e-3
        print(f"array check vs batch: max diff {d:.2e}, velocity mismatch {vel:.4%}: {'PASSED' if arr_ok else 'FAILED'}")

    machine = stream_sim.machine_description()
    results_db.record(run, metrics, machine, p, r, f, len(cpp_notes), len(reference), dec_ok, arr_ok)
    print(f"Stored the run and score in {results_db.DB}")

    rm = ROOT / "results.md"
    lag = metrics["frame_lag_ms"]
    # --fast runs don't pace the audio, so they have no lag to report. Neither do runs
    # that dropped audio: the engine times frame N as file position N, wrong after a drop.
    has_lag = metrics["realtime"] and lag["n"] > 0 and metrics["overrun_samples"] == 0
    lag_p50 = f"{lag['p50']:.0f}" if has_lag else "not measured"
    lag_p99 = f"{lag['p99']:.0f}" if has_lag else "not measured"
    command = metrics["command"].replace(str(ROOT) + os.sep, "")
    stride = f"{metrics['stride_frames'] * 16} ms"
    if metrics.get("delay_frames", 0) > 0:  # older runs have no delay field
        stride += f" + delay {metrics['delay_frames'] * 16} ms"
    device = metrics["provider"]
    if metrics.get("precision") == "fp16":  # half-precision graphs (--fp16)
        device += " FP16"
    if metrics.get("ring_buffer") == "spsc":  # older runs have no ring_buffer field: they used the mutex
        device += ", spsc ring"
    row = (
        f"| {datetime.now():%Y-%m-%d %H:%M} | {wav.name} | {device} | {stride} "
        f"| {metrics['heads']} | {'yes' if metrics['cache'] else 'no'} | {p:.4f} | {r:.4f} | {f:.4f} "
        f"| {metrics['pass_ms']['p50']:.1f} | {metrics['pass_ms']['p99']:.1f} "
        f"| {lag_p50} | {lag_p99} | {metrics['real_time_factor']:.3f} "
        f"| {metrics['overrun_samples']} | {'ok' if dec_ok else 'MISMATCH'} "
        f"| {machine} | `{command}` |"
    )
    header = (
        "\n## C++ engine runs\n\n"
        "F1 vs batch notes (50 ms onsets, offsets ignored). Lag = time from audio arriving to its output frame "
        "being ready (real-time runs with 0 overruns only). RTF = compute time / clip length, so it reads too low "
        "on runs with overruns: compare pass ms to the stride there. Overruns = audio samples dropped. "
        'Rows marked "spsc ring" used the lock-free ring buffer, and all others used the mutex one. '
        "Commands run from the repo root, and each run is scored with "
        "`.venv\\Scripts\\python.exe eval\\score_engine.py <--out dir> <--wav file>`. "
        "GPU runs need the CUDA DLLs on PATH (see RUN.md).\n\n"
        "| when | song | device | stride | heads | cache | P | R | F1 | pass p50 ms | pass p99 ms "
        "| lag p50 ms | lag p99 ms | RTF | overruns | decoder | machine | command |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    text = rm.read_text(encoding="utf-8") if rm.exists() else ""
    rm.write_text(add_row(text, header, row), encoding="utf-8", newline="\n")
    print(f"Added a row to {rm}")
    return 0 if dec_ok and arr_ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
