"""Simulate live streaming with AMT-APC and measure accuracy against batch output.

For each wav in test_audio/, the spectrogram is computed once. Then, every stride
S, one model pass runs on the latest 576 frames and only the newest S output
frames are kept. The stitched output goes through Pipeline.mpe2note and is scored
against Pipeline.wav2midi on the same file, style vector and seed.

Alignment
---------
A pass reads 576 input frames (32 back margin + 512 + 32 forward margin) and
predicts the middle 512. So a pass whose newest input frame is end - 1 predicts up
to frame end - 33, and output always lags input by margin_f = 32 frames (0.512 s).
To place the newest S output frames at [o, o + S), the pass must end at
o + S + margin_f, so passes run at end = margin_f + S, margin_f + 2S, ...

At S = 512 these are exactly the batch windows, which is what the F1 = 1.0 check
relies on. Passes that run past the end of the song read min_value, as batch does.

Delay
-----
A delay of D frames emits the same output frames from a pass that runs D frames
later: rows [512 - S - D, 512 - D) of its output, so S + D can't exceed 512. Those
frames then have D newer frames in the window, and lag grows by D * 16 ms.

Usage (from the repo root)
--------------------------
    .venv/Scripts/python.exe py/stream_sim.py
    .venv/Scripts/python.exe py/stream_sim.py --verify-only
    .venv/Scripts/python.exe py/stream_sim.py --strides 0.256 0.512 --delays 0 16 32 64 128
"""

from __future__ import annotations

import subprocess
import sys

# amt-apc opens its JSON files with the platform default encoding, which fails on
# Windows (cp1252). It can't be modified, so run in UTF-8 mode instead.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import math
import os
import platform
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AMT_APC = ROOT / "amt-apc"
sys.dont_write_bytecode = True  # Leave no __pycache__ inside ./amt-apc.
sys.path.insert(0, str(AMT_APC))

import mir_eval
import numpy as np
import torch

from data import SVSampler
from models import Pipeline
from models.hFT_Transformer import amt as amt_module
from utils import config

DEFAULT_STRIDES = (0.256, 0.512, 1.024, 8.192)
BATCH_STRIDE_FRAMES = config.data.input.num_frame
HOP_SEC = config.data.feature.hop_sample / config.data.feature.sr
ONSET_TOLERANCE_SEC = 0.05
OFFSET_RATIO = 0.2
LAG_FLOOR_MS = 576  # 512 ms model lookahead + 64 ms mel window


def read_wav(path) -> tuple[torch.Tensor, int]:
    """Decode a wav to ([channels, samples] float32 in [-1, 1], sample rate).

    Replaces torchaudio.load, which needs torchcodec and FFmpeg shared libraries.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    fmt = pcm = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if chunk_id == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag opens the GUID.
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = tag, channels, rate, bits
        elif chunk_id == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"Missing fmt or data chunk: {path}")

    tag, channels, rate, bits = fmt
    if (tag, bits) == (1, 8):
        samples = (np.frombuffer(pcm, np.uint8).astype(np.float64) - 128) / 128
    elif (tag, bits) == (1, 16):
        samples = np.frombuffer(pcm, "<i2") / 2**15
    elif (tag, bits) == (1, 24):
        raw = np.frombuffer(pcm[: len(pcm) // 3 * 3], np.uint8).reshape(-1, 3)
        padded = np.zeros((len(raw), 4), np.uint8)
        padded[:, 1:] = raw  # Shift left one byte so the sign bit lands correctly.
        samples = padded.view("<i4").ravel() / 2**31
    elif (tag, bits) == (1, 32):
        samples = np.frombuffer(pcm, "<i4") / 2**31
    elif (tag, bits) == (3, 32):
        samples = np.frombuffer(pcm, "<f4")
    elif (tag, bits) == (3, 64):
        samples = np.frombuffer(pcm, "<f8")
    else:
        raise ValueError(f"Unsupported wav format (tag={tag}, bits={bits}): {path}")

    samples = samples[: len(samples) // channels * channels].astype(np.float32)
    return torch.from_numpy(samples.reshape(-1, channels).T.copy()), rate


# Pipeline.wav2feature looks this up at call time, so its code runs unchanged.
amt_module.torchaudio.load = read_wav


@dataclass
class Stream:
    """Stitched model output plus the cost of producing it."""

    onset: np.ndarray
    offset: np.ndarray
    frame: np.ndarray
    velocity: np.ndarray
    passes: int
    mean_pass_ms: float


@dataclass
class Row:
    stride_sec: float
    delay_ms: float
    precision: float
    recall: float
    f1: float
    f1_with_offset: float
    passes: float
    mean_pass_ms: float


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def stride_to_frames(stride_sec: float) -> int:
    """Convert seconds to frames, rejecting strides that can't be stitched."""
    frames = round(stride_sec / HOP_SEC)
    if not math.isclose(frames * HOP_SEC, stride_sec, abs_tol=1e-9):
        raise ValueError(f"Stride {stride_sec} s is not a multiple of {HOP_SEC} s")
    if not 1 <= frames <= BATCH_STRIDE_FRAMES:
        raise ValueError(
            f"Stride {stride_sec} s must be 1 to {BATCH_STRIDE_FRAMES} frames: "
            "a pass only outputs that many"
        )
    return frames


def latest_window(feature: np.ndarray, end: int) -> np.ndarray:
    """Return frames [end - 576, end), with min_value outside the song."""
    cfg = config.data.input
    length = cfg.margin_b + cfg.num_frame + cfg.margin_f
    window = np.full(
        (length, feature.shape[1]), cfg.min_value, dtype=np.float32
    )
    start = end - length
    src_start, src_end = max(start, 0), min(end, len(feature))
    if src_end > src_start:
        window[src_start - start:src_end - start] = feature[src_start:src_end]
    return window


def simulate_stream(
    pipe: Pipeline, feature: np.ndarray, sv: torch.Tensor, stride: int, delay: int = 0
) -> Stream:
    """Run one pass per stride and stitch the output S frames, D back from the newest."""
    cfg = config.data.input
    newest = cfg.num_frame - delay  # One past the last output row to keep.
    # Batch pads its output to a multiple of 512 frames. mpe2note uses that length
    # to end the last note, so every stride produces the same length.
    n_output = math.ceil(len(feature) / cfg.num_frame) * cfg.num_frame
    n_passes = math.ceil(n_output / stride)

    n_note = config.data.midi.num_note
    onset = np.zeros((n_output, n_note), dtype=np.float32)
    offset = np.zeros((n_output, n_note), dtype=np.float32)
    frame = np.zeros((n_output, n_note), dtype=np.float32)
    velocity = np.zeros((n_output, n_note), dtype=np.int8)

    pipe.model.eval()  # Dropout would make every pass differ from batch.
    elapsed = 0.0
    for k in range(n_passes):
        end = cfg.margin_f + (k + 1) * stride + delay
        spec = torch.from_numpy(latest_window(feature, end))
        spec = spec.T.unsqueeze(0).to(pipe.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = pipe.model(spec, sv)
        # wav2midi uses the second decoder's outputs: indices 5 to 8 of 9.
        pass_onset, pass_offset, pass_frame, pass_velocity = (
            o.squeeze(0)[newest - stride:newest] for o in outputs[5:9]
        )
        pass_onset = pass_onset.cpu().numpy()
        pass_offset = pass_offset.cpu().numpy()
        pass_frame = pass_frame.cpu().numpy()
        pass_velocity = pass_velocity.argmax(2).cpu().numpy()
        elapsed += time.perf_counter() - start

        lo = k * stride
        hi = min(lo + stride, n_output)
        onset[lo:hi] = pass_onset[: hi - lo]
        offset[lo:hi] = pass_offset[: hi - lo]
        frame[lo:hi] = pass_frame[: hi - lo]
        velocity[lo:hi] = pass_velocity[: hi - lo]

    return Stream(
        onset, offset, frame, velocity, n_passes, elapsed / n_passes * 1000
    )


def drop_short_notes(notes: list[dict], min_length: float) -> list[dict]:
    """Apply the filter Pipeline.note2midi applies when it writes the MIDI."""
    return [n for n in notes if n["offset"] - n["onset"] >= min_length]


def stream_to_notes(pipe: Pipeline, stream: Stream) -> list[dict]:
    threshold = config.infer.threshold
    notes = pipe.mpe2note(
        stream.onset,
        stream.offset,
        stream.frame,
        stream.velocity,
        thred_onset=threshold.onset,
        thred_offset=threshold.offset,
        thred_mpe=threshold.frame,
    )
    return drop_short_notes(notes, config.infer.min_duration)


def batch_reference(
    pipe: Pipeline, path: Path, sv: np.ndarray
) -> tuple[list[dict], tuple[np.ndarray, ...]]:
    """Run Pipeline.wav2midi, capturing the notes it would write.

    Capturing at note2midi keeps exact note times and avoids a MIDI parser. The
    transcript arrays are captured too, to diagnose any stitching difference.
    """
    captured = {}
    original_transcript = pipe.transcript

    def capture_transcript(*args, **kwargs):
        captured["arrays"] = original_transcript(*args, **kwargs)[4:8]
        return (None, None, None, None, *captured["arrays"])

    def capture_notes(notes, _path, min_length=0.0):
        captured["notes"] = drop_short_notes(notes, min_length)

    pipe.transcript = capture_transcript
    pipe.note2midi = capture_notes
    try:
        pipe.wav2midi(str(path), os.devnull, sv)
    finally:
        del pipe.transcript, pipe.note2midi
    return captured["notes"], captured["arrays"]


def notes_to_arrays(notes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array(
        [[n["onset"], n["offset"]] for n in notes], dtype=np.float64
    ).reshape(-1, 2)
    pitches = mir_eval.util.midi_to_hz(
        np.array([n["pitch"] for n in notes], dtype=np.float64)
    )
    return intervals, pitches


def score(
    reference: list[dict], estimate: list[dict], stream: Stream, stride: int, delay: int = 0
) -> Row:
    ref = notes_to_arrays(reference)
    est = notes_to_arrays(estimate)
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        *ref, *est, onset_tolerance=ONSET_TOLERANCE_SEC, offset_ratio=None
    )
    _, _, f1_with_offset, _ = mir_eval.transcription.precision_recall_f1_overlap(
        *ref, *est, onset_tolerance=ONSET_TOLERANCE_SEC, offset_ratio=OFFSET_RATIO
    )
    return Row(
        stride * HOP_SEC,
        delay * HOP_SEC * 1000,
        precision,
        recall,
        f1,
        f1_with_offset,
        stream.passes,
        stream.mean_pass_ms,
    )


def verify_stitching(
    name: str, row: Row, stream: Stream, arrays: tuple[np.ndarray, ...]
) -> bool:
    """The batch-window stride has to reproduce batch mode exactly."""
    max_diff = max(
        float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        for a, b in zip(
            (stream.onset, stream.offset, stream.frame, stream.velocity), arrays
        )
    )
    passed = row.f1 == 1.0 and row.f1_with_offset == 1.0
    print(
        f"  [verify] {name}: F1={row.f1:.6f} "
        f"F1(with offset)={row.f1_with_offset:.6f} "
        f"max output difference={max_diff:.3g} -> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def format_rows(rows: list[Row]) -> str:
    lines = [
        "| stride | delay ms | est. lag ms | precision | recall | F1 | passes | mean pass ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        passes = f"{r.passes:.0f}" if float(r.passes).is_integer() else f"{r.passes:.1f}"
        est_lag = LAG_FLOOR_MS + r.stride_sec * 1000 + r.delay_ms
        lines.append(
            f"| {r.stride_sec:.3f} s | {r.delay_ms:.0f} | {est_lag:.0f} "
            f"| {r.precision:.4f} | {r.recall:.4f} "
            f"| {r.f1:.4f} | {passes} | {r.mean_pass_ms:.1f} |"
        )
    return "\n".join(lines)


def average_rows(all_rows: list[list[Row]]) -> list[Row]:
    """Average each setting across songs, weighting every song equally."""
    return [
        Row(
            group[0].stride_sec,
            group[0].delay_ms,
            *(
                float(np.mean([getattr(r, field) for r in group]))
                for field in (
                    "precision",
                    "recall",
                    "f1",
                    "f1_with_offset",
                    "passes",
                    "mean_pass_ms",
                )
            ),
        )
        for group in zip(*all_rows)
    ]


def machine_description() -> str:
    """CPU, GPU and OS, for results.md."""
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
    return f"{cpu}, {gpu}, {platform.system()} {platform.release()}"


def append(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "test_audio")
    parser.add_argument("--output", type=Path, default=ROOT / "results.md")
    parser.add_argument("--strides", type=float, nargs="+", default=DEFAULT_STRIDES)
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[0],
        help="Delays in frames (16 ms each). Pairs with stride + delay over 512 frames are skipped.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--style", default="level2")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run only the 8.192 s stitching check and write nothing.",
    )
    args = parser.parse_args()
    if min(args.delays) < 0:
        parser.error("--delays can't be negative")

    wavs = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        print(f"No wav files in {args.audio_dir}", file=sys.stderr)
        return 2

    strides = [stride_to_frames(s) for s in args.strides]
    settings = [
        (s, d) for s in strides for d in args.delays if s + d <= BATCH_STRIDE_FRAMES
    ]
    if (BATCH_STRIDE_FRAMES, 0) not in settings:
        settings.append((BATCH_STRIDE_FRAMES, 0))  # The check always runs.

    device = torch.device(args.device) if args.device else None
    seed_all(args.seed)
    pipe = Pipeline(device=device) if device else Pipeline()
    sv_array = SVSampler().sample(args.style)
    sv = torch.tensor(sv_array).unsqueeze(0).to(pipe.device).to(torch.float32)

    # Check every song first, so nothing is reported from a broken stitch.
    songs = []
    for path in wavs:
        print(f"{path.name}: batch reference and {BATCH_STRIDE_FRAMES * HOP_SEC:.3f} s stride", flush=True)
        feature = pipe.wav2feature(str(path)).numpy().astype(np.float32)
        seed_all(args.seed)
        reference, arrays = batch_reference(pipe, path, sv_array)
        if not reference:
            print(f"  {path.name}: batch produced no notes, so the check can't pass", file=sys.stderr)
            return 1
        seed_all(args.seed)
        stream = simulate_stream(pipe, feature, sv, BATCH_STRIDE_FRAMES)
        row = score(reference, stream_to_notes(pipe, stream), stream, BATCH_STRIDE_FRAMES)
        if not verify_stitching(path.name, row, stream, arrays):
            print("Stitching check failed. No results written.", file=sys.stderr)
            return 1
        songs.append((path, feature, reference, row))

    if args.verify_only:
        print("Stitching check passed for every song.")
        return 0

    append(
        args.output,
        f"\n## Stream simulation, {datetime.now():%Y-%m-%d %H:%M}\n\n"
        f"Command: `{subprocess.list2cmdline(['python', *sys.argv])}`\n\n"
        f"Machine: {machine_description()}\n\n"
        f"Style {args.style}, seed {args.seed}, device {pipe.device}, "
        f"{torch.get_num_threads()} torch threads. "
        f"Reference: Pipeline.wav2midi notes. Match: pitch and onset within "
        f"{ONSET_TOLERANCE_SEC * 1000:.0f} ms, offsets ignored.\n",
    )

    all_rows = []
    for path, feature, reference, verified_row in songs:
        print(f"{path.name}: {len(feature)} frames, {len(reference)} reference notes", flush=True)
        rows = []
        for stride, delay in settings:
            if (stride, delay) == (BATCH_STRIDE_FRAMES, 0):
                row = verified_row
            else:
                seed_all(args.seed)
                stream = simulate_stream(pipe, feature, sv, stride, delay)
                row = score(reference, stream_to_notes(pipe, stream), stream, stride, delay)
            rows.append(row)
            print(
                f"  stride {row.stride_sec:.3f} s, delay {row.delay_ms:.0f} ms: P={row.precision:.4f} "
                f"R={row.recall:.4f} F1={row.f1:.4f} "
                f"F1(with offset)={row.f1_with_offset:.4f} "
                f"passes={row.passes:.0f} mean pass={row.mean_pass_ms:.1f} ms",
                flush=True,
            )
        all_rows.append(rows)
        append(
            args.output,
            f"\n### {path.name}\n\n"
            f"{len(feature) * HOP_SEC:.1f} s, {len(reference)} reference notes.\n\n"
            f"{format_rows(rows)}\n",
        )

    append(
        args.output,
        f"\n### Average ({len(all_rows)} songs)\n\n"
        f"{format_rows(average_rows(all_rows))}\n",
    )
    print(f"Appended to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
