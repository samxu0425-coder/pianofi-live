"""Export AMT-APC as two ONNX graphs for the streaming engine, and check them.

Why two graphs
--------------
Everything in AMT-APC up to the "time" stage is per-frame: output frame t
depends only on input frames [t, t + 65) (32 before, itself, 32 after), and on
no other frame. That covers the CNN, the frequency encoder, the style-vector
gate and the first decoder stage (CAfreq). Only the last stage (SAtime) attends
across all 512 frames of a window.

So the streaming engine can compute the per-frame part once per new frame and
cache it, then rerun only SAtime over the window each pass:

    frame.onnx  spec [1, 256, S + 64], sv [1, 24]
                -> h [S, 88, 256]            (cache this, one row per frame)
                -> onset_a/offset_a/mpe_a [S, 88], velocity_a [S, 88] int64
                   (the per-frame "A" heads: a zero-context, low-lag option)
    time.onnx   h [512, 88, 256]
                -> onset/offset/mpe [512, 88], velocity [512, 88] int64
                   (the "B" heads that Pipeline.wav2midi uses)

Checks run here (all must pass before the files are written):
  1. Wrappers vs the original model on the same window (B and A outputs).
  2. Per-frame independence: frame model on 512 frames at once == in chunks.
     This is what makes caching exact.
  3. ONNX Runtime vs PyTorch for both graphs.

Usage (from the repo root):
    .venv\\Scripts\\python.exe py\\export_onnx.py
    .venv\\Scripts\\python.exe py\\export_onnx.py --wav test_audio\\song.wav
Writes models/frame.onnx, models/time.onnx, models/sv_<style>.bin, models/meta.json
"""

from __future__ import annotations

import subprocess
import sys

# amt-apc reads its JSON with the platform encoding, which breaks on Windows.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "amt-apc"))
sys.path.insert(0, str(ROOT / "py"))

import numpy as np
import torch
import torch.nn as nn

from data import SVSampler
from models import Pipeline
from utils import config

import stream_sim  # noqa: F401  (patches torchaudio.load with its wav reader)

CFG = config.data
MARGIN = CFG.input.margin_b  # 32, same as margin_f
N_PROC = 2 * MARGIN + 1  # 65 input frames per output frame
N_FRAME = CFG.input.num_frame  # 512
N_BIN = CFG.feature.n_bins  # 256
N_NOTE = CFG.midi.num_note  # 88


class FrameModel(nn.Module):
    """Per-frame part: CNN + freq encoder + style gate + CAfreq + A heads."""

    def __init__(self, model):
        super().__init__()
        self.enc = model.encoder
        self.dec = model.decoder
        self.fc_sv = model.fc_sv
        self.gate_sv = model.gate_sv

    def forward(self, spec, sv):
        # spec: [1, n_bin, S + 64]. Replaces Tensor.unfold with a gather so the
        # number of frames S can be dynamic in ONNX.
        n_out = spec.shape[2] - (N_PROC - 1)
        idx = torch.arange(n_out).unsqueeze(1) + torch.arange(N_PROC).unsqueeze(0)
        x = spec[0][:, idx]  # [n_bin, S, 65]
        x = x.permute(1, 0, 2).unsqueeze(1)  # [S, 1, n_bin, 65]

        enc = self.enc
        x = enc.conv(x).permute(0, 2, 1, 3).contiguous()  # [S, n_bin, ch, 61]
        x = x.reshape(n_out, N_BIN, enc.cnn_dim)
        pos = torch.arange(N_BIN)
        x = enc.tok_embedding_freq(x) * enc.scale_freq + enc.pos_embedding_freq(pos)
        for layer in enc.layers_freq:
            x = layer(x)  # [S, n_bin, hid]

        # Style-vector gate (Spec2MIDI.encode), per element so per frame.
        s = self.fc_sv(sv)  # [1, hid]
        z = self.gate_sv(x)
        x = z * x + (1 - z) * s.unsqueeze(1)

        # CAfreq (Decoder_SPEC2MIDI.forward, first half).
        dec = self.dec
        q = dec.pos_embedding_freq(torch.arange(N_NOTE)).unsqueeze(0)
        q = q.expand(n_out, N_NOTE, enc.hid_dim)
        h, _ = dec.layer_zero_freq(x, q)
        for layer in dec.layers_freq:
            h, _ = layer(x, h)  # [S, 88, hid]

        onset_a = torch.sigmoid(dec.fc_onset_freq(h).squeeze(-1))
        offset_a = torch.sigmoid(dec.fc_offset_freq(h).squeeze(-1))
        mpe_a = torch.sigmoid(dec.fc_mpe_freq(h).squeeze(-1))
        velocity_a = argmax_last(dec.fc_velocity_freq(h))
        return h, onset_a, offset_a, mpe_a, velocity_a


class TimeModel(nn.Module):
    """SAtime + B heads over one 512-frame window of cached h."""

    def __init__(self, model):
        super().__init__()
        self.dec = model.decoder

    def forward(self, h):
        dec = self.dec
        x = h.permute(1, 0, 2)  # [88, 512, hid]
        pos = torch.arange(N_FRAME)
        x = x * dec.scale_time + dec.pos_embedding_time(pos)
        for layer in dec.layers_time:
            x = layer(x)
        onset = torch.sigmoid(dec.fc_onset_time(x).squeeze(-1)).transpose(0, 1)
        offset = torch.sigmoid(dec.fc_offset_time(x).squeeze(-1)).transpose(0, 1)
        mpe = torch.sigmoid(dec.fc_mpe_time(x).squeeze(-1)).transpose(0, 1)
        velocity = argmax_last(dec.fc_velocity_time(x)).transpose(0, 1)
        return onset, offset, mpe, velocity


USE_ARGMAX = False  # --argmax: export the plain ArgMax node, to compare against


def argmax_last(x):
    """x.argmax(-1), written as top-1 so it exports as TopK instead of ArgMax.

    ONNX Runtime's CUDA provider runs TopK on the GPU but sends this ArgMax to the CPU, where
    it waits on the CPU thread pool for 5 to 19 ms a pass (results.md, "Profile of a pass").
    Same result: both return the lowest index among equal values.
    """
    if USE_ARGMAX:
        return x.argmax(-1)
    return torch.topk(x, 1, dim=-1).indices.squeeze(-1)


def max_diff(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def check_wrappers(model, frame_model, time_model, window, sv, tol):
    """Original model vs FrameModel + TimeModel on one 576-frame window."""
    with torch.no_grad():
        ref = model(window, sv)
        h, *a_out = frame_model(window, sv)
        b_out = time_model(h)
    names = ("onset", "offset", "mpe")
    ok = True
    for i, name in enumerate(names):
        d_b = max_diff(ref[5 + i].squeeze(0), b_out[i])
        d_a = max_diff(ref[i].squeeze(0), a_out[i])
        print(f"  {name}: B max diff {d_b:.2e}, A max diff {d_a:.2e}")
        ok &= d_b < tol and d_a < tol
    vel_b = (ref[8].squeeze(0).argmax(-1) != b_out[3]).float().mean().item()
    vel_a = (ref[3].squeeze(0).argmax(-1) != a_out[3]).float().mean().item()
    print(f"  velocity argmax mismatch: B {vel_b:.4%}, A {vel_a:.4%}")
    ok &= vel_b < 1e-3 and vel_a < 1e-3
    return ok


def check_per_frame(frame_model, window, sv, tol, chunk=32):
    """Frame model on all 512 frames at once == stitched from chunks."""
    with torch.no_grad():
        whole = frame_model(window, sv)[0]
        parts = []
        for start in range(0, N_FRAME, chunk):
            parts.append(frame_model(window[:, :, start:start + chunk + N_PROC - 1], sv)[0])
    d = max_diff(whole, torch.cat(parts))
    print(f"  chunked ({chunk} frames) vs whole: max diff {d:.2e}")
    return d < tol


def export(module, args, path, input_names, output_names, dynamic_axes):
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch has no dynamo flag
        torch.onnx.export(module, args, str(path), **kwargs)


def check_onnx(frame_path, time_path, frame_model, time_model, window, sv, tol):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    fs = ort.InferenceSession(str(frame_path), opts, providers=providers)
    ts = ort.InferenceSession(str(time_path), opts, providers=providers)
    with torch.no_grad():
        h_ref, *a_ref = frame_model(window, sv)
        b_ref = time_model(h_ref)
    f_out = fs.run(None, {"spec": window.numpy(), "sv": sv.numpy()})
    t_out = ts.run(None, {"h": f_out[0]})
    d_h = float(np.abs(f_out[0] - h_ref.numpy()).max())
    d_b = max(float(np.abs(t_out[i] - b_ref[i].numpy()).max()) for i in range(3))
    vel = float((t_out[3] != b_ref[3].numpy()).mean())
    print(f"  ONNX vs torch: h {d_h:.2e}, B probs {d_b:.2e}, velocity mismatch {vel:.4%}")

    # Time one pass of each on CPU, as a first speed number.
    def bench(fn, n=5):
        fn()
        ts_ = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts_.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts_))

    spec32 = window[:, :, : 32 + N_PROC - 1].numpy()
    ms_frame32 = bench(lambda: fs.run(None, {"spec": spec32, "sv": sv.numpy()}))
    ms_frame512 = bench(lambda: fs.run(None, {"spec": window.numpy(), "sv": sv.numpy()}), n=2)
    ms_time = bench(lambda: ts.run(None, {"h": f_out[0]}))
    print(
        f"  ONNX Runtime CPU median: frame model 32 frames {ms_frame32:.0f} ms, "
        f"512 frames {ms_frame512:.0f} ms, time model {ms_time:.0f} ms"
    )
    return d_h < tol and d_b < tol and vel < 1e-3, {
        "frame32_ms": ms_frame32,
        "frame512_ms": ms_frame512,
        "time_ms": ms_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=None, help="Use a real window from this song for the checks")
    parser.add_argument("--style", default="level2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--argmax", action="store_true",
                        help="export velocity as ArgMax (runs on the CPU in ONNX Runtime; the slow form, for comparison)")
    args = parser.parse_args()
    global USE_ARGMAX
    USE_ARGMAX = args.argmax

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pipe = Pipeline(device=torch.device("cpu"))
    model = pipe.model.eval()
    sv = torch.tensor(SVSampler().sample(args.style), dtype=torch.float32).unsqueeze(0)

    if args.wav:
        feature = pipe.wav2feature(str(args.wav)).numpy().astype(np.float32)
        start = min(len(feature) // 3, max(len(feature) - (N_FRAME + 2 * MARGIN), 0))
        win = np.full((N_FRAME + 2 * MARGIN, N_BIN), CFG.input.min_value, np.float32)
        part = feature[start:start + N_FRAME + 2 * MARGIN]
        win[: len(part)] = part
        source = f"{args.wav.name} from frame {start}"
    else:
        rng = np.random.default_rng(args.seed)
        win = rng.uniform(-18, 2, size=(N_FRAME + 2 * MARGIN, N_BIN)).astype(np.float32)
        source = "random input"
    window = torch.from_numpy(win.T.copy()).unsqueeze(0)  # [1, 256, 576]
    print(f"Checks on {source}")

    frame_model = FrameModel(model).eval()
    time_model = TimeModel(model).eval()

    print("1. Wrappers vs original model")
    if not check_wrappers(model, frame_model, time_model, window, sv, args.tol):
        print("FAILED: wrappers do not match the original model", file=sys.stderr)
        return 1
    print("2. Per-frame independence")
    if not check_per_frame(frame_model, window, sv, args.tol):
        print("FAILED: frame model output depends on other frames", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.out_dir / "frame.onnx"
    time_path = args.out_dir / "time.onnx"
    print("Exporting ONNX ...")
    with torch.no_grad():
        export(
            frame_model, (window, sv), frame_path, ["spec", "sv"],
            ["h", "onset_a", "offset_a", "mpe_a", "velocity_a"],
            {"spec": {2: "in_frames"}, "h": {0: "frames"}, "onset_a": {0: "frames"},
             "offset_a": {0: "frames"}, "mpe_a": {0: "frames"}, "velocity_a": {0: "frames"}},
        )
        h = frame_model(window, sv)[0]
        export(time_model, (h,), time_path, ["h"], ["onset", "offset", "mpe", "velocity"], None)

    print("3. ONNX Runtime vs PyTorch")
    ok, timing = check_onnx(frame_path, time_path, frame_model, time_model, window, sv, args.tol)
    if not ok:
        print("FAILED: ONNX output does not match PyTorch", file=sys.stderr)
        return 1

    sv_path = args.out_dir / f"sv_{args.style}.bin"
    sv.numpy().astype("<f4").tofile(sv_path)
    meta = {
        "sr": CFG.feature.sr,
        "hop": CFG.feature.hop_sample,
        "n_fft": CFG.feature.fft_bins,
        "n_mels": CFG.feature.mel_bins,
        "log_offset": CFG.feature.log_offset,
        "margin": MARGIN,
        "num_frame": N_FRAME,
        "min_value": CFG.input.min_value,
        "num_note": N_NOTE,
        "note_min": CFG.midi.note_min,
        "threshold": {"onset": config.infer.threshold.onset, "offset": config.infer.threshold.offset, "frame": config.infer.threshold.frame},
        "min_duration": config.infer.min_duration,
        "style": args.style,
        "seed": args.seed,
        "cpu_timing_ms": timing,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"All checks passed. Wrote {frame_path.name}, {time_path.name}, {sv_path.name}, meta.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
