r"""Largest attention score in every attention layer, in FP32, on real audio.

    .venv\Scripts\python.exe py\attention_range.py runs\check512 [more run folders]

Each folder is an engine run made with --dump (it holds mel.f32). The score is query x key
before the divide by sqrt(d) = 8. FP16 can hold at most 65504, so a layer whose scores come
near that cannot run in FP16. py/export_fp16.py keeps those layers' scores in FP32
(FP32_SCORE_LAYERS there); this script is where that list comes from. Runs on the CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent.parent
FP16_MAX = 65504.0


def with_scores(path: Path):
    """The graph with every attention score MatMul added as an output."""
    model = onnx.load(str(path))
    softmax_in = {n.input[0] for n in model.graph.node if n.op_type == "Softmax"}
    div_in = {n.input[0] for n in model.graph.node if n.op_type == "Div" and n.output[0] in softmax_in}
    names = [n.output[0] for n in model.graph.node if n.op_type == "MatMul" and n.output[0] in div_in]
    for name in names:
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]), names


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    frame, frame_names = with_scores(ROOT / "models" / "frame.onnx")
    time_, time_names = with_scores(ROOT / "models" / "time.onnx")
    sv = np.fromfile(ROOT / "models" / "sv_level2.bin", "<f4").reshape(1, 24)
    worst: dict[str, float] = {}
    for run in sys.argv[1:]:
        mel = np.fromfile(Path(run) / "mel.f32", "<f4").reshape(-1, 256)
        windows = range(0, len(mel) - 576 + 1, 512)
        print(f"{run}: {len(mel)} frames, {len(windows)} windows", flush=True)
        for start in windows:
            spec = np.ascontiguousarray(mel[start:start + 576].T)[None]
            out = frame.run(None, {"spec": spec, "sv": sv})
            scores = list(zip(frame_names, out[5:])) + list(zip(time_names, time_.run(None, {"h": out[0]})[4:]))
            for name, a in scores:
                worst[name] = max(worst.get(name, 0.0), float(np.abs(a).max()))
    print("| layer | largest score | share of the FP16 limit |\n|---|---|---|")
    for name, v in worst.items():
        print(f"| `{name.rsplit('/', 1)[0]}` | {v:.0f} | {v / FP16_MAX:.1%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
