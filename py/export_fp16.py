r"""Make half-precision (FP16) copies of the two ONNX graphs.

    .venv\Scripts\python.exe py\export_fp16.py

Reads models/frame.onnx and models/time.onnx (from py/export_onnx.py) and writes
models/frame_fp16.onnx and models/time_fp16.onnx. Weights and the math inside become
16-bit floats; the inputs and outputs stay 32-bit (a Cast at each edge), so the engine
feeds and reads them exactly like the FP32 graphs. Run the engine with --fp16 to use them.

Mixed precision: in two attention layers the scores (query x key, the divide by sqrt(d)
and the softmax) stay in FP32, because they are larger than FP16 can hold (65504) and would
turn every output into NaN. Those are the first layer of the frequency encoder and the first
layer of the time model, the two that see inputs no LayerNorm has scaled yet. Measured with
py/attention_range.py on two songs: scores up to 394,824 and 1,848,219 there, and at most 256
in the other nine layers. Everything else, including all weights, is FP16.

This is not exact: FP16 moves the outputs away from batch, so the stride 512 array check
is expected to fail with it. FP32 stays the reference; eval/score_engine.py measures what
FP16 costs in F1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent


# Attention layers whose scores overflow FP16 (py/attention_range.py). The trailing slash
# matters: "/layers_freq.0/self_attention_1/" is a different layer, and is fine in FP16.
FP32_SCORE_LAYERS = ("/layers_freq.0/self_attention/", "/layers_time.0/self_attention/")


def attention_score_nodes(model: onnx.ModelProto, layers: tuple[str, ...]) -> list[str]:
    """Names of the Softmax, Div and MatMul (the attention scores) of the given layers."""
    made_by = {out: node for node in model.graph.node for out in node.output}
    names = []
    for node in model.graph.node:
        if node.op_type != "Softmax" or not node.name.startswith(layers):
            continue
        names.append(node.name)
        cur = made_by.get(node.input[0])
        while cur is not None and cur.op_type in ("Div", "MatMul"):
            names.append(cur.name)
            if cur.op_type == "MatMul":
                break
            cur = made_by.get(cur.input[0])
    return names


def check(src: Path, dst: Path) -> None:
    """Run both graphs on the CPU with the same random input: FP16 must be finite, and close."""
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    full = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    half = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in full.get_inputs():
        shape = [d if isinstance(d, int) else 96 for d in inp.shape]  # 96 input frames = 32 output frames
        feeds[inp.name] = rng.uniform(-18, 2, shape).astype(np.float32) if inp.name == "spec" else rng.normal(0, 1, shape).astype(np.float32)
    worst = 0.0
    for a, b, out in zip(full.run(None, feeds), half.run(None, feeds), full.get_outputs()):
        if a.dtype.kind != "f":
            continue
        if not np.isfinite(b).all():
            raise SystemExit(f"FAILED: {dst.name} output {out.name} has NaN or inf")
        if out.name != "h":  # h is not a probability: its scale is not 0..1
            worst = max(worst, float(np.abs(a - b).max()))
    print(f"  random input, CPU: all outputs finite, largest probability difference vs FP32 {worst:.2e}")


def convert(src: Path, dst: Path, layers: tuple[str, ...]) -> None:
    model = onnx.load(str(src))
    half = float16.convert_float_to_float16(model, keep_io_types=True, node_block_list=attention_score_nodes(model, layers))
    # The converter puts a Cast to float32 in front of every graph output. h is an output
    # and also feeds the A heads inside the graph, and those would now get the float32
    # copy (a type error at load). Point them at the float16 tensor in front of the Cast.
    outputs = {o.name for o in half.graph.output}
    before_cast = {n.output[0]: n.input[0] for n in half.graph.node if n.op_type == "Cast" and n.output[0] in outputs}
    for node in half.graph.node:
        for i, name in enumerate(node.input):
            if name in before_cast:
                node.input[i] = before_cast[name]
    onnx.checker.check_model(half)
    onnx.shape_inference.infer_shapes(half, check_type=True, strict_mode=True)  # raises on a type mismatch
    onnx.save(half, str(dst))
    kinds = Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in half.graph.initializer)
    casts = sum(node.op_type == "Cast" for node in half.graph.node)
    print(f"{src.name} ({src.stat().st_size / 1e6:.1f} MB) -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB), "
          f"weights {dict(kinds)}, {casts} Cast nodes")
    check(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=ROOT / "models")
    parser.add_argument("--all-layers", action="store_true",
                        help="keep the scores of every attention layer in FP32 (slower; for comparison)")
    args = parser.parse_args()
    layers = ("/",) if args.all_layers else FP32_SCORE_LAYERS
    for name in ("frame", "time"):
        convert(args.dir / f"{name}.onnx", args.dir / f"{name}_fp16.onnx", layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
