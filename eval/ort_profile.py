r"""Read the ONNX Runtime profiles an engine run made with --profile, and show where a pass goes.

    .venv\Scripts\python.exe eval\ort_profile.py runs\prof_fp16_rt [more run folders] [--graph frame|time] [--top 12]

For each run folder and graph: the median time of one model run, the same split by execution
provider (CUDA nodes, CPU nodes, copies between them), and the slowest nodes (median over the
runs). Times are what the CPU thread waited at each node. CUDA calls return before the GPU
has finished, so GPU work shows up at the next node that has to wait for it (often a copy to
the CPU), not always at the node that caused it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(run: Path, graph: str):
    files = sorted(run.glob(f"ort_{graph}_*.json"))
    if not files:
        raise SystemExit(f"no ort_{graph}_*.json in {run}: run the engine with --profile")
    events = json.loads(files[-1].read_text())
    runs = [e["dur"] for e in events if e["cat"] == "Session" and e["name"] == "model_run"]
    nodes = defaultdict(list)  # node name -> [dur per run]
    info = {}
    for e in events:
        if e["cat"] == "Node" and e["name"].endswith("_kernel_time"):
            name = e["name"][: -len("_kernel_time")]
            nodes[name].append(e["dur"])
            info[name] = (e["args"]["op_name"], e["args"].get("provider", "?").replace("ExecutionProvider", ""))
    return runs, nodes, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--graph", default="frame", choices=["frame", "time"])
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    for run in args.runs:
        runs, nodes, info = load(run, args.graph)
        med = {name: statistics.median(d) for name, d in nodes.items()}
        by = defaultdict(float)
        count = defaultdict(int)
        for name, us in med.items():
            op, provider = info[name]
            key = "copies CPU<->GPU" if op.startswith("Memcpy") else f"{provider} nodes"
            by[key] += us
            count[key] += 1
        print(f"\n## {run.name}, {args.graph} graph: {len(runs)} runs, median run {statistics.median(runs) / 1000:.1f} ms, "
              f"sum of node medians {sum(med.values()) / 1000:.1f} ms")
        for key in sorted(by, key=by.get, reverse=True):
            print(f"  {key:18s} {by[key] / 1000:7.2f} ms in {count[key]} nodes")
        print("  | node | op | provider | median ms | p90 ms |\n  |---|---|---|---|---|")
        for name in sorted(med, key=med.get, reverse=True)[: args.top]:
            d = sorted(nodes[name])
            print(f"  | `{name}` | {info[name][0]} | {info[name][1]} | {med[name] / 1000:.2f} | {d[int(len(d) * 0.9)] / 1000:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
