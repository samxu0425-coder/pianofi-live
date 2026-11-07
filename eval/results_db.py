"""SQLite results store for engine runs: runs/results.db (schema in eval/sql/schema.sql).

    .venv\\Scripts\\python.exe eval\\results_db.py import            # every runs/*/metrics.json, plus scores from results.md
    .venv\\Scripts\\python.exe eval\\results_db.py report            # all reports in eval/sql/
    .venv\\Scripts\\python.exe eval\\results_db.py report by_setting
    .venv\\Scripts\\python.exe eval\\results_db.py sql "SELECT name, f1 FROM run_summary ORDER BY f1 DESC"

eval/score_engine.py calls record() after scoring, so new runs go in on their own.
Standard library only.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "runs" / "results.db"
SQL = ROOT / "eval" / "sql"
STAGES = ("pass_ms", "frame_model_ms", "time_model_ms", "decode_ms", "frame_lag_ms", "note_lag_ms", "push_jitter_ms", "gain")


def connect(path: Path = DB) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript((SQL / "schema.sql").read_text(encoding="utf-8"))
    return con


def upsert_run(con: sqlite3.Connection, name: str, m: dict, machine: str | None = None, gpu_mb: int | None = None) -> int:
    """Insert or refresh one run from its metrics.json. Returns run_id."""
    row = {
        "name": name, "wav": m["wav"], "input": m.get("input", "file"), "provider": m["provider"],
        "stride_frames": m["stride_frames"], "delay_frames": m.get("delay_frames", 0), "heads": m["heads"],
        "cache": int(m["cache"]), "ring_buffer": m.get("ring_buffer", "mutex"), "realtime": int(m["realtime"]),
        "normalize": int(m.get("normalize", False)), "audio_sec": m["audio_sec"], "wall_sec": m["wall_sec"],
        "compute_sec": m["compute_sec"], "real_time_factor": m["real_time_factor"], "passes": m["passes"],
        "notes": m["notes"], "overrun_samples": m["overrun_samples"], "peak_memory_mb": m.get("peak_memory_mb"),
        "max_decoder_frames": m.get("max_decoder_frames"), "gpu_memory_mb": gpu_mb, "machine": machine,
        "command": m["command"],
    }
    cols = ", ".join(row)
    marks = ", ".join(f":{c}" for c in row)
    # Keep a machine or GPU reading already stored if this import has none.
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, runs.{c})" if c in ("machine", "gpu_memory_mb") else f"{c} = excluded.{c}"
        for c in row if c != "name"
    )
    con.execute(f"INSERT INTO runs ({cols}) VALUES ({marks}) ON CONFLICT(name) DO UPDATE SET {updates}", row)
    run_id = con.execute("SELECT run_id FROM runs WHERE name = ?", (name,)).fetchone()[0]
    con.execute("DELETE FROM timings WHERE run_id = ?", (run_id,))
    for stage in STAGES:
        t = m.get(stage)
        if t and t.get("n", 0) > 0:
            con.execute(
                "INSERT INTO timings (run_id, stage, n, mean, p50, p95, p99, max) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, stage, t["n"], t["mean"], t["p50"], t["p95"], t["p99"], t["max"]),
            )
    return run_id


def add_score(con: sqlite3.Connection, run_id: int, scored_at: str, p: float, r: float, f: float,
              engine_notes: int | None, batch_notes: int | None, decoder_ok: bool | None,
              array_ok: bool | None, source: str) -> None:
    con.execute(
        "INSERT INTO scores (run_id, scored_at, precision, recall, f1, engine_notes, batch_notes, decoder_ok, array_ok, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, scored_at, p, r, f, engine_notes, batch_notes,
         None if decoder_ok is None else int(decoder_ok), None if array_ok is None else int(array_ok), source),
    )


def record(run: Path, metrics: dict, machine: str, p: float, r: float, f: float,
           engine_notes: int, batch_notes: int, decoder_ok: bool, array_ok: bool | None) -> None:
    """Called by score_engine.py: store the run and this score."""
    with connect() as con:
        run_id = upsert_run(con, run.name, metrics, machine, read_gpu_mb(run))
        add_score(con, run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), p, r, f,
                  engine_notes, batch_notes, decoder_ok, array_ok, "score_engine")


def read_gpu_mb(run: Path) -> int | None:
    """Peak of the nvidia-smi samples RUN.md step 9 writes next to the run folder, if any."""
    f = run.parent / f"{run.name}.gpu_mb.txt"
    if not f.exists():
        return None
    values = [int(x) for x in f.read_text().split() if x.isdigit()]
    return max(values) if values else None


def results_md_scores() -> dict[str, list[dict]]:
    """Scores already in results.md's engine table, keyed by run folder name (from --out in the command)."""
    text = (ROOT / "results.md").read_text(encoding="utf-8")
    start = text.index("## C++ engine runs")
    end = text.find("\n## ", start + 1)
    out: dict[str, list[dict]] = {}
    for line in text[start:end if end != -1 else None].splitlines():
        if not re.match(r"\| 20\d\d-", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        m = re.search(r"--out\s+(\S+)", c[-1])
        if not m or len(c) < 17:
            continue
        name = m.group(1).replace("\\", "/").rstrip("`").split("/")[-1]
        out.setdefault(name, []).append({
            "scored_at": c[0] + ":00", "p": float(c[6]), "r": float(c[7]), "f1": float(c[8]),
            "decoder_ok": c[15] == "ok", "machine": c[16],
        })
    return out


def import_all(con: sqlite3.Connection) -> None:
    md = results_md_scores()
    n_runs = n_scores = 0
    for metrics_path in sorted((ROOT / "runs").glob("*/metrics.json")):
        run = metrics_path.parent
        m = json.loads(metrics_path.read_text())
        rows = md.get(run.name, [])
        machine = rows[-1]["machine"] if rows else None
        run_id = upsert_run(con, run.name, m, machine, read_gpu_mb(run))
        n_runs += 1
        con.execute("DELETE FROM scores WHERE run_id = ? AND source = 'results.md'", (run_id,))
        for s in rows:
            add_score(con, run_id, s["scored_at"], s["p"], s["r"], s["f1"], m["notes"], None, s["decoder_ok"], None, "results.md")
            n_scores += 1
    print(f"imported {n_runs} runs and {n_scores} scores from results.md into {DB}")


def table(cur: sqlite3.Cursor) -> str:
    """Rows as a markdown table (or the row count for an UPDATE/DELETE)."""
    if cur.description is None:
        return f"{cur.rowcount} rows changed"
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    def fmt(v):
        if v is None:
            return "not measured"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) <= 1 else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def report(con: sqlite3.Connection, names: list[str]) -> None:
    files = [SQL / f"{n}.sql" for n in names] if names else sorted(f for f in SQL.glob("*.sql") if f.name != "schema.sql")
    for f in files:
        sql = f.read_text(encoding="utf-8")
        title = sql.splitlines()[0].lstrip("- ").strip()
        print(f"\n### {f.stem}: {title}\n")
        print(table(con.execute(sql)))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    with connect() as con:
        if argv[0] == "import":
            import_all(con)
        elif argv[0] == "report":
            report(con, argv[1:])
        elif argv[0] == "sql":
            print(table(con.execute(" ".join(argv[1:]))))
        else:
            print(__doc__)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
