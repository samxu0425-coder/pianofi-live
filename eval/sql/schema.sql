-- Results store for PianoFi Live engine runs (SQLite, file runs/results.db).
-- One row in `runs` per run folder (its metrics.json). A run can be scored
-- against batch more than once (`scores`), and has one `timings` row per
-- timed stage (pass, frame model, time model, decode, lag, jitter).

CREATE TABLE IF NOT EXISTS runs (
    run_id             INTEGER PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,   -- run folder name under runs/
    wav                TEXT NOT NULL,
    input              TEXT NOT NULL,          -- file, mic, loopback
    provider           TEXT NOT NULL,          -- CUDA or CPU
    stride_frames      INTEGER NOT NULL,       -- 1 frame = 16 ms
    delay_frames       INTEGER NOT NULL DEFAULT 0,
    heads              TEXT NOT NULL,          -- a or b
    cache              INTEGER NOT NULL,       -- 1 = per-frame cache on
    ring_buffer        TEXT NOT NULL,          -- spsc or mutex
    realtime           INTEGER NOT NULL,       -- 0 = --fast (no pacing, no lag)
    normalize          INTEGER NOT NULL,       -- automatic input level
    audio_sec          REAL NOT NULL,
    wall_sec           REAL NOT NULL,
    compute_sec        REAL NOT NULL,
    real_time_factor   REAL NOT NULL,
    passes             INTEGER NOT NULL,
    notes              INTEGER NOT NULL,
    overrun_samples    INTEGER NOT NULL,       -- audio samples dropped
    peak_memory_mb     REAL,
    max_decoder_frames INTEGER,
    gpu_memory_mb      INTEGER,                -- peak nvidia-smi reading, when sampled
    machine            TEXT,                   -- CPU, GPU, OS
    command            TEXT NOT NULL,
    imported_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS timings (
    run_id  INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    stage   TEXT NOT NULL,     -- pass_ms, frame_model_ms, time_model_ms, decode_ms,
                               -- frame_lag_ms, note_lag_ms, push_jitter_ms, gain
    n       INTEGER NOT NULL,
    mean    REAL NOT NULL,
    p50     REAL NOT NULL,
    p95     REAL NOT NULL,
    p99     REAL NOT NULL,
    max     REAL NOT NULL,
    PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS scores (
    score_id    INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    scored_at   TEXT NOT NULL,
    precision   REAL NOT NULL,
    recall      REAL NOT NULL,
    f1          REAL NOT NULL,                 -- mir_eval note F1 vs batch, 50 ms onsets
    engine_notes INTEGER,
    batch_notes  INTEGER,
    decoder_ok  INTEGER,                       -- C++ decoder == mpe2note on the same arrays
    array_ok    INTEGER,                       -- stride 512 only: arrays == batch
    source      TEXT NOT NULL                  -- score_engine or results.md
);

-- Lag is only meaningful for paced runs that dropped nothing. The view is
-- recreated on every connect so edits here take effect in existing databases.
DROP VIEW IF EXISTS run_summary;
CREATE VIEW run_summary AS
SELECT r.run_id, r.name, r.wav, r.provider, r.machine,
       CASE WHEN r.provider = 'CPU'            THEN 'CPU only'
            WHEN r.machine LIKE '%Tesla T4%'   THEN 'T4 (AWS)'
            WHEN r.machine LIKE '%RTX 4070%'   THEN 'RTX 4070' || CASE WHEN r.machine LIKE '%Docker%' THEN ' (Docker)' ELSE '' END
            ELSE r.machine END AS gpu,
       r.stride_frames * 16 AS stride_ms,
       r.delay_frames * 16  AS delay_ms,
       r.heads, r.cache, r.ring_buffer, r.realtime, r.normalize,
       r.notes, r.overrun_samples, r.real_time_factor, r.gpu_memory_mb,
       p.p50 AS pass_p50_ms, p.p99 AS pass_p99_ms,
       CASE WHEN r.realtime AND r.overrun_samples = 0 THEN l.p50 END AS lag_p50_ms,
       CASE WHEN r.realtime AND r.overrun_samples = 0 THEN l.p99 END AS lag_p99_ms,
       s.precision, s.recall, s.f1, s.decoder_ok, s.scored_at
FROM runs r
LEFT JOIN timings p ON p.run_id = r.run_id AND p.stage = 'pass_ms'
LEFT JOIN timings l ON l.run_id = r.run_id AND l.stage = 'frame_lag_ms'
LEFT JOIN scores  s ON s.score_id = (SELECT score_id FROM scores
                                     WHERE run_id = r.run_id ORDER BY scored_at DESC LIMIT 1);
