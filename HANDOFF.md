# HANDOFF: status and task queue (written 2026-11-05)

Read this after CLAUDE.md. Do the tasks in order, one per session or prompt.
Stop after each task, show Sam the output, and wait for a go-ahead.
Sam prefers short, informal answers and no em dashes.

## Goal
Live version of PianoFi: any song playing → piano-cover notes scrolling on screen about 1 s behind.
C++ engine for the streaming part. It's a resume project, so every number must be measured, and
Sam has to be able to explain every line in an interview.

## Status

### Verified on Sam's PC (RTX 4070, CUDA torch)
`py/stream_sim.py` (Python streaming simulation), results in results.md:
- Stitching is exact: stride 8.192 s reproduces batch with max output difference 0.
- Agreement with batch notes (mir_eval note F1, 50 ms onsets, offsets ignored), average of 2 longer songs
  (42 s piano + 119 s pop song with 1495 notes):
  stride 0.256 s F1 0.917 | 0.512 s 0.933 | 1.024 s 0.940 | 8.192 s 1.000
- Both precision and recall drop (~0.91 to 0.95), so live mode misses some batch notes AND adds some.
- Full-window pass on GPU without caching: ~150 ms at every stride.
Wording: this is agreement with batch output, not accuracy against ground truth.

### Written and tested in a Linux sandbox, NOT yet run on Sam's PC
Everything in RUN.md: py/export_onnx.py, py/prep_audio.py, py/check_mel.py, cpp/ (engine),
eval/score_engine.py, eval/sweep.py. Sandbox checks: C++ unit tests pass; C++ mel matched a
torch.stft + HTK/slaney reference to 2.5e-5; engine output (cached, uncached, odd strides, A heads)
matched a Python stream simulation exactly using stand-in ONNX models of the same shape; the note
decoder matched a Python transcription of mpe2note on 40 fuzz cases. Nothing has run with the
real AMT-APC weights yet. Expect small Windows/MSVC build fixes.

## Key facts (so you don't re-derive them)
- AMT-APC = hFT-Transformer. Everything up to the SAtime stage is per-frame (each output frame
  depends only on input frames t-32..t+32), so its result `h` can be cached per frame. Only
  SAtime (3 layers over the 512-frame window, ~20% of compute) must rerun each pass.
- Lag floor ≈ 576 ms: 512 ms model lookahead (32 frames) + 64 ms mel window. Real lag adds up to
  one stride of waiting plus compute. Expected ~0.75 to 1.0 s at stride 0.256 s. Measured lag comes
  from the C++ engine (`frame_lag_ms` in metrics.json), not stream_sim.
- Hypothesis for the F1 gap (unproven): in live mode every emitted frame sits at the newest edge
  of the window, with no later frames for SAtime to attend to. The delay experiment tests this.
- Deployment GPUs differ (T4 ≈ 3x slower than the 4070, A10G ≈ same). Caching matters most there:
  concurrent users per GPU ≈ stride / pass time.

## Task queue

### 1. RUN.md steps 1 to 4 (export, build, correctness checks)
Done when: export prints "All checks passed"; pf_tests passes; check_mel PASSED; at --stride 512,
score_engine prints array check PASSED, decoder check PASSED, F1 = 1.0000.
Fix build errors in cpp/ as needed. If a numeric check fails, explain the cause first.

### 2. RUN.md step 5 (real-time runs + sweep)
Run `eval/sweep.py` on the 119 s pop song (16 kHz copy) with `--seconds 60`.
Done when: results.md has the C++ table for all 9 runs. Then summarize for Sam: measured lag p50/p99,
pass time cached vs uncached, CPU vs GPU, overruns (should be 0).

### 3. Delay experiment
Add `--delays` to py/stream_sim.py (default 0). Delay D frames: each pass emits output frames
[end-S-D, end-D) instead of the newest S, so passes happen D frames later and lag grows by D*16 ms.
Run strides 0.256 s and 0.512 s with delays 0, 16, 32, 64, 128 frames. Add "delay ms" and
"est. lag ms" (= 576 + stride + delay) columns. Verify: stride 8.192 s with delay 0 still gives F1 1.0.
Then add the same `--delay N` option to the C++ engine (engine.cpp: emit rows [512-S-D, 512-D) of
the time-model output, and require D more frames before each pass) and check one setting matches
stream_sim's F1. Deliverable: an F1-vs-lag table Sam can turn into a chart.

### 4. Live note events + piano roll page
- Engine: add `--events` flag. When set, print one JSON line per event to stdout and flush:
  `{"type":"start","t_wall":0}` when the audio thread starts,
  `{"type":"on","pitch":60,"onset":12.345,"vel":64,"t_wall":13.41}` from the decoder's NoteOn callback,
  `{"type":"off","pitch":60,"onset":12.345,"offset":12.9,"t_wall":13.8}` from NoteOff.
  onset/offset are song time in seconds, t_wall is seconds since start. No other stdout output when
  --events is on (move logs to stderr).
- ui/server.py: starts pianofi_live as a subprocess, reads its stdout, broadcasts each line over a
  WebSocket (`websockets` package), and serves ui/index.html (simple aiohttp or http.server thread).
- ui/index.html: one file, no build step. Canvas piano roll: x = time (now line at 75% width,
  notes scroll left), y = pitch (piano keys on the left), bar color by velocity. Plays the same wav
  with <audio> started on the "start" event, so notes line up with what you hear. Show a live
  "lag" readout = t_wall - onset per note (median of last 20).
Done when: Sam runs one command, opens localhost, hears the song, and sees notes scroll about 1 s behind.

### 5. Live audio input
Replace the file-playing audio thread with miniaudio capture (single header from GitHub):
`--input file|mic|loopback`. Loopback = WASAPI loopback so Spotify/YouTube audio feeds the engine.
Capture at 16 kHz mono (let miniaudio resample) and push from the callback into the same ring buffer.
Callback rules: no allocation, no locks, no I/O. Note in RUN.md that miniaudio's resampler differs
slightly from torchaudio's, so F1 checks stay on file input.

### Not for Claude Code
- Resume bullets: Sam does these in the Claude chat app from results.md numbers.

## Resume-relevant numbers to collect along the way
- F1 vs batch per stride and delay (task 3)
- Measured lag p50/p99 (task 2)
- Pass time: GPU uncached vs cached, CPU vs GPU (task 2)
- Ring buffer push latency, mutex vs SPSC (bench_ring). Claude wrote the SPSC buffer at Sam's request (2026-11-06).
- Overruns = 0 over a full song in real time (task 2)
