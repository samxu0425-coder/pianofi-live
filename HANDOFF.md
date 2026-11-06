# HANDOFF: status (updated 2026-11-06)

Read this after CLAUDE.md. RUN.md has every command. results.md has every measured number.
Sam prefers short, informal answers and no em dashes. Stop after each task, show the output, and
wait for a go-ahead.

## Goal
Live version of PianoFi: any song playing → piano-cover notes scrolling on screen about 1 s behind.
C++ engine for the streaming part. It's a resume project, so every number must be measured, and
Sam has to be able to explain every line in an interview.

## Status: the local version is complete
The five original tasks are done (export and checks, real-time sweep, delay experiment, piano roll,
live input), plus the hardening that live input turned out to need.

What works, all on Sam's PC (i5-13600KF, RTX 4070, Windows 10):
- **Engine** (`cpp/`): wav file, microphone, or system audio (WASAPI loopback) in; notes out.
  Default live setting is stride 256 ms + delay 256 ms: lag 1.01 s p50 / 1.14 s p99, 0 dropped samples.
- **Exactness**: at stride 8192 ms the engine reproduces batch (F1 1.0000, array diff 2.88e-04,
  decoder identical). These checks pass on the current build.
- **Piano roll** (`ui/`): one command, wav or live. Stop ends the engine cleanly and saves the notes.
- **Automatic input level** (`cpp/include/pf/auto_gain.h`, live input only): the model finds fewer
  notes in quiet audio, so the loudest peak of the last 10 s is scaled to full scale.
- **Long sessions**: live input drops history it no longer needs and runs until stopped.
- **Device choice**: `--list-devices`, `--device NAME`.

## Not verified, or known limits
- **Microphone with music**: the mic path captures, corrects level and stops cleanly, but nobody has
  yet seen it transcribe music played into a mic. Sam's default mic (a SteelSeries Sonar virtual
  device) delivers silence; `--device CMTECK` is the real one.
- **Real Spotify or YouTube**: loopback was only tested with the test songs played from a wav.
- **Loopback hears everything** the PC plays, so other apps' sounds get transcribed too.
- **Tone changes** from the PC's sound output are not corrected. On Sam's PC the treble came through
  about 4 dB stronger than the bass. In simulation that costs about 0.02 F1 after level correction.
- **Quiet source files**: automatic level brings any audio to full scale, so a file that is itself
  quiet (the panflute clip peaks at 0.25) no longer matches batch run on that quiet file
  (0.95 → 0.85). There is no ground truth to say which is better.
- **One-off slow or large runs** have shown up three times (a 267 s pass, an 8 s pass, a 3.7 GB
  memory peak). The likely cause is competition for GPU memory, but that is not proven.
- F1 everywhere means agreement with batch output, not accuracy against a human piano cover.
- All numbers are from one PC and three songs.

## Key facts (so you don't re-derive them)
- AMT-APC = hFT-Transformer. Everything up to the SAtime stage is per-frame (each output frame
  depends only on input frames t-32..t+32), so its result `h` can be cached per frame. Only
  SAtime (3 layers over the 512-frame window) must rerun each pass.
- Lag floor ≈ 576 ms: 512 ms model lookahead + 64 ms mel window. Real lag adds about half a stride
  of waiting, any delay, and the pass time.
- Delay D frames lets each output frame see D newer frames. 256 ms of delay raised agreement from
  0.917 to 0.940 in simulation. It levels off near 0.95, so the edge-of-window effect is only part
  of the gap to batch.
- The model is sensitive to input level. Same song at x0.4 volume: 0.879 → 0.695 (rap instrumental),
  0.929 → 0.884 (pop). Automatic level gets back to 0.877 and 0.912.
- WASAPI loopback: delivers no samples while nothing is playing (the engine plays a silent stream
  to keep the clock running), and its level depends on the PC's sound setup.
- mpe2note keeps note times as float32, so the decoder check allows one float32 step past 128 s.
- ONNX Runtime on CUDA: TF32 is turned off (it moved outputs away from batch), and the memory pool
  grows only by what is asked for (3.1 GB per engine instead of 4.8 GB, same speed).
- Run one engine at a time on this GPU. Two at once crawl.

## Next: cloud deploy (not started, and not yet planned with Sam)
Measured numbers that will matter:
- One engine: about 3.1 GB of GPU memory and 52 to 66 ms per pass at stride 256 ms, so about 4 passes
  fit in one stride. Without the cache it is 6.0 GB and 163 ms.
- The capture code is Windows-only (WASAPI through miniaudio). A server would get audio from the
  browser instead, so it needs a different way in, not this capture path.
- Network delay adds to the 1.0 s lag.
- The piano roll page and the event format (`--events`, one JSON line per note) can be reused.

## Resume-relevant numbers (all in results.md)
- Lag p50/p99 and 0 overruns in real time, file and loopback
- Pass time: cached vs uncached, CPU vs GPU
- F1 vs batch per stride and delay, and the exactness check
- Ring buffer push latency, mutex vs lock-free. Claude wrote the lock-free buffer at Sam's request.
- Automatic input level: F1 with and without, at several volumes
- Long-session memory: frames held with and without dropping history
- GPU memory per engine
