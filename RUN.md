# PianoFi Live: how to run the C++ engine

What's here:

| Path | What it does |
|---|---|
| `py/export_onnx.py` | Splits AMT-APC into `frame.onnx` (per-frame, cacheable) and `time.onnx` (window), checks both against the original model |
| `py/prep_audio.py` | Converts songs to 16 kHz mono exactly like the batch pipeline |
| `py/check_mel.py` | C++ mel spectrogram vs a float64 run of the `Pipeline.wav2feature` transform |
| `cpp/` | Engine: audio thread → ring buffer → mel → frame model + h cache → time model → note decoder |
| `eval/score_engine.py` | F1 vs batch, decoder check vs `mpe2note`, appends a row to `results.md` |
| `eval/sweep.py` | Runs the grid (CPU/GPU, cache on/off, strides, A/B heads) and scores every run |
| `eval/level_eval.py` | Automatic input level: quiet, tilted and dipping copies of each song, scored against batch |
| `eval/memory_check.py` | Long sessions: dropping old history changes no notes and keeps memory bounded |
| `eval/gpu_memory.py` | GPU memory one engine run takes |
| `ui/server.py`, `ui/index.html` | Live piano roll for a wav file, the mic, or whatever the PC is playing |

All of this has run on Sam's PC with the real model weights. Every measured number, with the
command that produced it, is in results.md.

Run one engine at a time. One engine takes about 3 GB of GPU memory at the default settings
(6 GB without the cache), and two at once fight over it and crawl. Closing games and other
GPU-heavy apps first avoids the same problem.

## 1. Python packages (in the .venv)

```powershell
pip install onnx onnxruntime soundfile websockets
```

## 2. Prepare audio and export the model

```powershell
.venv\Scripts\python.exe py\prep_audio.py test_audio\*.wav
.venv\Scripts\python.exe py\export_onnx.py --wav test_audio\16k\<song>.wav
```

`export_onnx.py` must print `All checks passed`. It also prints a first CPU timing for each graph.

## 3. Build the C++ code

Needs Visual Studio Build Tools 2022 ("Desktop development with C++", includes CMake).

1. Download `onnxruntime-win-x64-gpu-1.20.1.zip` from
   https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1 and unzip it, e.g. to `C:\libs\`.
2. For the GPU: ONNX Runtime 1.20 needs CUDA 12.x and cuDNN 9.x DLLs on PATH. CUDA torch already
   ships them in its `lib` folder. `eval\sweep.py` adds that folder to PATH itself. For direct
   engine runs, set it first (the path below is this PC's). Without the DLLs the engine prints
   "CUDA unavailable" and uses the CPU.

```powershell
$env:PATH = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Lib\site-packages\torch\lib;$env:PATH"
```

3. Open "Developer PowerShell for VS 2022" in `pianofi-live` and run:

```powershell
cmake -S cpp -B cpp\build -DONNXRUNTIME_DIR=C:\libs\onnxruntime-win-x64-gpu-1.20.1
cmake --build cpp\build --config Release
cpp\build\Release\pf_tests.exe
cpp\build\Release\bench_ring.exe
```

## 4. Check the pieces against Python

```powershell
.venv\Scripts\python.exe py\check_mel.py test_audio\16k\<song>.wav
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 512 --fast --dump --out runs\check512
.venv\Scripts\python.exe eval\score_engine.py runs\check512 test_audio\16k\<song>.wav
```

Expect: mel PASSED, array check vs batch PASSED, decoder check PASSED, F1 = 1.0000.
Stride 512 runs the same windows as batch mode, so anything else means a bug.

## 5. Real-time runs and the sweep

```powershell
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 16
.venv\Scripts\python.exe eval\sweep.py test_audio\16k\<song>.wav --seconds 60
```

Each sweep run plays the clip in real time. Results land in `results.md`; raw numbers in `runs\*\metrics.json`.

The engine turns off NVIDIA's reduced-precision TF32 math on CUDA, because it moves outputs away
from batch and fails the array check. Python scripts that use torch are unaffected.

What the numbers mean:
- **pass ms**: compute per pass. Must stay under the stride (e.g. 256 ms at `--stride 16`) or the engine falls behind.
- **frame lag**: audio arriving → its output ready. Floor is about 576 ms with the B heads
  (512 ms model lookahead + 64 ms mel window) plus up to one stride of waiting. The A heads have
  the same lookahead but skip the time model.
- **RTF**: compute / audio time. Under 1 keeps up.
- **overruns**: samples dropped because the ring buffer was full. Should be 0.

### Delay experiment

Delay D frames emits each output frame D frames later, so it has newer frames in its window. Lag grows by D*16 ms.

```powershell
.venv\Scripts\python.exe py\stream_sim.py --strides 0.256 0.512 --delays 0 16 32 64 128
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 16 --delay 32 --fast --dump --out runs\delay-s16-d32
.venv\Scripts\python.exe eval\score_engine.py runs\delay-s16-d32 test_audio\16k\<song>.wav
```

The engine's F1 should match the stream_sim row for the same song, stride and delay.

### Live piano roll

```powershell
.venv\Scripts\python.exe ui\server.py test_audio\16k\<song>.wav
```

Open http://localhost:8000 and press Play. You hear the song and see its notes scroll about 1 s behind.
The default is `--stride 16 --delay 16`. The button is there because browsers only allow sound after a click.
Engine logs stay on this console. Ctrl+C quits.

For whatever the PC is playing (Spotify, YouTube), or the mic:

```powershell
.venv\Scripts\python.exe ui\server.py --input loopback
.venv\Scripts\python.exe ui\server.py --input mic
```

Press Play, then play your music. The page doesn't play anything itself here, and the button
turns into Stop. Stop lets the engine finish its last notes and write `runs\ui\notes.csv`.

### Live input (mic or system audio)

```powershell
cpp\build\Release\pianofi_live.exe --input loopback --stride 16 --delay 16
cpp\build\Release\pianofi_live.exe --input mic --stride 16 --delay 16
```

`loopback` captures whatever the PC is playing (Spotify, YouTube) through WASAPI loopback, so no
virtual cable is needed. `mic` uses a microphone. It runs until Ctrl+C (or `--max-seconds X`),
then finishes its last notes and writes notes.csv and metrics.json.
Set the CUDA PATH line from step 3 first: on the CPU the engine can't keep up.

Both use the Windows default device unless you name one. The default microphone is not always the
one that works (on Sam's PC it is a virtual device that delivers silence):

```powershell
cpp\build\Release\pianofi_live.exe --list-devices
cpp\build\Release\pianofi_live.exe --input mic --device CMTECK --stride 16 --delay 16
```

`--device` takes any part of a name from the list. `ui\server.py` accepts `--device` too.

What to know about live audio:

- **Loopback hears everything.** Other apps' sounds (a call, a video, a notification) are mixed in
  and get transcribed too. For clean results, let only the music play.
- **Volume is corrected automatically.** Live audio arrives at whatever level the PC's sound setup
  gives it (on Sam's PC, loopback came in anywhere from 0.03x to 0.85x of the file's level), and the
  model finds fewer notes in quiet audio. So for mic and loopback the engine scales the audio until
  the loudest peak of the last 10 s is at full scale (`cpp/include/pf/auto_gain.h`).
  `--no-normalize` turns it off. `--normalize` turns it on for a wav file, to test it.
- **Silence stays on the clock.** Windows sends no loopback samples while nothing is playing, so
  the engine plays its own silent stream to keep them coming.
- **It can't undo tone changes.** Sound effects or an equalizer on the output device change what
  loopback captures. Turn them off for the closest match to the file.
- **Memory stays bounded.** Live input drops mel and decoder frames once nothing needs them.
  `--bounded-memory` does the same for a wav file, to test it.
- Needs the lock-free ring buffer (the default build), because the audio callback must not lock.
- Live audio is converted to 16 kHz mono by Windows and miniaudio, not by torchaudio like
  `py\prep_audio.py`, so the samples differ slightly. F1 checks stay on file input.
- `cpp\build\Release\capture_dump.exe loopback|mic SECONDS RATE OUT.wav` records what the capture
  path delivers, to check it against a file (RATE 0 = the device's own rate).
- `cpp\third_party\miniaudio.h` is miniaudio v0.11.26 (public domain or MIT-0).

## 6. The lock-free ring buffer

`cpp/include/pf/ring_buffer.h` has `MutexRingBuffer` and `SpscRingBuffer` (code in
`cpp/src/spsc_ring_buffer.cpp`). The engine uses the lock-free one by default. Add
`-DPF_USE_SPSC=OFF` to the cmake configure line to go back to the mutex.

```powershell
cpp\build\Release\pf_tests.exe
cpp\build\Release\bench_ring.exe
```

`bench_ring` prints throughput and push latency for both. Results are in results.md.

## 7. Checks for live input

These use wav files, so they are repeatable. Each appends its table to results.md.

```powershell
.venv\Scripts\python.exe eval\level_eval.py
.venv\Scripts\python.exe eval\memory_check.py
.venv\Scripts\python.exe eval\gpu_memory.py test_audio\16k\<song>.wav
```

- `level_eval.py`: about 20 minutes. Shows what quiet or tilted audio costs, and what the automatic
  input level gets back.
- `memory_check.py`: about 20 minutes. Exit code 1 if dropping old history changed any note.
- `gpu_memory.py`: about 5 minutes. Close other GPU apps first.

## 8. Linux build in Docker (written, NOT yet run)

The `Dockerfile` builds the engine on Linux with CUDA 12, cuDNN 9 and ONNX Runtime 1.20.1, and
runs `pf_tests` as part of the build. Nobody has built or run it yet: this PC has no Docker, and
virtualization is off in its BIOS. Until it has run, treat every step here as untested.

Needs: virtualization on in the BIOS, Docker Desktop (WSL 2 backend), and a current NVIDIA driver.

```powershell
docker build -t pianofi-live .
docker run --rm --gpus all -v ${PWD}\models:/work/models -v ${PWD}\test_audio:/work/test_audio -v ${PWD}\runs:/work/runs pianofi-live --wav test_audio/16k/<song>.wav --stride 512 --fast --dump --out runs/docker_check512
.venv\Scripts\python.exe eval\score_engine.py runs\docker_check512 test_audio\16k\<song>.wav
```

- The first command fails if `pf_tests` fails on Linux.
- The second runs the engine in the container. The first line it prints must say `provider CUDA`.
  "CUDA unavailable" means the container can't see the GPU.
- The third scores that run on Windows, like step 4. Expect the same: array check PASSED, decoder
  check PASSED, F1 = 1.0000.
- `score_engine.py` writes this PC's name in the machine column. For a run made on another machine
  (a cloud GPU), fix that column by hand, or the row is wrong.
- Mic and loopback input do not work in the container: a server has no sound card.

## Next steps (not built yet)

- GPU-side h cache (IOBinding) so the 512-frame window isn't copied to the GPU every pass.
