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

Already verified in a Linux sandbox (not yet on your PC): C++ unit tests pass, the mel matches
torch.stft + an HTK/slaney filterbank to 2.5e-5, the engine's cached/uncached/any-stride outputs
match a Python stream simulation exactly (with stand-in models of the same shape), and the note
decoder matches a transcription of `mpe2note` on 40 random fuzz cases. Not verified: anything with
the real model weights. Steps 2, 4 and 5 below are where that gets checked.

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

## 6. The lock-free ring buffer

`cpp/include/pf/ring_buffer.h` has `MutexRingBuffer` and `SpscRingBuffer` (code in
`cpp/src/spsc_ring_buffer.cpp`). The engine uses the lock-free one by default. Add
`-DPF_USE_SPSC=OFF` to the cmake configure line to go back to the mutex.

```powershell
cpp\build\Release\pf_tests.exe
cpp\build\Release\bench_ring.exe
```

`bench_ring` prints throughput and push latency for both. Results are in results.md.

## Next steps (not built yet)

- Live input: miniaudio capture (mic, or WASAPI loopback for system audio) feeding the same ring buffer.
- GPU-side h cache (IOBinding) so the 512-frame window isn't copied to the GPU every pass.
