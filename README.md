# PianoFi Live

Play any song and watch its piano cover appear as notes on a scrolling piano roll, about one second behind the music. A C++ engine streams audio through a piano-cover transcription model in real time, from a file, a microphone, or whatever your PC is playing.

**Live:** https://54-174-0-35.sslip.io
&nbsp;·&nbsp; log in as `guest` with password `NJNAvBYK5pBKXd6T`
&nbsp;·&nbsp; [Report a bug](https://github.com/samxu0425-coder/pianofi-live/issues)

If the page says the engine is offline, press **Start the engine** and the GPU machine boots in about 20 s. The guest login gets three starts a day and cannot add songs.

## Contents

- [About the project](#about-the-project)
  - [How it works](#how-it-works)
  - [Numbers](#numbers)
  - [Built with](#built-with)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Running the tests](#running-the-tests)
- [Deploying](#deploying)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About the project

[AMT-APC](https://github.com/misya11p/amt-apc) is a model that turns a song into a piano cover, but it works on whole files: you give it a song, wait, and get a MIDI file back. PianoFi Live runs the same model on audio as it arrives, so the cover shows up while the song is still playing.

The catch is that the model wants to see 0.5 s of future audio for every frame it transcribes, and each pass looks at an 8 s window. Running it the batch way on a live stream would mean redoing the whole window every few hundred milliseconds. The engine splits the model in two so that the per-frame half runs once per frame and is cached, and only the lighter time half is rerun over the window. That brings one pass down to about 40 ms on an RTX 4070 (16 ms with TensorRT in half precision) and keeps the GPU memory at 3 GB instead of 6.

Lag is about 1 s end to end: 0.5 s is the model's lookahead and cannot be removed, and the rest is the window stride plus a deliberate delay that buys back accuracy (the engine gives each frame a little more future context before committing to it).

### How it works

```
audio thread (10 ms blocks)  ->  lock-free ring buffer  ->  mel spectrogram
    -> frame.onnx on new frames only (cached)  ->  time.onnx over the last 512 frames
    -> streaming note decoder  ->  notes.csv / WebSocket  ->  piano roll in the browser
```

- The audio callback never allocates, locks, or logs. It hands samples through a single-producer single-consumer ring buffer; dropped samples are counted as overruns and every reported run has zero.
- At stride 512 the engine runs exactly the windows the batch model runs, so its output must match batch note for note. That check (`F1 = 1.0000`, identical arrays) is how every change is validated.
- Live input gets an automatic input level (quiet audio loses notes) and bounded memory (old frames are dropped once nothing needs them, with a test that proves it changes no notes).
- `ui/server.py` runs one engine per open browser tab and streams the notes over a WebSocket. The browser decodes and resamples audio itself (Web Audio API), so the server only ever sees 16 kHz mono float32.

### Numbers

Everything below is from `results.md`, which has the exact command for each row. Test clip: a 60 s pop song, scored against the batch model's notes with mir_eval (50 ms onset tolerance, offsets ignored).

| setting | F1 vs batch | pass p50 | lag p50 / p99 | machine |
|---|---|---|---|---|
| stride 512 frames (batch windows) | 1.0000 | 170 ms (4070), 443 ms (T4) | n/a | RTX 4070 and a T4: identical output |
| stride 256 ms + delay 256 ms (the default) | 0.9233 | 41 ms | 992 / 1114 ms | RTX 4070 |
| same, half precision (FP16) | 0.9221 | 27 ms | 979 / 1100 ms | RTX 4070 |
| same, FP16 on TensorRT | 0.9228 | 16 ms | 967 / 1089 ms | RTX 4070 |
| the default, on the cloud machine | 0.9233 | 97 ms | 1047 / 1167 ms | AWS g4dn.xlarge spot (Tesla T4) |
| same, half precision (FP16) | 0.9227 | 51 ms | 1001 / 1121 ms | AWS g4dn.xlarge spot (Tesla T4) |
| same, FP16 on TensorRT | 0.9222 | 30 ms | 980 / 1100 ms | AWS g4dn.xlarge spot (Tesla T4) |
| two engines sharing the T4 | 0.9233 | 98 and 99 ms | 1049 / 1170 ms | AWS g4dn.xlarge spot (Tesla T4) |
| four engines sharing the T4, FP16 | 0.9227 | 96 to 135 ms | 1091 / 1214 ms (slowest engine) | AWS g4dn.xlarge spot (Tesla T4) |
| five engines sharing the T4, FP16 on TensorRT (what the site runs) | 0.9222 | 30 to 64 ms | 1014 / 1137 ms (slowest engine) | AWS g4dn.xlarge spot (Tesla T4) |
| stride 128 ms, no delay (measured before the two speedups below) | 0.9009 | 43 ms | 675 / 740 ms | RTX 4070 |

- A per-node profile showed ONNX Runtime running one small node (the velocity ArgMax) on the CPU, where it stalled on the thread pool. Exporting it as TopK moved it to the GPU: the pass on the 4070 went from 63 to 41 ms with identical notes.
- Half precision cuts the pass again (41 to 27 ms on the 4070, 97 to 51 ms on the T4) and GPU memory by 20%, for 0.001 F1. Two attention layers overflow in FP16 and stay in FP32. In FP32 a T4 keeps up with two engines and falls behind with three; in FP16 it keeps up with four.
- TensorRT (NVIDIA's inference compiler, used through ONNX Runtime) cuts the half-precision pass again: 27 to 16 ms on the 4070 and 51 to 30 ms on the T4, and in FP32 it still matches batch exactly. Left to pick its own FP16 layers it loses accuracy (F1 0.983 at stride 512), so it gets the same mixed-precision graphs and scores 0.998. On the T4 the limit is now GPU memory: five engines fit, a sixth does not.
- Through the deployed site in us-east-1, from my PC, the end-to-end lag (sample leaves the browser to its note arriving back) measured 1.08 to 1.10 s median, so the network adds under 0.1 s.
- The lock-free ring buffer pushes a block in 300 ns at p99 against 2800 ns for the mutex version, with a worst case 7 to 11 times lower.
- On the CPU the model does not keep up (real-time factor above 1 at every stride), so a GPU is required.

### Built with

- C++17, CMake, and [ONNX Runtime](https://onnxruntime.ai/) 1.20 with its CUDA and [TensorRT](https://developer.nvidia.com/tensorrt) providers for the engine
- [miniaudio](https://miniaud.io/) for microphone and WASAPI loopback capture (no virtual cable needed)
- [AMT-APC](https://github.com/misya11p/amt-apc) (a fine-tuned hFT-Transformer) for the piano covers, exported to ONNX with PyTorch
- Python 3.11+ for the export, checks and scoring: [mir_eval](https://github.com/mir-evaluation/mir_eval), soundfile, [websockets](https://websockets.readthedocs.io/)
- A plain HTML and JavaScript piano roll (Canvas and the Web Audio API, no framework)
- Docker (CUDA 12 image) for the Linux build, [Terraform](https://www.terraform.io/) and AWS (spot g4dn.xlarge plus a t4g.nano front) for the deployment, [Caddy](https://caddyserver.com/) for HTTPS and logins
- SQLite for the results database behind `results.md`

## Getting started

The engine was built and measured on Windows 10 with an RTX 4070. Linux works through the `Dockerfile` (that is what runs in the cloud). `RUN.md` walks through every step in order with the expected output of each.

### Prerequisites

- An NVIDIA GPU with a current driver (CUDA 12.x and cuDNN 9.x DLLs on PATH; a CUDA build of PyTorch ships them)
- Visual Studio Build Tools 2022 with "Desktop development with C++" (includes CMake)
- Python 3.11 or newer, with PyTorch (for the ONNX export and scoring)
- [ONNX Runtime GPU 1.20.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1) for Windows x64, unzipped somewhere

### Installation

1. Clone the repo and the model next to it:
   ```sh
   git clone https://github.com/samxu0425-coder/pianofi-live.git
   cd pianofi-live
   git clone https://github.com/misya11p/amt-apc.git
   ```
   Download the pre-trained weights to `amt-apc/models/params/apc.pth` as the AMT-APC README describes.
2. Python packages:
   ```sh
   pip install onnx onnxruntime soundfile websockets mir_eval
   ```
3. Convert a song to 16 kHz mono and export the model to ONNX (also checks both halves against the original):
   ```sh
   python py/prep_audio.py test_audio/your_song.wav
   python py/export_onnx.py --wav test_audio/16k/your_song.wav
   ```
   It must end with `All checks passed`.
4. Build the engine (from a "Developer PowerShell for VS 2022"):
   ```powershell
   cmake -S cpp -B cpp\build -DONNXRUNTIME_DIR=C:\libs\onnxruntime-win-x64-gpu-1.20.1
   cmake --build cpp\build --config Release
   cpp\build\Release\pf_tests.exe
   ```

## Usage

**The piano roll:**

```sh
python ui/server.py
```

Open http://localhost:8000, pick a song and press Play. You hear the song and see its notes scroll in about a second behind. Drop any audio file on the page to add it to the song list. Stop ends the engine cleanly and keeps the notes; Play again resumes where it stopped. Scroll or drag the roll to look back at earlier notes, "Replay" plays the finished roll again, and "Download MIDI" saves it as a standard MIDI file.

For whatever the PC is playing (Spotify, YouTube) or the microphone:

```sh
python ui/server.py --input loopback
python ui/server.py --input mic
```

**The engine on its own** writes `notes.csv` and `metrics.json` (timings, lag, overruns) to the `--out` folder:

```sh
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\your_song.wav --stride 16 --delay 16
cpp\build\Release\pianofi_live.exe --input loopback --stride 16 --delay 16
cpp\build\Release\pianofi_live.exe --list-devices
```

`--stride` is in frames of 16 ms (16 = a pass every 256 ms) and `--delay` is extra lookahead in frames. Two switches trade exactness for speed: `--fp16` runs the half-precision graphs (`python py/export_fp16.py` writes them), and `--trt --mixed` runs those graphs on TensorRT, the fastest path (`RUN.md` has the TensorRT setup). `--trt` alone is TensorRT in FP32, which still matches batch exactly. The engine refuses to start while another engine is running on the machine, because two on one GPU fight over memory and both fall behind; `--max-engines N` allows more.

## Running the tests

Three checks have to pass after any change, in this order:

```sh
cpp\build\Release\pf_tests.exe
python py/check_mel.py test_audio/16k/your_song.wav
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\your_song.wav --stride 512 --fast --dump --out runs\check512
python eval/score_engine.py runs/check512 test_audio/16k/your_song.wav
```

- `pf_tests` covers both ring buffers (including two-thread runs), the FFT and mel pipeline, the automatic input level, and the streaming note decoder.
- `check_mel.py` compares the C++ mel spectrogram with a float64 run of the model's own transform.
- At stride 512 the engine runs the same windows as batch mode, so `score_engine.py` must report `array check PASSED`, `decoder check PASSED` and `F1 = 1.0000`. Anything else is a bug, not a tolerance to loosen.

Longer evaluations live in `eval/`: `sweep.py` runs the settings grid, `level_eval.py` measures what quiet or tilted audio costs and what the automatic level gets back, `memory_check.py` proves a long session stays bounded without changing a note, and `gpu_memory.py` measures GPU memory per engine. Each appends its table to `results.md`, and `results_db.py` keeps the same numbers in a SQLite database for queries across machines.

## Deploying

The site is two AWS machines, both created by `infra/deploy/` with Terraform:

- A **GPU box**, a persistent spot `g4dn.xlarge` (one T4), running the service image from the `Dockerfile`: the engine plus `ui/server.py`, five engines at once on TensorRT in half precision. It shuts itself down after 30 minutes with no engine running, and a stopped box costs only its disk.
- A **front**, a `t4g.nano` with a fixed address, running Caddy: HTTPS with a Let's Encrypt certificate, two logins, and a proxy to the GPU box. When the box is stopped, the front serves an offline page with a **Start the engine** button instead of going dark. The admin login can add songs and start the box as often as needed; the guest login can play and transcribe and gets three starts a day. The button calls `aws ec2 start-instances` through an IAM role that allows exactly that one instance.

```sh
terraform -chdir=infra/deploy apply
```

Then copy the code, model and songs to the GPU box and run `deploy/run.sh` there; `RUN.md` step 11 has the exact commands, the password setup, and the things that bit me (which zones have spot capacity, why the front's files are shipped by hand, how to tell nothing is left running). Finish with `terraform -chdir=infra/deploy destroy` and check that no instance, address or spot request remains.

## Roadmap

- [x] Half precision (FP16) with the two overflowing attention layers kept in FP32
- [x] TensorRT through ONNX Runtime, deployed: five users per T4, up from two
- [ ] Less GPU memory per engine (2.5 GB now), which is what caps a T4 at five users
- [ ] GPU-side cache (IOBinding) so the 512-frame window is not copied to the GPU every pass
- [ ] A real domain instead of the sslip.io address

## Contact

Sam - [@samxu0425-coder](https://github.com/samxu0425-coder)

Project: https://github.com/samxu0425-coder/pianofi-live

## Acknowledgments

- [AMT-APC](https://github.com/misya11p/amt-apc) by komiya, the model this runs (MIT), and the [paper](https://arxiv.org/abs/2409.14086) behind it
- Sony's hFT-Transformer, which AMT-APC fine-tunes
- [miniaudio](https://miniaud.io/) and [ONNX Runtime](https://onnxruntime.ai/)
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template), which this README is loosely based on
