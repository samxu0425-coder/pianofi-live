# PianoFi Live: how to run the C++ engine

What's here:

| Path | What it does |
|---|---|
| `py/export_onnx.py` | Splits AMT-APC into `frame.onnx` (per-frame, cacheable) and `time.onnx` (window), checks both against the original model |
| `py/prep_audio.py` | Converts songs to 16 kHz mono exactly like the batch pipeline |
| `py/export_fp16.py` | Half-precision copies of both graphs for `--fp16` (attention scores kept in FP32) |
| `py/attention_range.py` | Largest attention score per layer, which decides what stays FP32 in the FP16 graphs |
| `py/check_mel.py` | C++ mel spectrogram vs a float64 run of the `Pipeline.wav2feature` transform |
| `cpp/` | Engine: audio thread → ring buffer → mel → frame model + h cache → time model → note decoder |
| `eval/score_engine.py` | F1 vs batch, decoder check vs `mpe2note`, appends a row to `results.md` and stores the run in `runs/results.db` |
| `eval/results_db.py`, `eval/sql/` | SQLite results store: schema, import of every run, and the SQL reports behind the summary tables |
| `eval/ort_profile.py` | Reads the per-node profiles of a `--profile` run: where a pass spends its time |
| `eval/sweep.py` | Runs the grid (CPU/GPU, cache on/off, strides, A/B heads) and scores every run |
| `eval/level_eval.py` | Automatic input level: quiet, tilted and dipping copies of each song, scored against batch |
| `eval/memory_check.py` | Long sessions: dropping old history changes no notes and keeps memory bounded |
| `eval/gpu_memory.py` | GPU memory one engine run takes |
| `ui/server.py`, `ui/index.html` | Live piano roll for a wav file, the mic, or whatever the PC is playing |

All of this has run on Sam's PC with the real model weights. Every measured number, with the
command that produced it, is in results.md.

Run one engine at a time. One engine takes about 3 GB of GPU memory at the default settings
(6 GB without the cache), and two at once fight over it and crawl. Closing games and other
GPU-heavy apps first avoids the same problem. The engine enforces this itself: it takes a lock
slot in the temp folder (`cpp/include/pf/engine_lock.h`) and refuses to start while another
engine holds it, with exit code 1 and a message. `--max-engines N` allows N at once (the
deliberate two-engine test), `--max-engines 0` removes the limit. Engines in separate Docker
containers do not see each other's slots. The lock cannot see other GPU apps; `nvidia-smi` can.

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
The velocity pick is exported as TopK, not ArgMax: ONNX Runtime's CUDA provider runs ArgMax on the CPU, which cost
5 to 19 ms a pass (results.md, "Profile of a pass" and "Velocity ArgMax moved to the GPU"). `--argmax --out-dir models\argmax`
writes the old form to compare against (`pianofi_live --models models\argmax`).

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
.venv\Scripts\python.exe ui\server.py
```

Open http://localhost:8000, pick a song and press Play. You hear the song and see its notes scroll about
1 s behind. The default is `--stride 16 --delay 16`. The button is there because browsers only allow
sound after a click. Engine logs stay on this console. Ctrl+C quits.

Every open page is its own session: its own engine, song, notes and run folder
(`runs\ui\session-N`). `--max-sessions` (default 2, what one T4 keeps up with in FP32; 4 with `--fp16`, which runs the
engines on the half-precision graphs; 5 with `--trt`, TensorRT on those graphs) caps the engines
running at once; a page past the cap sees "All 2 engine slots are in use". Closing a page stops its
engine. This is the shape the cloud version uses: one engine per user, N per GPU.

The song list is every `.wav` in `test_audio\16k` (`--songs DIR` for another folder; a wav on the
command line is selected first). To add a song, drop any audio file the browser can play (wav, mp3,
m4a, ogg, flac) on the page or click the drop box: the browser decodes it, mixes it to mono and
resamples it to 16 kHz with the Web Audio API, then uploads it to the songs folder. The server only
accepts 16 kHz mono wavs. The browser's resampler is not `py/prep_audio.py` (torchaudio), so for
scored results convert the original with `prep_audio.py` instead; on the same 16 kHz file both agree
to about 1e-3. Songs uploaded this way are what the cloud version will need anyway: audio handled in
the browser, server storage behind it. With `--admin NAME` only that login (the `X-User` header the
front's Caddy sets, step 11) gets the drop box and may upload; on your own PC leave it off.

Looking back: every note of the session is kept, so scroll the wheel or drag the roll sideways to see
earlier ones (arrow keys move 1 s). Scrolling anchors the view on that spot, like a DAW: it stays put
while the music's playhead (a blue line) travels across it, and Home or "Back to live" makes the view
follow the music again (for a song, that skips it forward to where the engine is). There is no pause:
Stop and Play do that for real (below), and scrolling covers looking at earlier notes. The time axis
along the bottom is song time, and the header shows the view's position, how far behind the engine it
is, and the note count.

Play turns into Stop while the engine runs, for songs too: the engine stops reading, finishes the
notes it has (no padding to the next 512 frames) and writes them, so a stopped run keeps its notes.
Play again continues: the same song resumes from where it stopped (a new engine starts there with
`--start-seconds`, so its first second has no history and may miss a note), the mic or a tab
continues at the end of the previous capture, and the roll keeps everything. A different song or
source, or "Clear", starts a fresh roll.
After the engine finishes (or is stopped), "Replay" plays the notes on the page again from the start
with the song; the engine is not run again, since the same audio and settings give the same notes.
The button turns into "Stop replay" while it runs, and Play is disabled until the replay ends.
With live input there is no recording, so the roll replays silently. "Download MIDI" saves the notes
on the page as a standard MIDI file (format 0, one piano track, 120 bpm, same note times and
velocities as the engine's notes.csv), at any point from the first note on.

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

### FP16 (half precision)

```powershell
pip install onnxconverter-common
.venv\Scripts\python.exe py\export_fp16.py
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 16 --delay 16 --fp16
```

`export_fp16.py` makes `frame_fp16.onnx` and `time_fp16.onnx` from the FP32 graphs (weights and math in 16-bit
floats, inputs and outputs still 32-bit), and `--fp16` loads them. GPU only. Two attention layers keep their
scores in FP32: in FP16 they overflow and every output becomes NaN. `py\attention_range.py runs\<a --dump run>`
prints the largest score per layer, which is where that list of two comes from; run it again if the model or
the style changes. It is not exact, so with `--fp16` the stride 512 array check fails by design and F1 is
slightly under 1; the step 4 checks stay on FP32. Numbers: results.md, "FP16 on the CUDA provider".

### TensorRT

`--trt` puts ONNX Runtime's TensorRT provider in front of the CUDA provider. It needs the TensorRT
10.4 libraries on PATH, which ship as a pip package (libraries only, 1 GB; the Dockerfile installs
the Linux one):

```powershell
.venv\Scripts\python.exe -m pip download tensorrt-cu12-libs==10.4.0 --no-deps --extra-index-url https://pypi.nvidia.com -d C:\libs\tensorrt-dl
mkdir C:\libs\tensorrt-10.4
tar -xf C:\libs\tensorrt-dl\tensorrt_cu12_libs-10.4.0-py2.py3-none-win_amd64.whl -C C:\libs\tensorrt-10.4
$env:PATH = "C:\libs\tensorrt-10.4\tensorrt_libs;" + $env:PATH
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 512 --fast --trt --dump --out runs\trt_check512
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>_first60s.wav --stride 16 --delay 16 --trt --mixed --dump --out runs\trt_mixed_s16_d16_rt
```

- The `.whl` is a zip file; Windows' own `tar` unpacks it.
- `--trt` alone runs the FP32 graphs and must still pass the stride 512 check exactly (array check
  PASSED, F1 = 1.0000). `--trt --mixed` runs the mixed-precision graphs of `py\export_fp16.py`: the
  fast path, not exact, like `--fp16`.
- `--trt --fp16` lets TensorRT choose FP16 layers itself. It is less accurate (F1 0.9830 at stride
  512 against 0.9977), because it does not keep the two overflowing attention layers in FP32.
- The first run of each variant builds the engines (10 to 30 s on the RTX 4070, about 40 s on the
  T4) and keeps them in `models\trt_cache`. They belong to one GPU model; delete the folder after
  exporting the model again.
- If the TensorRT DLLs are not found the engine says `CUDA unavailable (... onnxruntime_providers_tensorrt.dll ...)`
  and runs on the CPU.

### Where a pass spends its time

```powershell
cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song>.wav --stride 16 --delay 16 --profile --out runs\prof
.venv\Scripts\python.exe eval\ort_profile.py runs\prof
.venv\Scripts\python.exe eval\ort_profile.py runs\prof --graph time
```

`--profile` turns on ONNX Runtime's own per-node profile for both graphs (two JSON files in `--out`, over 100 MB
for a minute of audio, so delete them afterwards). `ort_profile.py` prints the median time of one run of the graph,
split into CUDA nodes, CPU nodes and copies, and the slowest nodes. Profiling slows the pass, so use it to see
where time goes and measure the effect of a change without it. Times are what the CPU thread waited at each node:
GPU work can show up at a later node than the one that caused it. What it found is in results.md, "Profile of a pass".

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

## 8. Linux build in Docker

The `Dockerfile` builds the engine on Linux with CUDA 12, cuDNN 9 and ONNX Runtime 1.20.1, and
runs `pf_tests` as part of the build. Built and run on this PC on 2026-11-06 (Docker Desktop 29.9.2):
the stride 512 check passes with F1 = 1.0000, and the rows are in results.md.

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
- `score_engine.py` writes this PC's name in the machine column and the command as the container
  saw it (`/app/pianofi_live ...`). Fix both by hand in the new row: add that it ran in Docker (or on
  which cloud machine), and put the `docker run ...` part in front of the command.
- Mic and loopback input do not work in the container: a server has no sound card.

## 9. A spot GPU machine on AWS for tests

`infra/` is a Terraform script for one spot GPU machine (default `g4dn.xlarge`, one T4) in
`us-east-1`. It is a test box: it does not touch the PianoFi cluster. Applied once, on 2026-11-06: the machine came up, the image built there and the runs are in
results.md. AWS took the machine back after 13 minutes, before the run folders were copied off, so
those runs have timings but no F1.
Do step 8 on this PC first, so the paid time is not spent on build errors.

What it creates: the machine, a firewall rule that allows SSH from one address only, and an SSH key
entry. The machine deletes itself `max_minutes` (default 120) after it starts, as a safety net.

Needs: Terraform, the AWS CLI, and a login to the AWS account (Sam sets these up himself).

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\pianofi_live
Copy-Item infra\terraform.tfvars.example infra\terraform.tfvars
terraform -chdir=infra init
terraform -chdir=infra plan
terraform -chdir=infra apply
```

Put your public address in `infra\terraform.tfvars` before `init`. Read the `plan` output before
`apply`: it must say 3 to add, 0 to change, 0 to destroy. `apply` prints the machine's address.

Then copy the code, the model and one song up, build, run, and bring the result back
(`<ip>` is the `public_ip` that `apply` printed):

```powershell
tar -czf runs\pf.tgz --exclude cpp/build Dockerfile .dockerignore cpp models
ssh -i $HOME\.ssh\pianofi_live ec2-user@<ip> "mkdir -p pf/test_audio/16k pf/runs"
scp -i $HOME\.ssh\pianofi_live runs\pf.tgz ec2-user@<ip>:pf/
scp -i $HOME\.ssh\pianofi_live test_audio\16k\<song>.wav ec2-user@<ip>:pf/test_audio/16k/
ssh -i $HOME\.ssh\pianofi_live ec2-user@<ip>
```

The `tar` line leaves out `cpp/build`, which is over 1 GB of Windows files. On the machine:

```bash
cd pf && tar -xzf pf.tgz && docker build -t pianofi-live .
docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/<song>.wav --stride 16 --delay 16 --out runs/aws_s16
```

Back on this PC, fetch the run, score it, then delete the machine:

```powershell
scp -i $HOME\.ssh\pianofi_live -r ec2-user@<ip>:pf/runs/aws_s16 runs\
.venv\Scripts\python.exe eval\score_engine.py runs\aws_s16 test_audio\16k\<song>.wav
terraform -chdir=infra destroy
```

- Spot machines can be taken back at any moment with 2 minutes' notice. Copy each run folder back
  right after its run, not at the end.
- Always finish with `destroy`, and check in the AWS console that no instance or volume is left.
  `destroy` can take about 7 minutes.
- `score_engine.py` writes this PC's name in the machine column. Change it by hand to the AWS
  machine type for these rows.
- If `apply` fails with a spot limit error (`MaxSpotInstanceCountExceeded`), the account's limit
  "All G and VT Spot Instance Requests" is too low: one `g4dn.xlarge` needs 4 vCPUs of it. The
  account owner said to request an increase in Service Quotas.
- If it fails with no spot capacity, set `availability_zone` to another zone and apply again.
  Terraform does not give up on its own: it keeps retrying the spot request silently, and the
  `timeouts { create }` block does not stop that either, so kill it after 5 minutes if nothing was
  created, remove `infra/.terraform.tfstate.lock.info`, and run `destroy` to drop the group and key.
  On 2026-11-07 (00:00 to 03:00 UTC) us-east-1b never had a `g4dn.xlarge`; us-east-1a, 1c, 1d
  and 1f gave one within 15 seconds, so try those first.

## 10. The results database

Every run folder has a `metrics.json` (timings, settings, counts) and every scored run has a row in
results.md. `runs/results.db` is the same data in SQLite (standard library `sqlite3`, no extra
packages), so questions like "same setting, every machine" are one query instead of reading tables
by hand. Schema: `eval/sql/schema.sql`, three tables (`runs`, `timings`, `scores`) and the view
`run_summary` that joins them with the latest score per run.

```powershell
.venv\Scripts\python.exe eval\results_db.py import
.venv\Scripts\python.exe eval\results_db.py report
.venv\Scripts\python.exe eval\results_db.py sql "SELECT gpu, stride_ms, delay_ms, f1, pass_p50_ms, lag_p99_ms FROM run_summary WHERE f1 IS NOT NULL ORDER BY f1 DESC"
```

- `import` reads every `runs/*/metrics.json` (28 runs on 2026-11-07), the peak GPU memory from
  `runs/<name>.gpu_mb.txt` where step 9 wrote one, and the P, R, F1 and machine of every row in
  results.md's engine table (matched by the `--out` folder in the command; 23 rows). It can be run
  again at any time: runs are updated in place, results.md scores are replaced, scores written by
  `score_engine.py` are kept.
- `score_engine.py` stores each new score itself (with batch note count and the array check result,
  which results.md does not hold). The machine column is this PC's, so for runs made in Docker or on
  AWS, fix it once: `results_db.py sql "UPDATE runs SET machine = '...' WHERE name = '...'"`.
- `import` (and `score_engine.py`) also applies `schema.sql`, so after editing the view, run `import` once.
  `report` and `sql` only read, and leave the committed file unchanged.
- `report` runs every `eval/sql/*.sql` except the schema and prints each as a markdown table.
  `report by_setting` runs one. The first line of each file is its title. The reports are:
  `by_setting` (F1, speed, lag per setting), `cache_effect` (cache on vs off, same setting),
  `machines` (same setting across the 4070, Docker and the T4), `exactness` (every stride 512
  check), `stages` (where a pass spends its time).
- The database is committed (about 60 KB). `import` rebuilds it from the run folders and results.md
  if it is ever lost, except the batch note counts and array check results of scores that only
  `score_engine.py` wrote.

## 11. Deploy: a GPU box behind an always-on front, for a couple of users

Two machines, both made by `infra/deploy/`:

- **GPU box**: spot `g4dn.xlarge` (persistent request: a reclaim stops it, AWS restarts it when
  capacity returns), running the service image (`docker build --target service`: the engine plus
  `ui/server.py --max-sessions 5 --trt`, one engine per connected page, five at once on TensorRT
  with the mixed-precision graphs; results.md, "Tesla T4 with TensorRT"). It has no public ports except SSH
  from one address; its page and event ports (8000, 8001) answer only the front. It stops itself
  after `max_minutes` (default 8 hours, about $2 of T4 time).
- **Front**: a `t4g.nano` (about $3 a month) that holds the fixed address and runs Caddy as a plain
  binary (no Docker: 512 MB is not enough to install it): HTTPS with a Let's Encrypt certificate
  for `<ip-with-dashes>.sslip.io`, two logins, and a proxy to the GPU box over the VPC. When the
  box is stopped or reclaimed, the proxy fails and Caddy serves `deploy/front/offline.html`
  ("the piano engine is offline right now") with a **Start the engine** button instead of going dark.
- **Two logins** (Caddy basic auth; the login name reaches the service as `X-User`, set by Caddy so
  a client cannot forge it): the admin (`auth_user`, default `sam`) can add songs and start the GPU
  box as often as needed; the guest (`guest_user`, default `guest`) can play and transcribe, sees
  no drop box (and the server answers 403 to an upload anyway), and may start the box
  `guest_starts_per_day` times a day (default 3, UTC days, counted in `/opt/pianofi/data/starts.json`
  on the front). The button is `deploy/front/start.py` behind Caddy at `/start`; it calls
  `aws ec2 start-instances` with the front's IAM role, which allows exactly that one instance
  (`infra/deploy/iam.tf`).
- **Idle stop**: `deploy/run.sh` installs a systemd timer on the GPU box (`deploy/idle_stop.sh`,
  once a minute, reads `/status` from the service) that shuts the machine down after `IDLE_MINUTES`
  (default 30) with no engine running. A stopped box costs only its disk; the Start button brings it
  back. The old `max_minutes` cap still applies to the first boot.

Before the first deploy: `infra/deploy/terraform.tfvars` with `my_ip_cidr`, `auth_hash` and
`guest_hash` (copy the example; a hash comes from `docker run --rm caddy:2 caddy hash-password
--plaintext '...'`, or `caddy hash-password` on the front), `terraform -chdir=infra/deploy init`, the
SSH key from step 9, and `aws sso login --profile pianofi-live`.

```powershell
terraform -chdir=infra/deploy apply
tar -czf runs\pf-deploy.tgz --exclude cpp/build Dockerfile .dockerignore cpp ui deploy models
scp -i ~\.ssh\pianofi_live runs\pf-deploy.tgz ec2-user@<gpu_public_ip>:pf/
scp -i ~\.ssh\pianofi_live test_audio\16k\*.wav ec2-user@<gpu_public_ip>:pf/songs/
ssh -i ~\.ssh\pianofi_live ec2-user@<gpu_public_ip> "cd pf && tar -xzf pf-deploy.tgz && sed -i 's/\r$//' deploy/*.sh && chmod +x deploy/run.sh && ./deploy/run.sh"
terraform -chdir=infra/deploy destroy
```

- `<gpu_public_ip>` is a Terraform output (it changes when the box is replaced; `aws ec2
  describe-instances` has the current one; it also changes every time the box is stopped and
  started, the private address and the hostname do not). The front needs nothing shipped: its
  first boot installs Caddy and the files from `deploy/front/` (Caddyfile, offline page, `start.py`,
  `install.sh`) and writes `/opt/pianofi/env` (logins, the GPU box's address and id). A change to
  those files is shipped by hand, not by replacing the machine (each replacement costs a
  certificate, and Let's Encrypt allows 5 a week per name):
  `scp -i ~\.ssh\pianofi_live -r deploy\front ec2-user@<public_ip>:` then
  `ssh ... "sudo bash front/install.sh"`. A change that only touches the first-boot script makes
  Terraform stop and start the front (about 1 minute; the address and certificate survive).
- `run.sh` first builds the TensorRT engines with a 5 s run of the engine (about 40 s on a new
  machine, seconds after that): they are kept in `models/trt_cache`, and the service mounts
  `models` read-only. Five sessions is what fits in the T4's memory; a sixth engine gets no GPU
  memory and falls back to the CPU without failing, so do not raise `--max-sessions` past 5.
- `run.sh` starts the service container last, after the idle-stop timer is installed. A
  `systemctl daemon-reload` takes the GPU away from containers that are already running: the
  engines then log `CUDA unavailable ... using CPU` and no notes arrive. Check with
  `sudo docker exec service nvidia-smi`; `sudo docker restart service` brings the GPU back.
- On the GPU box, `ADMIN=sam IDLE_MINUTES=30 ./deploy/run.sh` are the defaults; `IDLE_MINUTES=0`
  turns the idle stop off.
- The first build on the box takes a few minutes (CUDA image, ONNX Runtime, compile, `pf_tests`);
  after that `run.sh` restarts in seconds. `run.sh SITE USER HASH` is the old single-box form
  (Caddy on the GPU box itself).
- The page is `https://<site>`, user `sam` or `guest`, the passwords you hashed. The browser's mic
  and tab audio need HTTPS, which is why Caddy is there.
- Lag through the cloud is measured by the page: "lag (end to end)" is from the sample leaving the
  browser to its note arriving back. The engine's own lag is in `runs/ui/session-N/metrics.json`
  on the box.
- Always `destroy` when done, and check that the address is released (an unattached Elastic IP
  costs money) and that no spot request is left open:
  `aws ec2 describe-spot-instance-requests --filters Name=state,Values=open,active`.
- Zones on this account, 2026-11-07: us-east-1d's default subnet routes through a NAT gateway (no
  public path in or out, the box is unreachable); 1a, 1b, 1c, 1e, 1f route through the internet
  gateway. us-east-1b had no spot capacity; 1a reclaimed a one-time machine after 7 hours.
- Replacing the GPU box takes 7 minutes (the old one takes about 6.5 to terminate). If `apply` is
  cut off while the new one is being created, the machine exists but is not in the state:
  `terraform -chdir=infra/deploy import aws_instance.gpu <id>` and apply again.
- The GPU box's security group keeps its original description: changing it would replace the
  group while it is attached. Its rules are all inline (a separate `aws_security_group_rule`
  next to inline rules gets stripped on every apply).

## Next steps (not built yet)

- GPU-side h cache (IOBinding) so the 512-frame window isn't copied to the GPU every pass.
