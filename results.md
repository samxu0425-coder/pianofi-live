
## Stream simulation, 2026-11-05 20:57

Style level2, seed 0, device cuda, 14 torch threads. Reference: Pipeline.wav2midi notes. Match: pitch and onset within 50 ms, offsets ignored.

### 255309__soundatic__piano_harmonicmodel.wav

3.9 s, 15 reference notes.

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 1.0000 | 0.6667 | 0.8000 | 32 | 150.4 |
| 0.512 s | 1.0000 | 0.8000 | 0.8889 | 16 | 147.0 |
| 1.024 s | 1.0000 | 0.8000 | 0.8889 | 8 | 147.4 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 1 | 145.8 |

### 367533__leewarner__mypianophrase.wav

4.5 s, 9 reference notes.

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 1.0000 | 0.7778 | 0.8750 | 32 | 147.2 |
| 0.512 s | 1.0000 | 0.8889 | 0.9412 | 16 | 146.9 |
| 1.024 s | 1.0000 | 1.0000 | 1.0000 | 8 | 147.1 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 1 | 146.0 |

### Average (2 songs)

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 1.0000 | 0.7222 | 0.8375 | 32 | 148.8 |
| 0.512 s | 1.0000 | 0.8444 | 0.9150 | 16 | 147.0 |
| 1.024 s | 1.0000 | 0.9000 | 0.9444 | 8 | 147.2 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 1 | 145.9 |

## Stream simulation, 2026-11-05 21:04

Style level2, seed 0, device cuda, 14 torch threads. Reference: Pipeline.wav2midi notes. Match: pitch and onset within 50 ms, offsets ignored.

### 479957__liezen3__panflute-phrase-and-piano.wav

42.0 s, 141 reference notes.

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 0.9103 | 0.9362 | 0.9231 | 192 | 149.3 |
| 0.512 s | 0.9310 | 0.9574 | 0.9441 | 96 | 149.0 |
| 1.024 s | 0.9320 | 0.9716 | 0.9514 | 48 | 150.4 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 6 | 148.6 |

### 686610__seth_makes_sounds__bubbglegum-pop-song.wav

119.0 s, 1495 reference notes.

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 0.9160 | 0.9043 | 0.9101 | 480 | 148.6 |
| 0.512 s | 0.9219 | 0.9237 | 0.9228 | 240 | 152.6 |
| 1.024 s | 0.9227 | 0.9344 | 0.9285 | 120 | 157.8 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 15 | 147.3 |

### Average (2 songs)

| stride | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|
| 0.256 s | 0.9132 | 0.9203 | 0.9166 | 336 | 148.9 |
| 0.512 s | 0.9265 | 0.9406 | 0.9334 | 168 | 150.8 |
| 1.024 s | 0.9273 | 0.9530 | 0.9400 | 84 | 154.1 |
| 8.192 s | 1.0000 | 1.0000 | 1.0000 | 10.5 | 148.0 |

## C++ engine runs

F1 vs batch notes (50 ms onsets, offsets ignored). Lag = time from audio arriving to its output frame being ready (real-time runs with 0 overruns only). RTF = compute time / clip length, so it reads too low on runs with overruns: compare pass ms to the stride there. Overruns = audio samples dropped. Rows marked "spsc ring" used the lock-free ring buffer, and all others used the mutex one. Commands run from the repo root, and each run is scored with `.venv\Scripts\python.exe eval\score_engine.py <--out dir> <--wav file>`. GPU runs need the CUDA DLLs on PATH (see RUN.md).

| when | song | device | stride | heads | cache | P | R | F1 | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | RTF | overruns | decoder | machine | command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-06 00:45 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 182.6 | 278.6 | not measured | not measured | 0.024 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs\check512` |
| 2026-11-06 00:50 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 512 ms | b | no | 0.9191 | 0.9167 | 0.9179 | 175.9 | 201.9 | 1003 | 1257 | 0.381 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-nocache-512ms --dump --no-cache --stride 32` |
| 2026-11-06 00:52 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 512 ms | b | yes | 0.9191 | 0.9167 | 0.9179 | 65.5 | 80.1 | 887 | 1144 | 0.138 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-cache-512ms --dump --stride 32` |
| 2026-11-06 00:53 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 256 ms | b | yes | 0.9110 | 0.9074 | 0.9092 | 61.1 | 78.1 | 755 | 889 | 0.257 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-cache-256ms --dump --stride 16` |
| 2026-11-06 00:54 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 128 ms | b | yes | 0.9188 | 0.8836 | 0.9009 | 43.1 | 64.2 | 675 | 740 | 0.382 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-cache-128ms --dump --stride 8` |
| 2026-11-06 00:55 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 64 ms | b | yes | 0.9268 | 0.8704 | 0.8977 | 48.7 | 66.6 | 649 | 686 | 0.848 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-cache-64ms --dump --stride 4` |
| 2026-11-06 00:56 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 128 ms | a | yes | 0.9312 | 0.6442 | 0.7615 | 4.0 | 38.3 | 639 | 723 | 0.077 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-a-heads-128ms --dump --heads a --stride 8` |
| 2026-11-06 00:57 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 203.4 | 288.2 | 4876 | 8887 | 0.028 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-gpu-batch-windows --dump --stride 512` |
| 2026-11-06 01:02 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CPU | 512 ms | b | no | 0.1083 | 0.0172 | 0.0297 | 3751.4 | 4027.9 | not measured | not measured | 2.014 | 787072 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-cpu-nocache-512ms --dump --cpu --no-cache --stride 32` |
| 2026-11-06 01:02 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CPU | 512 ms | b | yes | 0.1132 | 0.0794 | 0.0933 | 832.9 | 994.4 | not measured | not measured | 1.118 | 341984 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --models models --out runs\686610__seth_makes_sounds__bubbglegum-pop-song_first60s-cpu-cache-512ms --dump --cpu --stride 32` |
| 2026-11-06 01:21 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA | 256 ms + delay 512 ms | b | yes | 0.9241 | 0.9284 | 0.9263 | 52.9 | 70.2 | not measured | not measured | 0.215 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 16 --delay 32 --fast --dump --out runs\delay-s16-d32` |
| 2026-11-06 01:22 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 178.2 | 277.2 | not measured | not measured | 0.023 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs\check512` |
| 2026-11-06 01:24 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 256 ms + delay 512 ms | b | yes | 0.9196 | 0.9378 | 0.9286 | 59.9 | 75.7 | 1265 | 1401 | 0.252 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 32 --dump --out runs\delay-s16-d32-rt` |
| 2026-11-06 02:01 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 63.2 | 75.6 | 1013 | 1144 | 0.267 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs\delay-s16-d16-rt` |
| 2026-11-06 02:11 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 187.8 | 284.8 | not measured | not measured | 0.025 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs\check512` |
| 2026-11-06 02:12 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 62.2 | 77.4 | 1013 | 1144 | 0.261 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs\spsc-s16-d16-rt` |

## Ring buffer benchmark, 2026-11-06

Command: `cpp\build\Release\bench_ring.exe` (built with `-DPF_USE_SPSC=ON`), run 3 times.

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10. Other apps were open (CPU load 19 to 35% before the runs).

One producer pushes 160-sample blocks as fast as it can for 2 s while one consumer pops. This is a stress test, far busier than the real audio thread's one push per 10 ms. Every timing is a multiple of 100 ns, so that looks like the clock's resolution and p50 can't separate the two.

| run | buffer | M samples/s | push p50 ns | push p99 ns | push p99.9 ns | push max ns |
|---|---|---|---|---|---|---|
| 1 | mutex | 437.0 | 100 | 2800 | 16500 | 1603900 |
| 1 | spsc | 1214.8 | 100 | 300 | 400 | 142700 |
| 2 | mutex | 381.8 | 100 | 2100 | 17100 | 1547900 |
| 2 | spsc | 1224.3 | 100 | 300 | 400 | 227900 |
| 3 | mutex | 395.5 | 100 | 2800 | 21300 | 810200 |
| 3 | spsc | 1301.2 | 100 | 200 | 300 | 101900 |

## Stream simulation, 2026-11-06 01:07

Command: `python py\stream_sim.py --strides 0.256 0.512 --delays 0 16 32 64 128`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

Style level2, seed 0, device cuda, 14 torch threads. Reference: Pipeline.wav2midi notes. Match: pitch and onset within 50 ms, offsets ignored.

### 479957__liezen3__panflute-phrase-and-piano.wav

42.0 s, 141 reference notes.

| stride | delay ms | est. lag ms | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|---|---|
| 0.256 s | 0 | 832 | 0.9103 | 0.9362 | 0.9231 | 192 | 158.3 |
| 0.256 s | 256 | 1088 | 0.9262 | 0.9787 | 0.9517 | 192 | 156.4 |
| 0.256 s | 512 | 1344 | 0.9384 | 0.9716 | 0.9547 | 192 | 156.5 |
| 0.256 s | 1024 | 1856 | 0.9586 | 0.9858 | 0.9720 | 192 | 156.8 |
| 0.256 s | 2048 | 2880 | 0.9448 | 0.9716 | 0.9580 | 192 | 156.3 |
| 0.512 s | 0 | 1088 | 0.9310 | 0.9574 | 0.9441 | 96 | 157.2 |
| 0.512 s | 256 | 1344 | 0.9320 | 0.9716 | 0.9514 | 96 | 155.8 |
| 0.512 s | 512 | 1600 | 0.9388 | 0.9787 | 0.9583 | 96 | 156.5 |
| 0.512 s | 1024 | 2112 | 0.9456 | 0.9858 | 0.9653 | 96 | 158.7 |
| 0.512 s | 2048 | 3136 | 0.9452 | 0.9787 | 0.9617 | 96 | 154.9 |
| 8.192 s | 0 | 8768 | 1.0000 | 1.0000 | 1.0000 | 6 | 160.0 |

### 686610__seth_makes_sounds__bubbglegum-pop-song.wav

119.0 s, 1495 reference notes.

| stride | delay ms | est. lag ms | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|---|---|
| 0.256 s | 0 | 832 | 0.9160 | 0.9043 | 0.9101 | 480 | 151.6 |
| 0.256 s | 256 | 1088 | 0.9211 | 0.9371 | 0.9290 | 480 | 153.3 |
| 0.256 s | 512 | 1344 | 0.9241 | 0.9284 | 0.9263 | 480 | 148.9 |
| 0.256 s | 1024 | 1856 | 0.9272 | 0.9458 | 0.9364 | 480 | 155.8 |
| 0.256 s | 2048 | 2880 | 0.9384 | 0.9378 | 0.9381 | 480 | 157.2 |
| 0.512 s | 0 | 1088 | 0.9219 | 0.9237 | 0.9228 | 240 | 156.5 |
| 0.512 s | 256 | 1344 | 0.9220 | 0.9331 | 0.9275 | 240 | 157.6 |
| 0.512 s | 512 | 1600 | 0.9212 | 0.9378 | 0.9294 | 240 | 157.2 |
| 0.512 s | 1024 | 2112 | 0.9329 | 0.9478 | 0.9403 | 240 | 157.7 |
| 0.512 s | 2048 | 3136 | 0.9411 | 0.9405 | 0.9408 | 240 | 157.4 |
| 8.192 s | 0 | 8768 | 1.0000 | 1.0000 | 1.0000 | 15 | 158.4 |

### Average (2 songs)

| stride | delay ms | est. lag ms | precision | recall | F1 | passes | mean pass ms |
|---|---|---|---|---|---|---|---|
| 0.256 s | 0 | 832 | 0.9132 | 0.9203 | 0.9166 | 336 | 155.0 |
| 0.256 s | 256 | 1088 | 0.9236 | 0.9579 | 0.9404 | 336 | 154.8 |
| 0.256 s | 512 | 1344 | 0.9312 | 0.9500 | 0.9405 | 336 | 152.7 |
| 0.256 s | 1024 | 1856 | 0.9429 | 0.9658 | 0.9542 | 336 | 156.3 |
| 0.256 s | 2048 | 2880 | 0.9416 | 0.9547 | 0.9481 | 336 | 156.8 |
| 0.512 s | 0 | 1088 | 0.9265 | 0.9406 | 0.9334 | 168 | 156.9 |
| 0.512 s | 256 | 1344 | 0.9270 | 0.9524 | 0.9395 | 168 | 156.7 |
| 0.512 s | 512 | 1600 | 0.9300 | 0.9583 | 0.9439 | 168 | 156.8 |
| 0.512 s | 1024 | 2112 | 0.9392 | 0.9668 | 0.9528 | 168 | 158.2 |
| 0.512 s | 2048 | 3136 | 0.9432 | 0.9596 | 0.9512 | 168 | 156.1 |
| 8.192 s | 0 | 8768 | 1.0000 | 1.0000 | 1.0000 | 10.5 | 159.2 |
