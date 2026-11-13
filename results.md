
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

The same rows, plus every run's full `metrics.json`, are in the SQLite store `runs/results.db` (RUN.md step 10).
The summary below is the `machines` report, `.venv\Scripts\python.exe eval\results_db.py report machines`, run
2026-11-07: every setting that ran on more than one machine gave the same F1 and note count everywhere, and only
the pass time changed.

| stride_ms | delay_ms | gpu | runs | f1_min | f1_max | notes_min | notes_max | pass_p50_min | pass_p50_max | overruns |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0 | RTX 4070 | 1 | 0.8977 | 0.8977 | 710 | 710 | 48.7 | 48.7 | 0 |
| 128 | 0 | RTX 4070 | 1 | 0.9009 | 0.9009 | 727 | 727 | 43.1 | 43.1 | 0 |
| 128 | 0 | T4 (AWS) | 1 | 0.9009 | 0.9009 | 727 | 727 | 97.2 | 97.2 | 0 |
| 256 | 0 | RTX 4070 | 1 | 0.9092 | 0.9092 | 753 | 753 | 61.1 | 61.1 | 0 |
| 256 | 256 | RTX 4070 | 2 | 0.9233 | 0.9233 | 769 | 769 | 62.2 | 63.2 | 0 |
| 256 | 256 | RTX 4070 (Docker) | 1 | 0.9233 | 0.9233 | 769 | 769 | 57.8 | 57.8 | 0 |
| 256 | 256 | T4 (AWS) | 3 | 0.9233 | 0.9233 | 769 | 769 | 102.0 | 156.1 | 0 |
| 256 | 512 | RTX 4070 | 1 | 0.9286 | 0.9286 | 771 | 771 | 59.9 | 59.9 | 0 |
| 512 | 0 | RTX 4070 | 1 | 0.9179 | 0.9179 | 754 | 754 | 65.5 | 65.5 | 0 |
| 8192 | 0 | RTX 4070 | 1 | 1.0000 | 1.0000 | 756 | 756 | 203.4 | 203.4 | 0 |
| 2026-11-12 16:00 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 169.8 | 239.9 | not measured | not measured | 0.022 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs/check512` |
| 2026-11-12 16:02 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA FP16, spsc ring | 8192 ms | b | yes | 0.9967 | 0.9993 | 0.9980 | 92.9 | 155.2 | not measured | not measured | 0.012 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --fp16 --out runs/fp16_check512` |
| 2026-11-12 16:03 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 41.1 | 45.3 | 992 | 1114 | 0.177 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/fp32_s16_d16_rt` |
| 2026-11-12 16:05 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9131 | 0.9312 | 0.9221 | 26.9 | 30.0 | 979 | 1100 | 0.115 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --fp16 --out runs/fp16_s16_d16_rt` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 443.1 | 551.5 | not measured | not measured | 0.057 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs/t4b_check512` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA FP16, spsc ring | 8192 ms | b | yes | 0.9967 | 0.9973 | 0.9970 | 202.2 | 296.3 | not measured | not measured | 0.026 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --fp16 --dump --out runs/t4b_check512_fp16` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 96.6 | 102.9 | 1047 | 1167 | 0.415 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_s16_d16` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 51.1 | 52.8 | 1001 | 1121 | 0.219 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_s16_d16_fp16` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 99.0 | 209.2 | 1049 | 1170 | 0.469 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x2_1` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 98.0 | 209.1 | 1048 | 1169 | 0.465 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x2_2` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 101.5 | 107.4 | 1052 | 1172 | 0.435 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x2_fp16_1` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 101.4 | 107.5 | 1052 | 1172 | 0.436 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x2_fp16_2` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1925 | 0.1693 | 0.1802 | 313.6 | 322.0 | not measured | not measured | 1.166 | 149632 | fell behind (overruns), three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x3_1` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1897 | 0.1746 | 0.1818 | 313.6 | 320.8 | not measured | not measured | 1.162 | 146464 | fell behind (overruns), three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x3_2` |
| 2026-11-12 20:22 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1785 | 0.1627 | 0.1702 | 313.2 | 322.8 | not measured | not measured | 1.164 | 151872 | fell behind (overruns), three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x3_3` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 138.8 | 155.9 | 1089 | 1211 | 0.589 | 0 | ok, three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x3_fp16_1` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 138.9 | 149.1 | 1090 | 1211 | 0.591 | 0 | ok, three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x3_fp16_2` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 127.4 | 155.9 | 1078 | 1205 | 0.546 | 0 | ok, three engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x3_fp16_3` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1660 | 0.1151 | 0.1359 | 420.2 | 437.7 | not measured | not measured | 1.111 | 340800 | fell behind (overruns), four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x4_1` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1291 | 0.0886 | 0.1051 | 418.7 | 437.9 | not measured | not measured | 1.113 | 344704 | fell behind (overruns), four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x4_2` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1660 | 0.1151 | 0.1359 | 418.6 | 429.7 | not measured | not measured | 1.113 | 338848 | fell behind (overruns), four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x4_3` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.1453 | 0.1032 | 0.1206 | 418.6 | 428.1 | not measured | not measured | 1.111 | 338048 | fell behind (overruns), four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/t4b_x4_4` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 132.2 | 199.5 | 1086 | 1209 | 0.588 | 0 | ok, four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x4_fp16_1` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 134.5 | 199.9 | 1091 | 1214 | 0.602 | 0 | ok, four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x4_fp16_2` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 101.7 | 191.0 | 1058 | 1180 | 0.475 | 0 | ok, four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x4_fp16_3` |
| 2026-11-12 20:23 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA FP16, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9143 | 0.9312 | 0.9227 | 96.3 | 190.0 | 1056 | 1190 | 0.467 | 0 | ok, four engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --fp16 --dump --out runs/t4b_x4_fp16_4` |

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
| 2026-11-06 01:24 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 256 ms + delay 512 ms | b | yes | 0.9196 | 0.9378 | 0.9286 | 59.9 | 75.7 | 1265 | 1401 | 0.252 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 32 --dump --out runs\delay-s16-d32-rt` |
| 2026-11-06 02:01 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 63.2 | 75.6 | 1013 | 1144 | 0.267 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs\delay-s16-d16-rt` |
| 2026-11-06 02:12 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 62.2 | 77.4 | 1013 | 1144 | 0.261 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs\spsc-s16-d16-rt` |
| 2026-11-06 02:42 | prettifun - Light (Instrumental) - swagdisease (128k).wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.8421 | 0.9192 | 0.8790 | 54.9 | 73.0 | not measured | not measured | 0.221 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav "test_audio\16k\prettifun - Light (Instrumental) - swagdisease (128k).wav" --stride 16 --delay 16 --fast --dump --out runs\prettifun-s16-d16` |
| 2026-11-06 13:38 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 189.8 | 259.9 | not measured | not measured | 0.025 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10 | `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs\check512` |
| 2026-11-06 14:52 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 200.7 | 333.3 | not measured | not measured | 0.026 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10, in Docker Desktop (Linux container on WSL 2) | `docker run --rm --gpus all -v ${PWD}\models:/work/models -v ${PWD}\test_audio:/work/test_audio -v ${PWD}\runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs/docker_check512` |
uns:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs/docker_check512` |
| 2026-11-06 14:55 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 57.8 | 72.4 | 1008 | 1135 | 0.244 | 0 | ok | 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10, in Docker Desktop (Linux container on WSL 2) | `docker run --rm --gpus all -v ${PWD}\models:/work/models -v ${PWD}\test_audio:/work/test_audio -v ${PWD}\runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/docker_s16_d16` |
uns:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/docker_s16_d16` |
| 2026-11-06 21:51 | 686610__seth_makes_sounds__bubbglegum-pop-song.wav | CUDA, spsc ring | 8192 ms | b | yes | 1.0000 | 1.0000 | 1.0000 | 453.1 | 564.1 | not measured | not measured | 0.058 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song.wav --stride 512 --fast --dump --out runs/aws2_t4_check512` |
| 2026-11-06 21:52 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 102.0 | 106.8 | 1052 | 1173 | 0.437 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/aws2_t4_s16_d16` |
| 2026-11-06 21:53 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 128 ms | b | yes | 0.9188 | 0.8836 | 0.9009 | 97.2 | 101.9 | 728 | 788 | 0.833 | 0 | ok | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 8 --dump --out runs/aws2_t4_s8` |
| 2026-11-06 21:55 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 148.0 | 204.8 | 1098 | 1219 | 0.656 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/aws2_t4_two_a` |
| 2026-11-06 21:55 | 686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav | CUDA, spsc ring | 256 ms + delay 256 ms | b | yes | 0.9155 | 0.9312 | 0.9233 | 156.1 | 203.8 | 1106 | 1227 | 0.687 | 0 | ok, two engines at once on one T4 | AWS g4dn.xlarge spot (Tesla T4 15 GB, Xeon 8259CL 4 vCPUs), Amazon Linux 2023 ECS GPU image, in Docker | `docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --dump --out runs/aws2_t4_two_b` |

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

## Live input, 2026-11-06 02:23

Command, run while `test_audio\686610__seth_makes_sounds__bubbglegum-pop-song.wav` was playing on the PC: `cpp\build\Release\pianofi_live.exe --input loopback --stride 16 --delay 16 --max-seconds 60 --out runs\live-loopback`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10. Device: Speakers (Realtek(R) Audio), WASAPI loopback. CUDA, lock-free ring buffer.

| input | seconds | passes | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | overruns | notes |
|---|---|---|---|---|---|---|---|---|
| loopback | 60.0 | 256 | 62.7 | 78.0 | 1011 | 1143 | 0 | 741 |

F1 vs batch: not measured. A live capture has no fixed start time, so it can't be scored like a file.

## Live input, 2026-11-06 02:32

Command, run while `prettifun - Light (Instrumental) - swagdisease (128k).wav` (48 kHz original) was playing on the PC: `cpp\build\Release\pianofi_live.exe --input loopback --stride 16 --delay 16 --max-seconds 60 --out runs\live-loopback-prettifun`

Same machine and device as above. Windows master volume was 65% (read about 10 minutes later). The capture is quieter than the file and the engine finds fewer notes at lower volume, so the note count depends on the PC's volume.

| input | seconds | passes | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | overruns | notes |
|---|---|---|---|---|---|---|---|---|
| loopback | 60.0 | 256 | 59.0 | 76.9 | 1012 | 1144 | 0 | 120 |

F1 vs batch: not measured, for the same reason.

## GPU memory per engine, 2026-11-06 04:37

Command: `python eval\gpu_memory.py test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

686610__seth_makes_sounds__bubbglegum-pop-song.wav, file input, `--fast`, one engine at a time. GPU memory is the peak rise in `nvidia-smi` memory.used during the run, sampled every 0.5 s, so short spikes can be missed.

| settings | GPU memory MB | pass p50 ms | pass max ms |
|---|---|---|---|
| stride 256 ms + delay 256 ms, cached | 3083 | 52.3 | 71 |
| stride 64 ms, cached | 2907 | 50.8 | 76 |
| stride 256 ms, no cache | 6001 | 162.6 | 193 |
| stride 8192 ms (batch windows) | 6180 | 171.4 | 240 |

## Live input with automatic level, 2026-11-06 04:59

Command: `python ui\server.py --input loopback`, then Play in the page, while `test_audio\686610__seth_makes_sounds__bubbglegum-pop-song.wav` played on the PC, then Stop. The engine ran `--input loopback --stride 16 --delay 16 --events` with automatic input level on.

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10. Device: Speakers (Realtek(R) Audio) (WASAPI, device format 16000 Hz, 2 channels). CUDA, lock-free ring buffer.

| input | seconds | passes | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | overruns | notes | gain p50 | gain max |
|---|---|---|---|---|---|---|---|---|---|---|
| loopback | 66.8 | 262 | 66.2 | 75.4 | 1011 | 1140 | 0 | 938 | x1.34 | x14.64 |

F1 vs batch: not measured (a live capture has no fixed start time). Stop ended the engine cleanly (exit code 0, notes.csv and metrics.json written).

## Automatic input level, 2026-11-06 05:02

Command: `python eval\level_eval.py`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

Whole songs through the engine (file input, stride 256 ms + delay 256 ms), changed as the first column says. P, R and F1 are against batch notes of the original file (50 ms onsets, offsets ignored). "Tone tilt" is the treble boost measured on this PC's sound output. "Dip" is x0.4 volume with 20 s at x0.03 in the middle.

### 479957__liezen3__panflute-phrase-and-piano.wav

42.0 s, file peak 0.248, 141 batch notes.

| input | auto level | notes | P | R | F1 | median gain |
|---|---|---|---|---|---|---|
| file as is | off | 149 | 0.9262 | 0.9787 | 0.9517 | - |
| file as is | on | 163 | 0.7914 | 0.9149 | 0.8487 | x4.41 |
| x0.4 | off | 127 | 0.9528 | 0.8582 | 0.9030 | - |
| x0.4 | on | 163 | 0.7914 | 0.9149 | 0.8487 | x11.03 |
| x0.1 | off | 80 | 0.9375 | 0.5319 | 0.6787 | - |
| x0.1 | on | 164 | 0.8049 | 0.9362 | 0.8656 | x32.00 |
| x0.03 | off | 27 | 1.0000 | 0.1915 | 0.3214 | - |
| x0.03 | on | 149 | 0.9262 | 0.9787 | 0.9517 | x32.00 |
| tone tilt, x0.4 | off | 131 | 0.9389 | 0.8723 | 0.9044 | - |
| tone tilt, x0.4 | on | 161 | 0.7888 | 0.9007 | 0.8411 | x10.63 |
| dip | off | 71 | 0.8873 | 0.4468 | 0.5943 | - |
| dip | on | 145 | 0.8138 | 0.8369 | 0.8252 | x11.31 |

### 686610__seth_makes_sounds__bubbglegum-pop-song.wav

119.0 s, file peak 0.879, 1495 batch notes.

| input | auto level | notes | P | R | F1 | median gain |
|---|---|---|---|---|---|---|
| file as is | off | 1521 | 0.9211 | 0.9371 | 0.9290 | - |
| file as is | on | 1592 | 0.8838 | 0.9411 | 0.9116 | x1.30 |
| x0.4 | off | 1385 | 0.9191 | 0.8515 | 0.8840 | - |
| x0.4 | on | 1592 | 0.8838 | 0.9411 | 0.9116 | x3.26 |
| x0.1 | off | 1003 | 0.8923 | 0.5987 | 0.7166 | - |
| x0.1 | on | 1577 | 0.8928 | 0.9418 | 0.9167 | x13.03 |
| x0.03 | off | 350 | 0.9429 | 0.2207 | 0.3577 | - |
| x0.03 | on | 1520 | 0.9224 | 0.9378 | 0.9300 | x32.00 |
| tone tilt, x0.4 | off | 1479 | 0.9040 | 0.8943 | 0.8991 | - |
| tone tilt, x0.4 | on | 1651 | 0.8613 | 0.9512 | 0.9040 | x2.70 |
| dip | off | 1231 | 0.9050 | 0.7452 | 0.8173 | - |
| dip | on | 1516 | 0.8852 | 0.8977 | 0.8914 | x3.26 |

### prettifun - Light (Instrumental) - swagdisease (128k).wav

185.0 s, file peak 1.061, 557 batch notes.

| input | auto level | notes | P | R | F1 | median gain |
|---|---|---|---|---|---|---|
| file as is | off | 608 | 0.8421 | 0.9192 | 0.8790 | - |
| file as is | on | 599 | 0.8464 | 0.9102 | 0.8772 | x0.96 |
| x0.4 | off | 370 | 0.8703 | 0.5781 | 0.6947 | - |
| x0.4 | on | 599 | 0.8464 | 0.9102 | 0.8772 | x2.41 |
| x0.1 | off | 165 | 0.7758 | 0.2298 | 0.3546 | - |
| x0.1 | on | 599 | 0.8464 | 0.9102 | 0.8772 | x9.65 |
| x0.03 | off | 45 | 0.8000 | 0.0646 | 0.1196 | - |
| x0.03 | on | 597 | 0.8492 | 0.9102 | 0.8787 | x32.00 |
| tone tilt, x0.4 | off | 423 | 0.8558 | 0.6499 | 0.7388 | - |
| tone tilt, x0.4 | on | 640 | 0.8016 | 0.9210 | 0.8571 | x2.15 |
| dip | off | 344 | 0.8663 | 0.5350 | 0.6615 | - |
| dip | on | 572 | 0.8409 | 0.8636 | 0.8521 | x2.42 |

## Long-session memory, 2026-11-06 05:18

Command: `python eval\memory_check.py`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

File input, `--fast`. "Dropped" runs use `--bounded-memory`, which is what live input always does.

| song | settings | notes | notes identical with history dropped |
|---|---|---|---|
| 479957__liezen3__panflute-phrase-and-piano.wav | `--stride 16 --delay 16` | 149 | yes |
| 479957__liezen3__panflute-phrase-and-piano.wav | `--stride 4` | 135 | yes |
| 686610__seth_makes_sounds__bubbglegum-pop-song.wav | `--stride 16 --delay 16` | 1521 | yes |
| 686610__seth_makes_sounds__bubbglegum-pop-song.wav | `--stride 4` | 1389 | yes |
| prettifun - Light (Instrumental) - swagdisease (128k).wav | `--stride 16 --delay 16` | 608 | yes |
| prettifun - Light (Instrumental) - swagdisease (128k).wav | `--stride 4` | 574 | yes |

The test songs repeated to the given length, stride 256 ms + delay 256 ms. Frames held = the most kept in memory at once. Old mel frames are dropped in every run without `--dump`, so only the decoder column differs. Peak memory is the whole process, which includes about 1 GB of CUDA libraries and the wav file itself (3.7 MB per minute).

| length | history | decoder frames held | mel frames held | peak memory MB |
|---|---|---|---|---|
| 5 min | kept | 18944 | 4205 | 994 |
| 5 min | dropped | 1296 | 4205 | 962 |
| 20 min | kept | 75264 | 4205 | 1183 |
| 20 min | dropped | 1490 | 4205 | 1019 |

## AWS spot GPU (Tesla T4), 2026-11-06 22:09 UTC

Machine: AWS `g4dn.xlarge` spot in us-east-1a (Tesla T4 15 GB, Intel Xeon Platinum 8259CL, 4 vCPUs), Amazon
Linux 2023 ECS GPU image, NVIDIA driver 580.178.04, Docker 25.0.14. Engine built there from the `Dockerfile`.

Commands: RUN.md step 9 creates the machine and builds the image. Each run is
`docker run --rm --gpus all -v $PWD/models:/work/models -v $PWD/test_audio:/work/test_audio -v $PWD/runs:/work/runs pianofi-live --wav test_audio/16k/<song> <settings> --dump --out runs/<name>`.
Song: `686610__seth_makes_sounds__bubbglegum-pop-song.wav` for stride 8192 ms, its first 60 s for the rest.
GPU memory is the highest `nvidia-smi` reading during the run (0.5 s samples, 1 MB when idle).

AWS took the spot machine back at 22:19:50 UTC, before the run folders were copied off it. So these
numbers are the engine's own printed summary, and nothing here was scored against batch: P, R, F1 and
the array and decoder checks are **not measured** on this machine. The note counts match the same runs
on the RTX 4070 (1495 and 769).

| settings | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | RTF | overruns | notes | GPU memory MB | F1 |
|---|---|---|---|---|---|---|---|---|---|
| `--stride 512 --fast` | 463.7 | 567.4 | not measured | not measured | 0.059 | 0 | 1495 | 6153 | not measured |
| `--stride 16 --delay 16` | 102.6 | 108.0 | 1053 | 1173 | 0.440 | 0 | 769 | 3021 | not measured |
| `--stride 8` | 98.4 | 103.0 | 729 | 789 | 0.843 | 0 | 727 | 2863 | not measured |
| `--stride 4` | 99.5 | 102.0 | not measured | not measured | 1.062 | 308,800 | 507 | 2845 | not measured |
| `--stride 16 --delay 16 --no-cache` | 472.1 | 478.7 | not measured | not measured | 1.259 | 400,992 | 466 | 5939 | not measured |
| `--stride 16 --delay 16`, two engines at once (A) | 190.2 | 231.4 | 1140 | 1262 | 0.820 | 0 | 769 | not measured | not measured |
| `--stride 16 --delay 16`, two engines at once (B) | 192.6 | 222.3 | 1143 | 1265 | 0.831 | 0 | 769 | not measured | not measured |

Lag is not measured where samples were dropped, because the lag clock assumes none were.

Spot availability that evening: a second machine was requested at 22:29 UTC in us-east-1b and had not
started 25 minutes later, when the request was cancelled. `aws ec2 get-spot-placement-scores` for one
`g4dn.xlarge` gave 1 out of 10 in every us-east-1 zone. The account's on-demand limit for G instances is 0.

### Second machine, 2026-11-07 02:49 UTC (scored)

Same instance type, zone and image as above (us-east-1a, instance i-04b9c40bfaeff4e26), got by retrying
the zones every 8 minutes; us-east-1b never had capacity, the other zones did. Each run folder was copied
off the machine right after its run, so these rows are scored against batch (`eval/score_engine.py`, rows
of 2026-11-06 21:51 to 21:55 local time in the table above). The machine was destroyed at the end.

| settings | pass p50 ms | pass p99 ms | lag p50 ms | lag p99 ms | RTF | overruns | notes | GPU memory MB | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `--stride 512 --fast` (full song) | 453.1 | 564.1 | not measured | not measured | 0.058 | 0 | 1495 | 6153 | 1.0000 | 1.0000 | 1.0000 |
| `--stride 16 --delay 16` | 102.0 | 106.8 | 1052 | 1173 | 0.437 | 0 | 769 | 3021 | 0.9155 | 0.9312 | 0.9233 |
| `--stride 8` | 97.2 | 101.9 | 728 | 788 | 0.833 | 0 | 727 | 2863 | 0.9188 | 0.8836 | 0.9009 |
| `--stride 16 --delay 16`, two engines at once (A) | 148.0 | 204.8 | 1098 | 1219 | 0.656 | 0 | 769 | 6039 (both) | 0.9155 | 0.9312 | 0.9233 |
| `--stride 16 --delay 16`, two engines at once (B) | 156.1 | 203.8 | 1106 | 1227 | 0.687 | 0 | 769 | 6039 (both) | 0.9155 | 0.9312 | 0.9233 |

The stride 512 run also passed the array check (max diff 2.69e-04 vs batch, velocity mismatch 0%) and
the decoder check; every row passed the decoder check. F1, P, R and note counts are identical to the same
settings on the RTX 4070 and in Docker on WSL 2, so the T4 changes speed only, not output. GPU memory is
the peak `nvidia-smi` reading over the whole run (one sampler per run, so the two-engine value is both together).

## Cloud deploy, 2026-11-07

One spot `g4dn.xlarge` (T4) made by `infra/deploy/` (RUN.md step 11): the service image (engine +
`ui/server.py`, one engine per connected page, two at once) behind Caddy with HTTPS (Let's Encrypt
for `54-174-0-35.sslip.io`, a fixed address) and a password.

First machine, one-time spot in us-east-1a: apply 09:35 UTC (ready in 13 s), image built on the
box and started 09:36 to 09:38, certificate obtained 09:38, page 401 without and 200 with the
password. AWS reclaimed it at 16:56 UTC (`instance-terminated-no-capacity`), after 7 h 20 min.
Second machine, persistent spot (stop on interruption) in us-east-1f (us-east-1d's default subnet
turned out to be private): ready 19:42 UTC, deployed 19:46 UTC, same address and hostname, the
certificate issued again. Spot price in us-east-1f at the time: $0.2555/h. Cost of both machines,
estimated from the price and not read from the bill: about $1.90 for the first, $0.26/h for the
second while it runs, plus the fixed address.

Measured from this PC against the second machine, over the WebSocket through Caddy, with
`cloud_session.py` and `cloud_mic.py` (scratch scripts that do what the page does; the engine on
the box ran `--stride 16 --delay 16`, two sessions allowed):

| path | notes | lag |
|---|---|---|
| song on the box (the engine reads the file there), 30 s | 300 | engine lag median 1.059 s, p99 1.177 s; event delivery jitter to this PC: median 0.000 s, max 0.013 s (measured against the engine's clock, so the constant network delay is not in it) |
| microphone path: this PC streams the 60 s clip as 20 ms float32 frames into the engine's stdin, 30 s | 420 | **end to end (sample leaves this PC to its note back here) median 1.082 s and 1.091 s in two runs**, first notes 1.17 to 1.19 s; p99 2.91 s, but every note over 1.5 s has an onset past 29.2 s, i.e. the flush when the stream is stopped, not steady state |

So the cloud round trip from this PC to us-east-1 adds under 0.1 s to the engine's own lag (1.00 s
for the same audio piped into stdin on the RTX 4070 here, 1.06 s on the T4). The 420 notes against
300 for the song path are the automatic input level, which is on for streamed audio and off for files.

Measured on this PC before the deploy, same page and server, engine on the RTX 4070: a simulated
microphone (the 60 s clip played through the Web Audio API into `getUserMedia`) streamed by the page's
AudioWorklet into `pianofi_live --input stdin`: 127 notes in 15 s, end-to-end lag median **1.10 s**
on the page. Two pages with different songs ran at once; a third was refused with "All 2 engine slots
are in use". The real microphone and tab-sharing dialogs were not exercised.

Front machine added 21:20 UTC: a `t4g.nano` (on-demand, $0.0042/h) took over the fixed address and
HTTPS, and proxies to the GPU box over the VPC (RUN.md step 11). Through it: 401 without and 200 with
the password, a 20 s session gave 135 notes at engine lag median 1.067 s, p99 1.177 s, event jitter
max 0.015 s, the same as directly. With the service container stopped on the GPU box, the site
answered with the offline page (HTTP 502 with the page as body, 994 bytes) within the proxy's 3 s
dial timeout, and the app was back 3 s after the container was started. The GPU box itself is no
longer reachable from the internet on its service ports (connection timed out from this PC).

### Two logins, Start button, idle stop, 2026-11-08 01:14 to 02:30 UTC

Added `infra/deploy/iam.tf` (a role for the front that may start the one GPU instance),
`deploy/front/start.py` (the Start button's handler), a guest login, `--admin` in `ui/server.py`
and `deploy/idle_stop.sh` (RUN.md step 11). `terraform apply`: 3 IAM resources in 7 s, the front
updated in place in 60 s. Checked from this PC through the front (`site_check.py`, `ws_mode.ps1`,
`start_loop.py`, scratch scripts; passwords read from files):

| check | result |
|---|---|
| page without login / as `sam` / as `guest` | 401 / 200 / 200 |
| `GET /start` as `sam` / `guest` | `admin: true, left: null` / `admin: false, left: 3` |
| `POST /start` while the box runs | `started: false`, guest's count unchanged |
| upload as `guest`, also with a forged `X-User: sam` header | 403 both (Caddy overwrites the header); as `sam`: 200 |
| first WebSocket message as `guest` (forged header sent) | `"user": "guest", "can_upload": false`; as `sam`: `true` |
| guest at 3 starts (counter set by hand) | `POST /start` 429 "you have used today's 3 starts"; admin 200 at the same moment |
| box stopped by hand, Start clicked as `guest` | `started: true, left: 2`; the app answered 19 s after the click (box boot + container restart); the box's public address changed, the private one did not |
| the same, clicked 2 s after the instance read "stopped" | AWS refused: `IncorrectSpotRequestState` (the spot request turned `disabled` 4 s after the instance showed stopped); the handler now retries for up to 30 s |
| stopping the box by hand (`aws ec2 stop-instances` to state `stopped`) | 413 s the one time it was timed (AWS's shutdown, the first stop was not timed) |
| the front after Terraform's stop/start for the new first-boot script | modified in 43 s, site answering (401 without login) 1 s after `apply` returned, certificate kept |
| idle stop (30 min with no engine) | the box started at 01:52:19 UTC (Start click), nobody played, `aws ec2 describe-instances` showed it `stopping` at 02:26:04 UTC (a once-a-minute poll): 33 min 45 s after the start, i.e. 30 idle minutes counted from the first timer tick plus boot and shutdown time |

## GPU memory per engine, 2026-11-12 13:04

Measured with the graphs as they were exported until 16:00 that day, where velocity is an ArgMax node (see "Velocity
ArgMax moved to the GPU" below for the current graphs). To reproduce: `.venv\Scripts\python.exe py\export_onnx.py --wav <song> --argmax --out-dir models\argmax`,
`.venv\Scripts\python.exe py\export_fp16.py --dir models\argmax`, and add `--models models\argmax` to the engine commands.
Their run folders were reused afterwards, so the engine table above holds the newer rows.

Command: `python eval\gpu_memory.py test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

686610__seth_makes_sounds__bubbglegum-pop-song.wav, file input, `--fast`, one engine at a time. GPU memory is the peak rise in `nvidia-smi` memory.used during the run, sampled every 0.5 s, so short spikes can be missed.

| settings | GPU memory MB | pass p50 ms | pass max ms |
|---|---|---|---|
| stride 256 ms + delay 256 ms, cached | 3065 | 53.7 | 91 |
| stride 256 ms + delay 256 ms, cached, FP16 | 2455 | 40.5 | 63 |
| stride 64 ms, cached | 2875 | 50.4 | 75 |
| stride 256 ms, no cache | 5980 | 163.8 | 204 |
| stride 8192 ms (batch windows) | 6181 | 175.1 | 254 |

## FP16 (half precision) on the CUDA provider, 2026-11-12

Measured with the graphs as they were exported until 16:00 that day, where velocity is an ArgMax node (see "Velocity
ArgMax moved to the GPU" below for the current graphs). To reproduce: `.venv\Scripts\python.exe py\export_onnx.py --wav <song> --argmax --out-dir models\argmax`,
`.venv\Scripts\python.exe py\export_fp16.py --dir models\argmax`, and add `--models models\argmax` to the engine commands.
Their run folders were reused afterwards, so the engine table above holds the newer rows.

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10. ONNX Runtime 1.20.1, CUDA provider.

`py\export_fp16.py` writes `models/frame_fp16.onnx` and `models/time_fp16.onnx`; the engine loads them with `--fp16`.
FP32 runs were repeated the same day. Scored rows are also in the engine table above (2026-11-12). Song:
`686610__seth_makes_sounds__bubbglegum-pop-song.wav` for batch windows, its first 60 s for the rest.
Commands are `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\<song> ...` with the options shown; FP16 adds `--fp16`.

| setting | precision | F1 vs batch | notes (batch) | frame model p50 ms | time model p50 ms | pass p50 ms | pass p99 ms | options, `--out` |
|---|---|---|---|---|---|---|---|---|
| stride 8192 ms (batch windows), `--fast` | FP32 | 1.0000 | 1495 (1495) | 141.9 | 36.4 | 181.4 | 246.1 | `--stride 512 --fast --dump --out runs\check512` |
| same | FP16 | 0.9980 | 1499 (1495) | 73.9 | 24.9 | 100.2 | 160.2 | `... --fp16 --out runs\fp16_check512` |
| stride 256 ms + delay 256 ms, `--fast` | FP32 | not measured | 769 | 16.2 | 38.2 | 54.3 | 70.1 | `--stride 16 --delay 16 --fast --out runs\fp32_s16_d16_fast` |
| same | FP16 | not measured | 771 | 11.0 | 28.7 | 43.4 | 54.8 | `... --fp16 --out runs\fp16_s16_d16_fast` |
| stride 256 ms + delay 256 ms, real time | FP32 | 0.9233 | 769 (756) | 12.7 | 40.4 | 62.4 | 83.3 | `--stride 16 --delay 16 --dump --out runs\fp32_s16_d16_rt` |
| same | FP16 | 0.9221 | 771 (756) | 27.8 | 31.9 | 58.3 | 68.2 | `... --fp16 --out runs\fp16_s16_d16_rt` |
| same, second run | FP32 | not measured | 769 | 12.9 | 37.8 | 57.1 | 71.0 | `--stride 16 --delay 16 --out runs\fp32_s16_d16_rt2` |
| same, second run | FP16 | not measured | 771 | 26.7 | 31.8 | 56.9 | 69.3 | `... --fp16 --out runs\fp16_s16_d16_rt2` |

GPU memory (`eval\gpu_memory.py`, the table above this section, whole song, `--fast`): 3065 MB FP32, 2455 MB FP16 at
stride 256 ms + delay 256 ms; pass p50 there was 53.7 ms FP32 and 40.5 ms FP16. Real-time frame lag p50: 1012 ms FP32, 1005 ms FP16.

What this shows, on this GPU and provider:

- Accuracy holds. At batch windows FP16 gives 1499 notes for batch's 1495 (F1 0.9980), and at the live setting F1
  moves from 0.9233 to 0.9221. The array check fails with FP16 as expected (max diff 3.22e-02 against a 1e-3 limit),
  so FP32 stays the reference and the exactness check stays on FP32.
- Run flat out (`--fast`), FP16 is faster: the time model takes 28.7 ms instead of 38.2, a live pass 43.4 ms instead of
  54.3 (20% less), and a batch-window pass 100.2 ms instead of 181.4 (45% less). GPU memory is 20% lower.
- In real time the pass did not get faster (58.3 vs 62.4 ms, then 56.9 vs 57.1). The time model keeps its gain
  (about 32 ms vs 38 to 40), but the frame model takes about 27 ms in FP16 against about 13 in FP32, in both runs.
  It only happens when the GPU sits idle between passes; run flat out the FP16 frame model is the faster one
  (11.0 vs 16.2 ms). The cause is not measured yet.

### Why two layers stay in FP32

With every node in FP16 all outputs were NaN (0 notes). `py\attention_range.py runs\check512 runs\prettifun-s16-d16`
(FP32 on the CPU, 36 windows of two songs) gives the largest attention score, query x key before the divide by 8, per layer:

| layer | largest score | share of the FP16 limit (65504) |
|---|---|---|
| `/layers_freq.0/self_attention` (frame model, first encoder layer) | 394824 | 602.7% |
| `/layers_freq.1/self_attention` | 118 | 0.2% |
| `/layers_freq.2/self_attention` | 100 | 0.2% |
| `/layer_zero_freq/encoder_attention` | 68 | 0.1% |
| `/layers_freq.0/self_attention_1` | 67 | 0.1% |
| `/layers_freq.0/encoder_attention` | 75 | 0.1% |
| `/layers_freq.1/self_attention_1` | 95 | 0.1% |
| `/layers_freq.1/encoder_attention` | 159 | 0.2% |
| `/layers_time.0/self_attention` (time model, first layer) | 1848219 | 2821.5% |
| `/layers_time.1/self_attention` | 243 | 0.4% |
| `/layers_time.2/self_attention` | 256 | 0.4% |

Two layers overflow, both the first of their stack. Dividing before the product instead of after would not save
the time model's layer (1848219 / 8 is still 3.5 times the limit), so `export_fp16.py` keeps the score MatMul, Div and
Softmax of those two layers in FP32 and everything else in FP16.

A first version kept the scores of all 11 layers in FP32 (`py\export_fp16.py --all-layers` still makes it). It was
slower and no smaller: time model 34.5 to 35.7 ms, `--fast` live pass 50.5 ms, batch-window pass 125.4 ms, 3041 MB,
F1 0.9990 at batch windows and 0.9227 at the live setting.

The T4 and more than one engine per GPU: see "Tesla T4 with the TopK graphs" below.

## Profile of a pass (ONNX Runtime per-node profile), 2026-11-12

Measured with the graphs as they were exported until 16:00 that day, where velocity is an ArgMax node (see "Velocity
ArgMax moved to the GPU" below for the current graphs). To reproduce: `.venv\Scripts\python.exe py\export_onnx.py --wav <song> --argmax --out-dir models\argmax`,
`.venv\Scripts\python.exe py\export_fp16.py --dir models\argmax`, and add `--models models\argmax` to the engine commands.
Their run folders were reused afterwards, so the engine table above holds the newer rows.

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF (20 threads), NVIDIA GeForce RTX 4070, Windows 10. ONNX Runtime 1.20.1, CUDA provider.

Question: why does the FP16 frame model take about 27 ms in real time when FP32 takes 13 and FP16 run flat out takes 11?

Runs: `cpp\build\Release\pianofi_live.exe --wav test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16 --profile --out runs\prof_<precision>_<rt or fast>`
(plus `--fp16` and `--fast` as named), read with `.venv\Scripts\python.exe eval\ort_profile.py runs\prof_fp32_rt runs\prof_fp16_rt runs\prof_fp32_fast runs\prof_fp16_fast`.
Profiling itself slows a pass (real-time pass p50 76.5 ms FP32 and 68.1 ms FP16 with it on), so these show where the time goes, not how much.

Frame graph, median over 290 runs of the graph:

| run | median run ms | all CUDA nodes ms | all CPU nodes ms | of which `/ArgMax` ms (p10 / p50 / p90) |
|---|---|---|---|---|
| FP32, real time | 30.7 | 8.35 (260 nodes) | 19.39 (40 nodes) | 0.08 / 18.65 / 26.98 |
| FP16, real time | 31.1 | 8.32 (263 nodes) | 19.25 (41 nodes) | 0.08 / 18.41 / 26.92 |
| FP32, `--fast` | 19.0 | 4.95 | 9.85 | 0.08 / 9.35 / 17.47 |
| FP16, `--fast` | 17.8 | 8.25 | 6.11 | 0.07 / 5.25 / 14.97 |

What it shows:

- One node is most of the frame model's time: `/ArgMax`, which picks the velocity for the per-frame heads. ONNX Runtime
  runs it on the CPU, not the GPU (so does the same node in the time graph), and splits its 16 x 88 x 128 input over
  the CPU thread pool. The work itself takes 0.05 to 0.08 ms when it goes well; the median is 5 to 19 ms. Every other
  node has a median under 0.2 ms.
- It is not an FP16 problem. With profiling on, FP32 and FP16 are equally slow there. FP16 only landed on the slow
  side more often in the unprofiled runs.
- The likely cause is the thread pool, not the arithmetic: the time goes into handing 13 small blocks to pool
  threads and waiting for them, and it is worst in real time, when the threads have gone to sleep between passes.
  The wake-up itself was not measured; the test below is the evidence.

Test: the same real-time runs with one ONNX Runtime thread (`--threads 1`, no profiling), so `/ArgMax` runs on the calling thread:

| run | frame model p50 / p99 ms | time model p50 ms | pass p50 / p99 ms | notes | options, `--out` |
|---|---|---|---|---|---|
| FP32, default threads | 12.9 / 31.3 | 37.8 | 57.1 / 71.0 | 769 | `--stride 16 --delay 16 --out runs\fp32_s16_d16_rt2` |
| FP32, `--threads 1` | 5.1 / 5.6 | 43.5 | 48.7 / 51.2 | 769 | `... --threads 1 --out runs\fp32_s16_d16_rt_t1` |
| FP16, default threads | 26.7 / 31.2 | 31.8 | 56.9 / 69.3 | 771 | `... --fp16 --out runs\fp16_s16_d16_rt2` |
| FP16, `--threads 1` | 4.6 / 5.4 | 30.9 | 35.6 / 40.4 | 771 | `... --fp16 --threads 1 --out runs\fp16_s16_d16_rt_t1` |

With one thread the frame model takes about 5 ms with almost no spread, the note counts are unchanged, and the
real-time pass drops from 57.1 to 48.7 ms in FP32 and from 56.9 to 35.6 ms in FP16. The FP32 time model is slower
with one thread (43.5 vs 37.8 ms): its own ArgMax is 32 times bigger (88 x 512 x 128) and does use the threads.
F1 for the `--threads 1` runs: not measured. The engine's default is unchanged (ONNX Runtime's default thread count).

## GPU memory per engine, 2026-11-12 16:10

Command: `python eval\gpu_memory.py test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song.wav`

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10

686610__seth_makes_sounds__bubbglegum-pop-song.wav, file input, `--fast`, one engine at a time. GPU memory is the peak rise in `nvidia-smi` memory.used during the run, sampled every 0.5 s, so short spikes can be missed.

| settings | GPU memory MB | pass p50 ms | pass max ms |
|---|---|---|---|
| stride 256 ms + delay 256 ms, cached | 3048 | 37.6 | 54 |
| stride 256 ms + delay 256 ms, cached, FP16 | 2455 | 22.7 | 35 |
| stride 64 ms, cached | 3077 | 35.6 | 48 |
| stride 256 ms, no cache | 6145 | 164.0 | 190 |
| stride 8192 ms (batch windows) | 6183 | 174.6 | 243 |

## Velocity ArgMax moved to the GPU (TopK), 2026-11-12 16:00

Machine: 13th Gen Intel(R) Core(TM) i5-13600KF, NVIDIA GeForce RTX 4070, Windows 10. ONNX Runtime 1.20.1, CUDA provider.

The profile above found that ONNX Runtime ran the velocity ArgMax of both graphs on the CPU, where it waited on the
CPU thread pool. `py\export_onnx.py` now writes that pick as top-1 (`torch.topk`, ONNX TopK), which the CUDA provider
runs on the GPU. Same result by definition (lowest index among equal values), and every check still passes:
`export_onnx.py` All checks passed (velocity mismatch 0.0000%), `check_mel.py` PASSED, `pf_tests` ALL TESTS PASSED, and
at stride 512 array check PASSED (max diff 2.88e-04, the same as before), decoder check PASSED, F1 = 1.0000.
A profile of the new graphs (`--profile --out runs\prof_topk_fp32_rt`) shows no CPU node left in the time graph and
0.46 ms of CPU nodes (shape arithmetic) in the frame graph.

Real time, stride 256 ms + delay 256 ms, first 60 s of `686610__seth_makes_sounds__bubbglegum-pop-song.wav`. "Before" is the
ArgMax graphs run again in the same session (`--models models\argmax`). Commands: `cpp\build\Release\pianofi_live.exe --wav
test_audio\16k\686610__seth_makes_sounds__bubbglegum-pop-song_first60s.wav --stride 16 --delay 16` plus the options shown.

| graphs | precision | F1 vs batch | notes (batch) | frame model p50 / p99 ms | time model p50 ms | pass p50 / p99 ms | frame lag p50 / p99 ms | GPU memory MB | options, `--out` |
|---|---|---|---|---|---|---|---|---|---|
| ArgMax (before) | FP32 | not measured (0.9233 earlier that day) | 769 | not recorded here | not recorded here | 63.2 / 75.0 | not recorded here | 3065 | `--models models\argmax --out runs\argmax_fp32_s16_d16_rt` |
| ArgMax (before) | FP16 | not measured (0.9221 earlier that day) | 771 | not recorded here | not recorded here | 58.1 / 70.3 | not recorded here | 2455 | `--models models\argmax --fp16 --out runs\argmax_fp16_s16_d16_rt` |
| TopK | FP32 | 0.9233 | 769 (756) | 4.7 / 5.6 | 36.2 | 41.1 / 45.3 | 992 / 1114 | 3048 | `--dump --out runs\fp32_s16_d16_rt` |
| TopK, second run | FP32 | not measured | 769 | 4.8 / 5.4 | 36.4 | 41.2 / 45.8 | 993 / 1114 | | `--out runs\fp32_s16_d16_rt2` |
| TopK | FP16 | 0.9221 | 771 (756) | 4.3 / 4.8 | 22.7 | 26.9 / 30.0 | 979 / 1100 | 2455 | `--fp16 --dump --out runs\fp16_s16_d16_rt` |
| TopK, second run | FP16 | not measured | 771 | 4.4 / 5.0 | 22.7 | 27.0 / 30.2 | 979 / 1100 | | `--fp16 --out runs\fp16_s16_d16_rt2` |

"Not recorded here": the numbers are in each run's `metrics.json`. GPU memory is from `eval\gpu_memory.py` (whole song, `--fast`):
the 13:04 table for the ArgMax graphs, the 16:10 table for TopK.

Other settings on the TopK graphs:

| setting | precision | F1 vs batch | frame model p50 ms | time model p50 ms | pass p50 / p99 ms | options, `--out` |
|---|---|---|---|---|---|---|
| stride 8192 ms (batch windows), whole song, `--fast` | FP32 | 1.0000 (1495 notes, batch 1495) | 134.8 | 33.9 | 169.8 / 239.9 | `--stride 512 --fast --dump --out runs\check512` |
| same | FP16 | 0.9980 (1499 notes) | 71.2 | 20.8 | 92.9 / 155.2 | `--stride 512 --fast --dump --fp16 --out runs\fp16_check512` |
| stride 256 ms + delay 256 ms, `--fast` | FP32 | not measured | 3.9 | 34.2 | 38.1 / 40.7 | `--stride 16 --delay 16 --fast --out runs\fp32_s16_d16_fast` |
| same | FP16 | not measured | 2.5 | 20.7 | 23.2 / 25.5 | `--stride 16 --delay 16 --fast --fp16 --out runs\fp16_s16_d16_fast` |

What this shows:

- Moving one node took the real-time pass from 63.2 to 41.1 ms in FP32 (35% less) with identical notes, and the p99
  from 75.0 to 45.3 ms. Nothing about the model's arithmetic changed; the exactness check gives the same max diff.
- FP16 now pays off in real time too: 26.9 ms a pass against 41.1 for FP32 (35% less), for F1 0.9221 against 0.9233.
  Its time model takes 22.7 ms instead of 36.2. With the ArgMax graphs its time model was about 32 ms: there the
  velocity scores were also copied to the CPU and converted to FP32 before the ArgMax, which is gone now.
- Both together: 63.2 ms to 26.9 ms a pass (57% less) and 3065 MB to 2455 MB (20% less), at a cost of 0.0012 F1.
- Lag moved by about 30 ms (frame lag p50 1012 ms in the morning's FP32 run, 992 ms now, 979 ms with FP16), because
  lag is set by the model's lookahead, the stride and the delay, not by compute. What the faster pass buys is
  headroom: more engines per GPU or a shorter stride. Neither is measured yet.

The T4 and more than one engine per GPU: see the next section. Not measured: CPU-only runs with the new graphs.

## Tesla T4 with the TopK graphs, FP32 and FP16, one to four engines, 2026-11-13 01:10 UTC

Machine: AWS `g4dn.xlarge` spot in us-east-1f (Tesla T4 15 GB, Xeon 8259CL, 4 vCPUs), Amazon Linux 2023 ECS GPU image,
engine in Docker (`docker build -t pianofi-live .`, `pf_tests` ALL TESTS PASSED in the build). ONNX Runtime 1.20.1,
CUDA provider. Graphs: the TopK export of 2026-11-12 (`models/frame.onnx`, `time.onnx` and the `_fp16` pair).
Every run is a row in the engine table above (`--out runs/t4b_*`) with its full `docker run` command. Each engine
was its own container; the engines of a group were started together. GPU memory is the peak of
`nvidia-smi --query-gpu=memory.used` sampled every 500 ms over the group, all engines together.

Stride 8192 ms (batch windows), whole song, `--fast`:

| precision | F1 vs batch | notes (batch) | array check | decoder check | pass p50 / p99 ms | GPU memory MB | `--out` |
|---|---|---|---|---|---|---|---|
| FP32 | 1.0000 | 1495 (1495) | PASSED, max diff 2.69e-04 | PASSED | 443.1 / 551.5 | 6153 | `runs/t4b_check512` |
| FP16 | 0.9970 | 1496 (1495) | FAILED, max diff 3.29e-02 (expected: half precision is not bit-close) | PASSED | 202.2 / 296.3 | 5447 | `runs/t4b_check512_fp16` |

Real time, stride 256 ms + delay 256 ms, first 60 s of `686610__seth_makes_sounds__bubbglegum-pop-song.wav`
(`--stride 16 --delay 16`, plus `--fp16` for the FP16 rows). With several engines the columns give the range over
the engines (lowest to highest p50, highest p99).

| engines | precision | F1 vs batch | pass p50 ms | pass p99 ms | frame lag p50 ms | frame lag p99 ms | overruns | GPU memory MB, all engines | `--out` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | FP32 | 0.9233 | 96.6 | 102.9 | 1047 | 1167 | 0 | 3021 | `runs/t4b_s16_d16` |
| 1 | FP16 | 0.9227 | 51.1 | 52.8 | 1001 | 1121 | 0 | 2423 | `runs/t4b_s16_d16_fp16` |
| 2 | FP32 | 0.9233 both | 98.0 to 99.0 | 209.2 | 1048 to 1049 | 1170 | 0 | 6039 | `runs/t4b_x2_1`, `_2` |
| 2 | FP16 | 0.9227 both | 101.4 to 101.5 | 107.5 | 1052 | 1172 | 0 | 4843 | `runs/t4b_x2_fp16_1`, `_2` |
| 3 | FP32 | 0.1702 to 0.1818 | 313.2 to 313.6 | 322.8 | 6813 to 7176 | 12983 | 146464 to 151872 samples | 9057 | `runs/t4b_x3_1` to `_3` |
| 3 | FP16 | 0.9227 all | 127.4 to 138.9 | 155.9 | 1078 to 1090 | 1211 | 0 | 7263 | `runs/t4b_x3_fp16_1` to `_3` |
| 4 | FP32 | 0.1051 to 0.1359 | 418.6 to 420.2 | 437.9 | 13208 to 13701 | 25831 | 338048 to 344704 samples | 12075 | `runs/t4b_x4_1` to `_4` |
| 4 | FP16 | 0.9227 all | 96.3 to 134.5 | 199.9 | 1056 to 1091 | 1214 | 0 | 9683 | `runs/t4b_x4_fp16_1` to `_4` |

- FP32 output on the T4 is unchanged by TopK: F1 1.0000 at stride 512 and 0.9233 at the default, the same as the
  2026-11-06 T4 runs with the ArgMax graphs.
- One FP32 engine: 96.6 ms a pass against 102.0 ms on 2026-11-06 (ArgMax graphs). The T4 gains far less from TopK than
  the RTX 4070 PC did (63.2 to 41.1 ms): here the GPU itself is the slow part.
- Two FP32 engines: 98.0 and 99.0 ms p50 against 148.0 and 156.1 ms on 2026-11-06, but the p99 is still about 209 ms
  (204.8 and 203.8 then), so passes still collide at times.
- FP16 halves the pass on the T4: 51.1 ms against 96.6, for F1 0.9227 against 0.9233 (770 notes against 769).
- FP32 does not hold three engines on a T4: a pass takes 313 ms against a 256 ms stride, the engines drop audio
  and F1 falls to 0.17 to 0.18. Two is its limit, as before.
- FP16 holds four: every engine has 0 overruns, F1 0.9227, frame lag p50 1056 to 1091 ms and p99 at most 1214 ms,
  in 9683 MB of the T4's 15 GB.
- The pass time of an engine that shares the GPU depends on whether its passes land at the same moment as another
  engine's (the two FP16 engines ran at 101 ms each, twice the single-engine 51.1 ms, while two of the four
  ran at 96 and 102 ms), so read these as one sample each.

Through the deployed site, 2026-11-13 01:27 UTC: the service now runs `ui/server.py --max-sessions 4 --fp16`
(`deploy/run.sh`). Four sessions opened at once from this PC over the WebSocket through Caddy, each playing the 60 s
clip on the box for 70 s (`cloud_session.py`, the scratch script from the 2026-11-07 cloud deploy section, four
copies started together): every session got 845 notes, engine lag median 1.110 to 1.142 s and p99 1.230 to 1.262 s,
event delivery jitter at most 0.016 s. The box logged `provider CUDA FP16` for all four and `nvidia-smi` showed
9683 MiB while they ran. Not scored against batch (the engine does not dump arrays in a session).

A first try a few minutes earlier got 0 notes in all four sessions: the engines logged `CUDA unavailable ... using
CPU`, and `nvidia-smi` inside the service container said `Failed to initialize NVML: Unknown Error` while the host
saw the T4. Cause: `deploy/run.sh` started the container and then ran `systemctl daemon-reload` for the idle-stop
timer, which takes the GPU away from containers that are already running. A reboot hid it, because the container
then starts after systemd is set up. `run.sh` now starts the container last.

Not measured: five or more engines, other strides, F1 of sessions through the site.
