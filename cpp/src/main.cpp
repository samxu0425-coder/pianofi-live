// pianofi_live: run the streaming engine on a 16 kHz wav and report metrics.
//
//   pianofi_live --wav test_audio\song_16k.wav [options]
//   pianofi_live --input mic|loopback [options]        Ctrl+C stops
//   pianofi_live --input stdin [options]               16 kHz mono float32 piped in; ends with the stream
//
//   pianofi_live --list-devices                        names for --device
//
//   --input KIND      file (default), mic, loopback (what the PC is playing), or stdin
//                     (raw 16 kHz mono float32 samples, what ui/server.py sends from the browser)
//   --device NAME     live input: part of a device name (default: the Windows default device)
//   --models DIR     folder with frame.onnx, time.onnx, sv_<style>.bin (default: models)
//   --style NAME      style vector file sv_NAME.bin (default: level2)
//   --stride N        output frames per pass, 16 ms each (default: 32 = 0.512 s)
//   --delay N         emit output N frames back from the newest; adds N*16 ms lag (default: 0)
//   --heads a|b       b = time model, same as batch (default); a = per-frame heads only
//   --no-cache        recompute all 512 frames every pass (the baseline to beat)
//   --fast            don't pace audio in real time (accuracy runs; no lag numbers)
//   --events          print one JSON line per note event to stdout; logs go to stderr
//   --normalize       automatic input level: scale the audio so its loudest recent peak is
//   --no-normalize    full scale. On by default for mic and loopback, off for wav files.
//   --cpu             use the CPU even if CUDA is available (default: try CUDA)
//   --fp16            half precision: load frame_fp16.onnx and time_fp16.onnx (py/export_fp16.py).
//                     GPU only, and not exact: outputs move away from batch
//   --trt             TensorRT provider in front of CUDA (TensorRT 10 DLLs on PATH). Takes the FP32
//                     graphs; with --fp16 TensorRT picks FP16 layers itself. The first run builds
//                     the engines (minutes) and keeps them in <models>/trt_cache
//   --mixed           with --trt: give TensorRT the mixed-precision graphs of py/export_fp16.py
//                     (two attention layers stay in FP32) instead of the FP32 graphs
//   --threads N       ONNX Runtime intra-op threads
//   --out DIR         where notes.csv and metrics.json go (default: runs/<timestamp>)
//   --dump            also write raw model outputs + mel to --out for the Python checks
//   --profile         also write ONNX Runtime's per-node timing profile of both graphs to --out
//                     (ort_frame_*.json, ort_time_*.json; read with eval/ort_profile.py). Slows the run a little
//   --max-seconds X   only use the first X seconds of the wav; live input stops after X
//   --start-seconds X skip the first X seconds of the wav (resume a stopped song; times count from X)
//                     (default: live input runs until Ctrl+C)
//   --bounded-memory  drop history that is no longer needed, as live input always does (for tests)
//   --max-engines N   refuse to start if N engines already run on this machine (default: 1,
//                     so two never share the GPU by accident; 0 = no limit)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <psapi.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#else
#include <sys/resource.h>
#endif

#include "pf/capture.h"
#include "pf/engine.h"
#include "pf/engine_lock.h"
#include "pf/wav.h"

namespace fs = std::filesystem;

namespace {

struct Stats {
    double p50 = 0, p95 = 0, p99 = 0, max = 0, mean = 0;
    size_t n = 0;
};

Stats stats(std::vector<double> v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) { return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))]; };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    s.max = v.back();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
    return s;
}

std::string json_stats(const char* name, const std::vector<double>& v) {
    Stats s = stats(v);
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "  \"%s\": {\"n\": %zu, \"mean\": %.3f, \"p50\": %.3f, \"p95\": %.3f, \"p99\": %.3f, \"max\": %.3f}",
                  name, s.n, s.mean, s.p50, s.p95, s.p99, s.max);
    return buf;
}

std::vector<float> read_floats(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (run py/export_onnx.py first)");
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float)));
    if (size_t(f.gcount()) != n * sizeof(float)) throw std::runtime_error(path + " is too short");
    return v;
}

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

// The command line as one string, so results.md can record how a run was made.
std::string command_line(int argc, char** argv) {
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i > 0) cmd += ' ';
        cmd += arg.find(' ') == std::string::npos ? arg : '"' + arg + '"';
    }
    return cmd;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

std::atomic<bool> g_stop{false};
void on_ctrl_c(int) { g_stop.store(true); }

// The most memory this process has held at once, in MB. 0 where it isn't measured.
double peak_memory_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters))
        return double(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
#else
    rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) return double(usage.ru_maxrss) / 1024.0;  // Linux reports KB
#endif
    return 0.0;
}

int list_devices() {
    try {
        std::printf("Microphones (--input mic):\n");
        for (const auto& name : pf::Capture::device_names(false)) std::printf("  %s\n", name.c_str());
        std::printf("Outputs (--input loopback):\n");
        for (const auto& name : pf::Capture::device_names(true)) std::printf("  %s\n", name.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "usage: pianofi_live (--wav FILE | --input mic|loopback|stdin [--device NAME] | --list-devices) [--models DIR] [--style NAME] [--stride N] [--delay N] [--heads a|b]\n"
                 "                    [--no-cache] [--fast] [--events] [--normalize | --no-normalize] [--bounded-memory]\n"
                 "                    [--cpu] [--fp16] [--threads N] [--out DIR] [--dump] [--profile] [--max-seconds X] [--start-seconds X] [--max-engines N]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    timeBeginPeriod(1);  // 1 ms sleep granularity instead of ~15 ms
#endif
    std::string wav_path, input_name = "file", models = "models", style = "level2", out_dir;
    pf::EngineConfig cfg;
    pf::ModelOptions mopt;
    bool try_cuda = true, dump = false, fp16 = false, profile = false, trt = false, mixed = false;
    int normalize = -1;  // -1 = not given: on for live input, off for files
    double max_seconds = 0, start_seconds = 0;
    int max_engines = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        try {
            if (a == "--wav") wav_path = next();
            else if (a == "--input") input_name = next();
            else if (a == "--device") cfg.device = next();
            else if (a == "--list-devices") return list_devices();
            else if (a == "--models") models = next();
            else if (a == "--style") style = next();
            else if (a == "--stride") cfg.stride = std::stoi(next());
            else if (a == "--delay") cfg.delay = std::stoi(next());
            else if (a == "--heads") cfg.heads = next().at(0);
            else if (a == "--no-cache") cfg.cache = false;
            else if (a == "--fast") cfg.realtime = false;
            else if (a == "--events") cfg.events = true;
            else if (a == "--normalize") normalize = 1;
            else if (a == "--no-normalize") normalize = 0;
            else if (a == "--bounded-memory") cfg.bounded_memory = true;
            else if (a == "--cpu") try_cuda = false;
            else if (a == "--fp16") fp16 = true;
            else if (a == "--trt") trt = true;
            else if (a == "--mixed") mixed = true;
            else if (a == "--threads") mopt.threads = std::stoi(next());
            else if (a == "--out") out_dir = next();
            else if (a == "--dump") dump = true;
            else if (a == "--profile") profile = true;
            else if (a == "--max-seconds") max_seconds = std::stod(next());
            else if (a == "--start-seconds") start_seconds = std::stod(next());
            else if (a == "--max-engines") max_engines = std::stoi(next());
            else return usage();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return usage();
        }
    }
    if (input_name == "mic") cfg.input = pf::Input::Mic;
    else if (input_name == "loopback") cfg.input = pf::Input::Loopback;
    else if (input_name == "stdin") cfg.input = pf::Input::Stdin;
    else if (input_name != "file") return usage();
    const bool live = cfg.input != pf::Input::File;
    if (live != wav_path.empty()) return usage();  // exactly one of a wav file or a live input
    cfg.normalize = normalize == -1 ? live : normalize == 1;
    if (live && max_seconds > 0) cfg.max_seconds = max_seconds;
#ifdef _WIN32
    if (cfg.input == pf::Input::Stdin) _setmode(_fileno(stdin), _O_BINARY);  // raw samples, no newline translation
#endif
    // Ctrl+C (and Ctrl+Break, what ui/server.py sends on Windows) ends any input cleanly: the
    // engine stops reading, finishes the notes it has, and writes notes.csv and metrics.json.
    cfg.stop = &g_stop;
    std::signal(SIGINT, on_ctrl_c);
#ifdef SIGBREAK
    std::signal(SIGBREAK, on_ctrl_c);
#endif
    if (out_dir.empty()) out_dir = "runs/" + timestamp();
    FILE* log = cfg.events ? stderr : stdout;  // with --events, stdout is only JSON

    try {
        pf::EngineLock lock(max_engines);  // before the GPU is touched; held until this engine exits
        fs::create_directories(out_dir);
        if (dump) cfg.dump_dir = out_dir;
        if (profile) mopt.profile = out_dir + "/ort";

        pf::Wav wav;
        if (!live) {
            wav = pf::read_wav(wav_path);
            if (wav.sample_rate != 16000) {
                std::fprintf(stderr, "%s is %d Hz. Convert it first: python py/prep_audio.py %s\n", wav_path.c_str(),
                             wav.sample_rate, wav_path.c_str());
                return 1;
            }
            if (start_seconds > 0) {  // resume: the run starts here, and its times count from here
                const size_t skip = std::min(wav.mono.size(), size_t(start_seconds * 16000));
                wav.mono.erase(wav.mono.begin(), wav.mono.begin() + (std::ptrdiff_t)skip);
            }
            if (max_seconds > 0 && wav.mono.size() > size_t(max_seconds * 16000)) wav.mono.resize(size_t(max_seconds * 16000));
        }
        const auto sv = read_floats(models + "/sv_" + style + ".bin", pf::kStyleDim);

        const bool fp16_graphs = trt ? mixed : fp16;
        mopt.frame_path = models + (fp16_graphs ? "/frame_fp16.onnx" : "/frame.onnx");
        mopt.time_path = models + (fp16_graphs ? "/time_fp16.onnx" : "/time.onnx");
        if (trt) {
            mopt.trt = true;
            mopt.trt_fp16 = fp16;
            mopt.trt_opt_frames = cfg.stride + 2 * pf::kMargin;
            mopt.trt_cache = models + "/trt_cache";
            fs::create_directories(mopt.trt_cache);
        }
        std::unique_ptr<pf::Model> model;
        if (try_cuda) {
            mopt.cuda = true;
            try {
                model = std::make_unique<pf::Model>(mopt);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "CUDA unavailable (%s), using CPU\n", e.what());
            }
        }
        if (!model) {
            mopt.cuda = false;
            model = std::make_unique<pf::Model>(mopt);
        }

        char length[48];
        if (live && cfg.max_seconds <= 0) std::snprintf(length, sizeof length, "until stopped");
        else std::snprintf(length, sizeof length, "%.1f s", live ? cfg.max_seconds : double(wav.mono.size()) / 16000);
        std::fprintf(log, "%s: %s, provider %s%s, stride %d frames (%.0f ms), delay %d frames (%.0f ms), heads %c, %s, %s%s\n",
                     live ? input_name.c_str() : wav_path.c_str(), length,
                     model->provider(), mixed ? " FP16 mixed" : fp16 ? " FP16" : "", cfg.stride, cfg.stride * 16.0,
                     cfg.delay, cfg.delay * 16.0, cfg.heads, cfg.cache ? "cached" : "no cache",
                     cfg.realtime ? "real time" : "fast", cfg.normalize ? ", auto level" : "");

        // Warm-up: the first ONNX Runtime run (especially on CUDA) is slow.
        {
            pf::EngineConfig warm = cfg;
            warm.input = pf::Input::File;
            warm.normalize = false;
            warm.stop = nullptr;
            warm.realtime = false;
            warm.events = false;
            warm.dump_dir.clear();
            pf::Engine e(*model, sv, warm);
            pf::Metrics m;
            std::vector<float> two_sec(32000, 0.0f);
            e.run(two_sec, m);
        }

        pf::Engine engine(*model, sv, cfg);
        pf::Metrics mx;
        const auto notes = engine.run(wav.mono, mx);
        // A live run stopped before any audio arrived has nothing to divide by.
        const double rtf = mx.audio_sec > 0 ? mx.compute_sec / mx.audio_sec : 0.0;

        {
            std::ofstream f(out_dir + "/notes.csv");
            f << "pitch,onset,offset,velocity\n";
            char buf[128];
            for (const auto& n : notes) {
                std::snprintf(buf, sizeof buf, "%d,%.6f,%.6f,%d\n", n.pitch, n.onset, n.offset, n.velocity);
                f << buf;
            }
        }
        {
            std::ofstream f(out_dir + "/metrics.json");
            f << "{\n";
            f << "  \"command\": \"" << json_escape(command_line(argc, argv)) << "\",\n";
            f << "  \"input\": \"" << input_name << "\",\n";
            f << "  \"capture_device\": \"" << json_escape(mx.capture_device) << "\",\n";
            f << "  \"wav\": \"" << fs::path(wav_path).filename().string() << "\",\n";
            f << "  \"provider\": \"" << mx.provider << "\",\n";
            f << "  \"precision\": \"" << ((fp16 || mixed) ? "fp16" : "fp32") << "\",\n";
            f << "  \"stride_frames\": " << mx.stride << ",\n";
            f << "  \"delay_frames\": " << mx.delay << ",\n";
#ifdef PF_USE_SPSC
            f << "  \"ring_buffer\": \"spsc\",\n";
#else
            f << "  \"ring_buffer\": \"mutex\",\n";
#endif
            f << "  \"heads\": \"" << mx.heads << "\",\n";
            f << "  \"cache\": " << (mx.cache ? "true" : "false") << ",\n";
            f << "  \"realtime\": " << (mx.realtime ? "true" : "false") << ",\n";
            f << "  \"normalize\": " << (cfg.normalize ? "true" : "false") << ",\n";
            f << json_stats("gain", mx.gain) << ",\n";
            f << "  \"audio_sec\": " << mx.audio_sec << ",\n";
            f << "  \"start_seconds\": " << start_seconds << ",\n";
            f << "  \"wall_sec\": " << mx.wall_sec << ",\n";
            f << "  \"compute_sec\": " << mx.compute_sec << ",\n";
            f << "  \"real_time_factor\": " << rtf << ",\n";
            f << "  \"passes\": " << mx.passes << ",\n";
            f << "  \"notes\": " << mx.notes << ",\n";
            f << "  \"overrun_samples\": " << mx.overrun_samples << ",\n";
            f << "  \"peak_memory_mb\": " << peak_memory_mb() << ",\n";
            f << "  \"max_mel_frames\": " << mx.max_mel_frames << ",\n";
            f << "  \"max_decoder_frames\": " << mx.max_decoder_frames << ",\n";
            f << json_stats("frame_model_ms", mx.frame_ms) << ",\n";
            f << json_stats("time_model_ms", mx.time_ms) << ",\n";
            f << json_stats("decode_ms", mx.decode_ms) << ",\n";
            f << json_stats("pass_ms", mx.pass_ms) << ",\n";
            f << json_stats("frame_lag_ms", mx.frame_lag_ms) << ",\n";
            f << json_stats("note_lag_ms", mx.note_lag_ms) << ",\n";
            f << json_stats("push_jitter_ms", mx.push_jitter_ms) << "\n";
            f << "}\n";
        }

        const Stats pass = stats(mx.pass_ms), lag = stats(mx.frame_lag_ms), nlag = stats(mx.note_lag_ms);
        std::fprintf(log, "passes %zu | pass ms p50 %.1f p99 %.1f | real-time factor %.3f | overruns %zu samples\n",
                     mx.passes, pass.p50, pass.p99, rtf, mx.overrun_samples);
        if (cfg.realtime)
            std::fprintf(log, "frame lag ms p50 %.0f p99 %.0f | note-on lag ms p50 %.0f p99 %.0f\n", lag.p50, lag.p99,
                         nlag.p50, nlag.p99);
        if (cfg.normalize) {
            const Stats gain = stats(mx.gain);
            std::fprintf(log, "input gain p50 x%.2f, lowest x%.2f, highest x%.2f\n", gain.p50,
                         mx.gain.empty() ? 0.0 : *std::min_element(mx.gain.begin(), mx.gain.end()), gain.max);
        }
        std::fprintf(log, "%zu notes -> %s\n", notes.size(), out_dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
