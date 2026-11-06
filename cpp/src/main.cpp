// pianofi_live: run the streaming engine on a 16 kHz wav and report metrics.
//
//   pianofi_live --wav test_audio\song_16k.wav [options]
//
//   --models DIR      folder with frame.onnx, time.onnx, sv_<style>.bin (default: models)
//   --style NAME      style vector file sv_NAME.bin (default: level2)
//   --stride N        output frames per pass, 16 ms each (default: 32 = 0.512 s)
//   --delay N         emit output N frames back from the newest; adds N*16 ms lag (default: 0)
//   --heads a|b       b = time model, same as batch (default); a = per-frame heads only
//   --no-cache        recompute all 512 frames every pass (the baseline to beat)
//   --fast            don't pace audio in real time (accuracy runs; no lag numbers)
//   --events          print one JSON line per note event to stdout; logs go to stderr
//   --cpu             use the CPU even if CUDA is available (default: try CUDA)
//   --threads N       ONNX Runtime intra-op threads
//   --out DIR         where notes.csv and metrics.json go (default: runs/<timestamp>)
//   --dump            also write raw model outputs + mel to --out for the Python checks
//   --max-seconds X   only use the first X seconds of the wav
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

#include "pf/engine.h"
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

int usage() {
    std::fprintf(stderr,
                 "usage: pianofi_live --wav FILE [--models DIR] [--style NAME] [--stride N] [--delay N] [--heads a|b]\n"
                 "                    [--no-cache] [--fast] [--events] [--cpu] [--threads N] [--out DIR] [--dump] [--max-seconds X]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    timeBeginPeriod(1);  // 1 ms sleep granularity instead of ~15 ms
#endif
    std::string wav_path, models = "models", style = "level2", out_dir;
    pf::EngineConfig cfg;
    pf::ModelOptions mopt;
    bool try_cuda = true, dump = false;
    double max_seconds = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        try {
            if (a == "--wav") wav_path = next();
            else if (a == "--models") models = next();
            else if (a == "--style") style = next();
            else if (a == "--stride") cfg.stride = std::stoi(next());
            else if (a == "--delay") cfg.delay = std::stoi(next());
            else if (a == "--heads") cfg.heads = next().at(0);
            else if (a == "--no-cache") cfg.cache = false;
            else if (a == "--fast") cfg.realtime = false;
            else if (a == "--events") cfg.events = true;
            else if (a == "--cpu") try_cuda = false;
            else if (a == "--threads") mopt.threads = std::stoi(next());
            else if (a == "--out") out_dir = next();
            else if (a == "--dump") dump = true;
            else if (a == "--max-seconds") max_seconds = std::stod(next());
            else return usage();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return usage();
        }
    }
    if (wav_path.empty()) return usage();
    if (out_dir.empty()) out_dir = "runs/" + timestamp();
    FILE* log = cfg.events ? stderr : stdout;  // with --events, stdout is only JSON

    try {
        fs::create_directories(out_dir);
        if (dump) cfg.dump_dir = out_dir;

        pf::Wav wav = pf::read_wav(wav_path);
        if (wav.sample_rate != 16000) {
            std::fprintf(stderr, "%s is %d Hz. Convert it first: python py/prep_audio.py %s\n", wav_path.c_str(),
                         wav.sample_rate, wav_path.c_str());
            return 1;
        }
        if (max_seconds > 0 && wav.mono.size() > size_t(max_seconds * 16000)) wav.mono.resize(size_t(max_seconds * 16000));
        const auto sv = read_floats(models + "/sv_" + style + ".bin", pf::kStyleDim);

        mopt.frame_path = models + "/frame.onnx";
        mopt.time_path = models + "/time.onnx";
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

        std::fprintf(log, "%s: %.1f s, provider %s, stride %d frames (%.0f ms), delay %d frames (%.0f ms), heads %c, %s, %s\n",
                     wav_path.c_str(), double(wav.mono.size()) / 16000, model->provider(), cfg.stride, cfg.stride * 16.0,
                     cfg.delay, cfg.delay * 16.0, cfg.heads, cfg.cache ? "cached" : "no cache",
                     cfg.realtime ? "real time" : "fast");

        // Warm-up: the first ONNX Runtime run (especially on CUDA) is slow.
        {
            pf::EngineConfig warm = cfg;
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
            f << "  \"wav\": \"" << fs::path(wav_path).filename().string() << "\",\n";
            f << "  \"provider\": \"" << mx.provider << "\",\n";
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
            f << "  \"audio_sec\": " << mx.audio_sec << ",\n";
            f << "  \"wall_sec\": " << mx.wall_sec << ",\n";
            f << "  \"compute_sec\": " << mx.compute_sec << ",\n";
            f << "  \"real_time_factor\": " << mx.compute_sec / mx.audio_sec << ",\n";
            f << "  \"passes\": " << mx.passes << ",\n";
            f << "  \"notes\": " << mx.notes << ",\n";
            f << "  \"overrun_samples\": " << mx.overrun_samples << ",\n";
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
                     mx.passes, pass.p50, pass.p99, mx.compute_sec / mx.audio_sec, mx.overrun_samples);
        if (cfg.realtime)
            std::fprintf(log, "frame lag ms p50 %.0f p99 %.0f | note-on lag ms p50 %.0f p99 %.0f\n", lag.p50, lag.p99,
                         nlag.p50, nlag.p99);
        std::fprintf(log, "%zu notes -> %s\n", notes.size(), out_dir.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
