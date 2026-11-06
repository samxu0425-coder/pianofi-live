#include "pf/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "pf/capture.h"
#include "pf/mel.h"
#include "pf/ring_buffer.h"

namespace pf {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kHop = 256;
constexpr int kSr = 16000;
constexpr float kMinValue = -18.0f;

double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

long long floor_mod(long long a, long long m) { return ((a % m) + m) % m; }

template <class T>
void write_raw(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

// What the audio callback needs. Everything is allocated before capture starts.
struct LiveAudio {
    AudioRingBuffer* ring;
    std::atomic<long long>* block_ns;  // arrival time of each block, for the lag numbers
    size_t n_blocks;
    size_t block;
    Clock::time_point t0;
    std::atomic<size_t>* overruns;
    std::atomic<size_t> samples_seen{0};
};

// The audio thread: no allocation, no locks, no I/O, no logging.
void on_live_audio(const float* samples, size_t n, void* user) {
    if (n == 0) return;
    auto& live = *static_cast<LiveAudio*>(user);
    const size_t seen = live.samples_seen.load(std::memory_order_relaxed);
    const long long now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - live.t0).count();
    const size_t last_block = std::min((seen + n - 1) / live.block, live.n_blocks - 1);
    for (size_t b = seen / live.block; b <= last_block; ++b) live.block_ns[b].store(now_ns, std::memory_order_relaxed);
    const size_t done = live.ring->push(samples, n);
    if (done < n) live.overruns->fetch_add(n - done, std::memory_order_relaxed);
    live.samples_seen.store(seen + n, std::memory_order_relaxed);
}

}  // namespace

Engine::Engine(Model& model, const std::vector<float>& sv, const EngineConfig& cfg)
    : model_(model), sv_(sv), cfg_(cfg) {
    if (sv_.size() != size_t(kStyleDim)) throw std::invalid_argument("style vector must have 24 floats");
    if (cfg_.stride < 1 || cfg_.stride > kWindow) throw std::invalid_argument("stride must be 1..512 frames");
    if (cfg_.heads != 'a' && cfg_.heads != 'b') throw std::invalid_argument("heads must be a or b");
    if (cfg_.delay < 0 || cfg_.stride + cfg_.delay > kWindow) throw std::invalid_argument("stride + delay must be 1..512 frames");
    if (cfg_.delay > 0 && cfg_.heads != 'b') throw std::invalid_argument("delay needs the time model (heads b)");
    if (cfg_.input != Input::File) {
        if (!cfg_.realtime) throw std::invalid_argument("live input can't run with --fast");
        if (cfg_.max_seconds <= 0) throw std::invalid_argument("max_seconds must be positive");
#ifndef PF_USE_SPSC
        throw std::invalid_argument("live input needs the lock-free ring buffer: build with -DPF_USE_SPSC=ON");
#endif
    }
}

std::vector<Note> Engine::run(const std::vector<float>& samples, Metrics& mx) {
    const int S = cfg_.stride;
    const int D = cfg_.delay;
    const bool use_b = cfg_.heads == 'b';
    const bool cache = cfg_.cache || !use_b;  // A heads never need the window
    const bool live = cfg_.input != Input::File;
    const size_t total_samples = live ? size_t(cfg_.max_seconds * kSr) : samples.size();  // most we can see
    mx = Metrics{};
    mx.provider = model_.provider();
    mx.stride = S;
    mx.delay = D;
    mx.heads = cfg_.heads;
    mx.cache = cache;
    mx.realtime = cfg_.realtime;
    mx.audio_sec = double(total_samples) / kSr;

    // ---------------------------------------------------------------- buffers
    // All allocation happens here, before the audio thread starts.
    AudioRingBuffer ring(cfg_.ring_capacity);
    const size_t n_blocks = (total_samples + cfg_.block - 1) / cfg_.block;
    std::unique_ptr<std::atomic<long long>[]> block_ns(new std::atomic<long long>[n_blocks + 1]);
    for (size_t b = 0; b <= n_blocks; ++b) block_ns[b].store(-1, std::memory_order_relaxed);
    std::vector<double> jitter(n_blocks, 0.0);
    std::atomic<bool> producer_done{false};
    std::atomic<size_t> overruns{0};
    LiveAudio live_audio{&ring, block_ns.get(), n_blocks, size_t(cfg_.block), {}, &overruns};
    std::unique_ptr<Capture> capture;
    if (live) {
        capture = std::make_unique<Capture>(cfg_.input == Input::Loopback, on_live_audio, &live_audio);
        mx.capture_device = capture->description();
        std::fprintf(stderr, "Capturing from %s\n", mx.capture_device.c_str());
    }

    // Mirrored h cache: frame t lives in slots t%512 and t%512 + 512, so any
    // 512 consecutive frames are contiguous and go to the time model with no copy.
    std::vector<float> hcache(cache && use_b ? 2 * size_t(kWindow) * kHSize : 0);
    std::vector<float> spec_in, h_stage, a_on, a_off, a_mpe;
    std::vector<int64_t> a_vel;
    std::vector<float> b_on(size_t(kWindow) * kNotes), b_off(b_on.size()), b_mpe(b_on.size());
    std::vector<int64_t> b_vel(b_on.size());
    std::vector<float> spec;  // all mel frames so far, [t][256]
    spec.reserve(size_t(mx.audio_sec * kSr / kHop + 1024) * kBins);

    auto spec_at = [&](long long t, int b, long long n_have) -> float {
        if (t < 0 || t >= n_have) return kMinValue;
        return spec[size_t(t) * kBins + b];
    };
    // Bin-major model input for output frames [lo, hi): spec frames [lo-32, hi+32).
    auto build_input = [&](long long lo, long long hi, long long n_have) {
        const int in_frames = int(hi - lo) + 2 * kMargin;
        spec_in.resize(size_t(kBins) * in_frames);
        for (int b = 0; b < kBins; ++b)
            for (int i = 0; i < in_frames; ++i)
                spec_in[size_t(b) * in_frames + i] = spec_at(lo - kMargin + i, b, n_have);
        return in_frames;
    };
    auto run_frames = [&](long long lo, long long hi, long long n_have) {
        const int in_frames = build_input(lo, hi, n_have);
        const size_t n = size_t(hi - lo);
        h_stage.resize(n * kHSize);
        a_on.resize(n * kNotes); a_off.resize(n * kNotes); a_mpe.resize(n * kNotes); a_vel.resize(n * kNotes);
        model_.run_frame(spec_in.data(), in_frames, sv_.data(), h_stage.data(), a_on.data(), a_off.data(),
                         a_mpe.data(), a_vel.data());
    };

    if (cache && use_b) {
        // Frames before -32 see only padding, so they all share one h.
        run_frames(0, 1, 0);
        for (int slot = 0; slot < 2 * kWindow; ++slot)
            std::memcpy(&hcache[size_t(slot) * kHSize], h_stage.data(), kHSize * sizeof(float));
    }

    // ---------------------------------------------------------------- decoder
    DecoderConfig dcfg;
    NoteDecoder decoder(dcfg);
    // Lag is only recorded while audio is still streaming: after the end of the
    // file the engine flushes with padding, which a live stream never does.
    bool eof = false;
    const auto t0 = Clock::now();
    auto arrival_ms = [&](long long sample) -> double {
        if (sample < 0 || size_t(sample) >= total_samples) return -1;
        const long long ns = block_ns[size_t(sample) / cfg_.block].load(std::memory_order_relaxed);
        return ns < 0 ? -1 : ns / 1e6;
    };
    auto t_wall = [&] { return ms_since(t0, Clock::now()) / 1000.0; };
    decoder.on_note_on([&](const Note& n, size_t frame) {
        if (cfg_.events) {
            std::printf("{\"type\":\"on\",\"pitch\":%d,\"onset\":%.6f,\"vel\":%d,\"t_wall\":%.3f}\n", n.pitch, n.onset,
                        n.velocity, t_wall());
            std::fflush(stdout);
        }
        if (!cfg_.realtime || eof) return;
        const double arr = arrival_ms((long long)frame * kHop);
        if (arr >= 0) mx.note_lag_ms.push_back(ms_since(t0, Clock::now()) - arr);
    });
    if (cfg_.events) {
        decoder.on_note_off([&](const Note& n) {
            std::printf("{\"type\":\"off\",\"pitch\":%d,\"onset\":%.6f,\"offset\":%.6f,\"t_wall\":%.3f}\n", n.pitch,
                        n.onset, n.offset, t_wall());
            std::fflush(stdout);
        });
    }

    // Raw outputs for py/check_decoder.py.
    std::vector<float> d_on, d_off, d_mpe;
    std::vector<int64_t> d_vel;
    auto emit = [&](const float* on, const float* off, const float* mpe, const int64_t* vel, long long first,
                    size_t count, long long n_output) {
        if (first + (long long)count > n_output) count = size_t(std::max(0LL, n_output - first));
        if (count == 0) return;
        const auto t_ready = Clock::now();
        if (cfg_.realtime && !eof) {
            for (size_t r = 0; r < count; ++r) {
                const double arr = arrival_ms((first + (long long)r) * kHop);
                if (arr >= 0) mx.frame_lag_ms.push_back(ms_since(t0, t_ready) - arr);
            }
        }
        decoder.add_frames(on, off, mpe, vel, count);
        if (!cfg_.dump_dir.empty()) {
            const size_t k = count * kNotes;
            d_on.insert(d_on.end(), on, on + k);
            d_off.insert(d_off.end(), off, off + k);
            d_mpe.insert(d_mpe.end(), mpe, mpe + k);
            d_vel.insert(d_vel.end(), vel, vel + k);
        }
    };

    // Printed here because the audio thread does no I/O.
    if (cfg_.events) {
        std::printf("{\"type\":\"start\",\"t_wall\":0}\n");
        std::fflush(stdout);
    }

    // ----------------------------------------------------------- audio thread
    // Live input: the sound card calls on_live_audio. File input: this thread acts like one.
    live_audio.t0 = t0;
    if (live) capture->start();
    std::thread producer;
    if (!live) producer = std::thread([&] {
        const double block_sec = double(cfg_.block) / kSr;
        for (size_t b = 0; b < n_blocks; ++b) {
            const size_t lo = b * cfg_.block;
            const size_t n = std::min<size_t>(cfg_.block, samples.size() - lo);
            if (cfg_.realtime) {
                const auto sched = t0 + std::chrono::duration_cast<Clock::duration>(
                                            std::chrono::duration<double>(double(b) * block_sec));
                std::this_thread::sleep_until(sched);
                jitter[b] = ms_since(sched, Clock::now());
            }
            block_ns[b].store(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count(),
                              std::memory_order_relaxed);
            size_t done = ring.push(samples.data() + lo, n);
            if (cfg_.realtime) {
                if (done < n) overruns.fetch_add(n - done, std::memory_order_relaxed);
            } else {
                while (done < n) {
                    std::this_thread::yield();
                    done += ring.push(samples.data() + lo + done, n - done);
                }
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    // ------------------------------------------------------- inference thread
    MelStream mel;
    std::vector<float> pop_buf(8192), frame(kBins);
    long long n_have = 0;      // mel frames available
    long long n_output = -1;   // known after eof: batch pads to a multiple of 512
    long long next_h = use_b ? -kMargin : 0;  // next frame whose h is not computed yet
    long long k = 0;           // next pass

    while (true) {
        if (live && !producer_done.load(std::memory_order_relaxed)) {
            const bool asked = cfg_.stop && cfg_.stop->load(std::memory_order_relaxed);
            if (asked || ms_since(t0, Clock::now()) >= cfg_.max_seconds * 1000.0) {
                capture->stop();
                producer_done.store(true, std::memory_order_release);
            }
        }
        size_t got = ring.pop(pop_buf.data(), pop_buf.size());
        if (got > 0) {
            mel.push(pop_buf.data(), got);
        } else if (!eof && producer_done.load(std::memory_order_acquire) && ring.size() == 0) {
            mel.finish();
            eof = true;
        }
        while (mel.next_frame(frame.data())) {
            spec.insert(spec.end(), frame.begin(), frame.end());
            ++n_have;
        }
        if (eof) n_output = (n_have + kWindow - 1) / kWindow * kWindow;

        bool ran = false;
        while (true) {
            const long long out_end = (k + 1) * S;
            const long long win_end = out_end + D;  // the window runs D frames past what we emit
            if (eof && k * S >= n_output) break;
            if (!eof && n_have < win_end + kMargin) break;  // need 32 frames of lookahead

            const auto p0 = Clock::now();
            double frame_ms = 0, time_ms = 0;
            if (!use_b) {
                run_frames(next_h, out_end, n_have);
                frame_ms = ms_since(p0, Clock::now());
                const auto d0 = Clock::now();
                emit(a_on.data(), a_off.data(), a_mpe.data(), a_vel.data(), next_h, size_t(out_end - next_h), eof ? n_output : out_end);
                mx.decode_ms.push_back(ms_since(d0, Clock::now()));
                next_h = out_end;
            } else {
                const long long ws = win_end - kWindow;
                const float* window;
                if (cache) {
                    run_frames(next_h, win_end, n_have);
                    for (long long t = next_h; t < win_end; ++t) {
                        const float* src = &h_stage[size_t(t - next_h) * kHSize];
                        const size_t slot = size_t(floor_mod(t, kWindow));
                        std::memcpy(&hcache[slot * kHSize], src, kHSize * sizeof(float));
                        std::memcpy(&hcache[(slot + kWindow) * kHSize], src, kHSize * sizeof(float));
                    }
                    next_h = win_end;
                    window = &hcache[size_t(floor_mod(ws, kWindow)) * kHSize];
                } else {
                    run_frames(ws, win_end, n_have);  // all 512 frames again
                    window = h_stage.data();
                }
                const auto p1 = Clock::now();
                frame_ms = ms_since(p0, p1);
                model_.run_time(window, b_on.data(), b_off.data(), b_mpe.data(), b_vel.data());
                const auto p2 = Clock::now();
                time_ms = ms_since(p1, p2);
                const size_t off = size_t(kWindow - S - D) * kNotes;  // rows [512-S-D, 512-D)
                emit(b_on.data() + off, b_off.data() + off, b_mpe.data() + off, b_vel.data() + off, out_end - S,
                     size_t(S), eof ? n_output : out_end);
                mx.decode_ms.push_back(ms_since(p2, Clock::now()));
            }
            mx.frame_ms.push_back(frame_ms);
            mx.time_ms.push_back(time_ms);
            mx.pass_ms.push_back(ms_since(p0, Clock::now()));
            ++k;
            ran = true;
        }
        if (eof && k * S >= n_output) break;
        if (got == 0 && !ran) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (producer.joinable()) producer.join();
    decoder.finish();

    if (live) mx.audio_sec = double(live_audio.samples_seen.load()) / kSr;
    mx.wall_sec = ms_since(t0, Clock::now()) / 1000.0;
    mx.passes = size_t(k);
    for (double v : mx.pass_ms) mx.compute_sec += v / 1000.0;
    mx.overrun_samples = overruns.load();
    if (cfg_.realtime && !live) mx.push_jitter_ms = jitter;

    if (!cfg_.dump_dir.empty()) {
        write_raw(cfg_.dump_dir + "/onset.f32", d_on);
        write_raw(cfg_.dump_dir + "/offset.f32", d_off);
        write_raw(cfg_.dump_dir + "/mpe.f32", d_mpe);
        write_raw(cfg_.dump_dir + "/velocity.i64", d_vel);
        write_raw(cfg_.dump_dir + "/mel.f32", spec);
    }
    auto notes = decoder.notes();
    mx.notes = notes.size();
    return notes;
}

}  // namespace pf
