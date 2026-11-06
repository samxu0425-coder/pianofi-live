// Unit tests, no framework: each TEST registers itself, main runs them all.
// Build target: pf_tests. Exit code 0 means everything passed.
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "pf/auto_gain.h"
#include "pf/decoder.h"
#include "pf/mel.h"
#include "pf/ring_buffer.h"

namespace {

struct Test {
    const char* name;
    std::function<void()> fn;
};
std::vector<Test>& registry() {
    static std::vector<Test> r;
    return r;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
int g_failures = 0;

#define TEST(name)                         \
    static void name();                    \
    static Reg reg_##name(#name, name);    \
    static void name()

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

// ---------------------------------------------------------------- ring buffers

template <class RB>
void ring_basic() {
    RB rb(5);  // rounds up to 8
    CHECK(rb.capacity() == 8);
    float in[10], out[10];
    for (int i = 0; i < 10; ++i) in[i] = float(i);
    CHECK(rb.pop(out, 4) == 0);         // empty
    CHECK(rb.push(in, 6) == 6);
    CHECK(rb.size() == 6);
    CHECK(rb.push(in + 6, 4) == 2);     // only 2 slots left: never blocks
    CHECK(rb.size() == 8);
    CHECK(rb.pop(out, 3) == 3);
    CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2);
    CHECK(rb.push(in, 3) == 3);         // wraps around the end of the array
    CHECK(rb.pop(out, 10) == 8);
    const float expect[8] = {3, 4, 5, 6, 7, 0, 1, 2};
    for (int i = 0; i < 8; ++i) CHECK(out[i] == expect[i]);
    CHECK(rb.size() == 0);
}

// Producer pushes 0, 1, 2, ... in random-sized blocks; consumer must see the
// same sequence with nothing lost, duplicated or reordered.
template <class RB>
void ring_two_threads() {
    RB rb(1024);
    const size_t total = 2'000'000;
    std::atomic<bool> ok{true};
    std::thread producer([&] {
        std::mt19937 rng(1);
        std::vector<float> block(300);
        size_t next = 0;
        while (next < total) {
            size_t n = std::min<size_t>(1 + rng() % 300, total - next);
            for (size_t k = 0; k < n; ++k) block[k] = float(next + k);
            size_t done = 0;
            while (done < n) done += rb.push(block.data() + done, n - done);
            next += n;
        }
    });
    std::thread consumer([&] {
        std::mt19937 rng(2);
        std::vector<float> block(300);
        size_t expect = 0;
        while (expect < total) {
            size_t got = rb.pop(block.data(), 1 + rng() % 300);
            for (size_t k = 0; k < got; ++k)
                if (block[k] != float(expect++ % 16777216)) ok = false;  // float is exact to 2^24
        }
    });
    producer.join();
    consumer.join();
    CHECK(ok.load());
}

TEST(mutex_ring_basic) { ring_basic<pf::MutexRingBuffer>(); }
TEST(mutex_ring_two_threads) { ring_two_threads<pf::MutexRingBuffer>(); }
#ifdef PF_USE_SPSC
TEST(spsc_ring_basic) { ring_basic<pf::SpscRingBuffer>(); }
TEST(spsc_ring_two_threads) { ring_two_threads<pf::SpscRingBuffer>(); }
#endif

// ------------------------------------------------------------------- auto gain

// One chunk per second keeps the numbers easy: a 10 s window is 10 chunks.
bool near(float a, float b) { return std::fabs(a - b) < 1e-4f * std::max(1.0f, std::fabs(b)); }

TEST(auto_gain_brings_peak_to_full_scale) {
    pf::AutoGain g(1.0);
    CHECK(near(g.gain(), 1.0f));          // nothing heard yet
    g.add_peak(0.25f);
    CHECK(near(g.update(1, 1.0), 4.0f));  // first reading: jump straight there
    g.add_peak(0.5f);
    CHECK(near(g.update(2, 1.0), 2.0f));  // louder audio: drop at once
}

TEST(auto_gain_rises_slowly_and_forgets_old_peaks) {
    pf::AutoGain g(1.0);
    g.add_peak(1.0f);
    CHECK(near(g.update(1, 1.0), 1.0f));
    for (int i = 0; i < 9; ++i) g.add_peak(0.1f);
    CHECK(near(g.update(10, 1.0), 1.0f));  // the loud chunk is still inside the 10 s window
    g.add_peak(0.1f);
    const float step = float(std::pow(10.0, 6.0 / 20.0));  // 6 dB per second
    CHECK(near(g.update(11, 1.0), step));  // window moved past it: rise, but only 6 dB
    CHECK(near(g.update(11, 1.0), step * step));
    for (int i = 0; i < 10; ++i) g.update(11, 1.0);
    CHECK(near(g.gain(), 10.0f));          // settles where the 0.1 peak is at full scale
}

TEST(auto_gain_holds_in_silence_and_caps) {
    pf::AutoGain g(1.0);
    g.add_peak(0.5f);
    CHECK(near(g.update(1, 1.0), 2.0f));
    for (int i = 0; i < 30; ++i) g.add_peak(0.0f);
    CHECK(near(g.update(31, 1.0), 2.0f));  // silence: nothing to measure, keep the gain
    pf::AutoGain quiet(1.0);
    quiet.add_peak(0.002f);
    CHECK(near(quiet.update(1, 1.0), 32.0f));  // 1/0.002 = 500, capped at 32
    pf::AutoGain early(1.0);
    early.add_peak(0.5f);
    early.add_peak(1.0f);
    CHECK(near(early.update(1, 1.0), 2.0f));  // a pass only uses the chunks it has heard
}

// ------------------------------------------------------------------------- mel

TEST(fft_matches_dft) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    const size_t n = 64;
    std::vector<std::complex<double>> a(n), ref(n);
    for (auto& x : a) x = {u(rng), u(rng)};
    for (size_t k = 0; k < n; ++k)
        for (size_t t = 0; t < n; ++t)
            ref[k] += a[t] * std::polar(1.0, -2 * 3.14159265358979323846 * double(k * t) / double(n));
    pf::MelStream::fft(a);
    double err = 0;
    for (size_t k = 0; k < n; ++k) err = std::max(err, std::abs(a[k] - ref[k]));
    CHECK(err < 1e-9);
}

TEST(mel_frame_count_and_streaming) {
    // 1 + N/hop frames, and pushing in odd-sized chunks gives the same frames
    // as pushing everything at once.
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<float> x(16000 * 3 + 123);
    for (auto& v : x) v = u(rng);

    auto run = [&](size_t chunk) {
        pf::MelStream ms;
        std::vector<float> frames, f(256);
        for (size_t i = 0; i < x.size(); i += chunk) {
            ms.push(x.data() + i, std::min(chunk, x.size() - i));
            while (ms.next_frame(f.data())) frames.insert(frames.end(), f.begin(), f.end());
        }
        ms.finish();
        while (ms.next_frame(f.data())) frames.insert(frames.end(), f.begin(), f.end());
        return frames;
    };
    auto all = run(x.size());
    auto streamed = run(157);
    CHECK(all.size() == (1 + x.size() / 256) * 256);
    CHECK(all == streamed);
}

TEST(mel_frame_ready_needs_lookahead) {
    pf::MelStream ms;
    std::vector<float> zeros(1024, 0.0f), f(256);
    ms.push(zeros.data(), 1023);
    CHECK(!ms.next_frame(f.data()));  // frame 0 needs samples up to 1024
    ms.push(zeros.data(), 1);
    CHECK(ms.next_frame(f.data()));
    CHECK(std::abs(f[0] - std::log(1e-8f)) < 1e-4);  // silence -> log(1e-8)
}

// --------------------------------------------------------------------- decoder

// Python mpe2note on hand-made arrays, worked out by hand.
TEST(decoder_single_note_mpe_end) {
    pf::DecoderConfig cfg;
    cfg.num_note = 1;
    cfg.hop_sec = 0.1;
    cfg.min_duration = 0.0;
    pf::NoteDecoder d(cfg);
    // onset peak at 2 (0.9, neighbours 0.2 and 0.4 -> interpolated)
    const float on[] = {0.1f, 0.2f, 0.9f, 0.4f, 0.1f, 0.1f, 0.1f, 0.1f};
    const float of[] = {0, 0, 0, 0, 0, 0, 0, 0};
    const float mp[] = {0, 0, 0.9f, 0.9f, 0.9f, 0.2f, 0, 0};
    const int64_t ve[] = {0, 0, 64, 0, 0, 0, 0, 0};
    // Feed one frame at a time, like the engine.
    for (int i = 0; i < 8; ++i) d.add_frames(on + i, of + i, mp + i, ve + i, 1);
    d.finish();
    auto n = d.notes();
    CHECK(n.size() == 1);
    if (n.size() == 1) {
        // p=0.2 < q=0.4: t = 0.2 + 0.1*0.5*(0.4-0.2)/(0.9-0.2)
        CHECK(std::abs(n[0].onset - (0.2 + 0.05 * 0.2 / 0.7)) < 1e-6);
        CHECK(std::abs(n[0].offset - 0.5) < 1e-9);  // first mpe < 0.5 at frame 5
        CHECK(n[0].velocity == 64 && n[0].pitch == 21);
    }
}

TEST(decoder_next_onset_ends_note_and_zero_velocity_dropped) {
    pf::DecoderConfig cfg;
    cfg.num_note = 1;
    cfg.hop_sec = 0.1;
    cfg.min_duration = 0.0;
    pf::NoteDecoder d(cfg);
    const float on[] = {0.9f, 0.1f, 0.1f, 0.8f, 0.1f, 0.7f, 0.1f};
    const float of[] = {0, 0, 0, 0, 0, 0, 0};
    const float mp[] = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
    const int64_t ve[] = {50, 0, 0, 0, 0, 40, 0};  // the onset at 3 has velocity 0
    d.add_frames(on, of, mp, ve, 7);
    d.finish();
    auto n = d.notes();
    CHECK(n.size() == 2);
    if (n.size() == 2) {
        CHECK(n[0].onset == 0.0);                         // first frame: no interpolation
        CHECK(std::abs(n[0].offset - 0.3) < 1e-9);         // ends at next onset (frame 3, p==q)
        CHECK(std::abs(n[1].offset - 0.6) < 1e-9);         // last note: (len-1)*hop
    }
}

TEST(decoder_waits_on_plateau) {
    pf::DecoderConfig cfg;
    cfg.num_note = 1;
    cfg.hop_sec = 0.1;
    pf::NoteDecoder d(cfg);
    int on_count = 0;
    d.on_note_on([&](const pf::Note&, size_t) { ++on_count; });
    const float a[] = {0.1f, 0.9f, 0.9f};
    const float z[] = {0, 0, 0};
    const float mp[] = {0.9f, 0.9f, 0.9f};
    const int64_t ve[] = {0, 10, 10};
    d.add_frames(a, z, mp, ve, 3);
    CHECK(on_count == 0);  // plateau reaches the newest frame: undecided
    const float a2[] = {0.2f}, z2[] = {0}, m2[] = {0.9f};
    const int64_t v2[] = {0};
    d.add_frames(a2, z2, m2, v2, 1);
    CHECK(on_count == 2);  // both plateau frames are peaks (Python treats equal neighbours as "keep scanning")
}

// Dropping old frames must not change a single note. Random data with short
// plateaus, saturated offsets and zero velocities, fed 16 frames at a time.
TEST(decoder_trimmed_history_gives_the_same_notes) {
    const int notes = 12, frames = 20000;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> on(size_t(frames) * notes), off(on.size()), mpe(on.size());
    std::vector<int64_t> vel(on.size());
    for (size_t i = 0; i < on.size(); ++i) {
        // Mostly quiet, with occasional peaks; repeat the previous frame sometimes to make plateaus.
        const bool repeat = i >= size_t(notes) && uni(rng) < 0.1f;
        on[i] = repeat ? on[i - notes] : (uni(rng) < 0.05f ? 0.5f + 0.5f * uni(rng) : 0.3f * uni(rng));
        off[i] = uni(rng) < 0.02f ? 1.0f : uni(rng);
        mpe[i] = uni(rng) < 0.97f ? 0.9f : 0.1f;  // notes last a while
        vel[i] = uni(rng) < 0.1f ? 0 : 1 + int64_t(126 * uni(rng));
    }
    pf::DecoderConfig cfg;
    cfg.num_note = notes;
    pf::NoteDecoder plain(cfg);
    cfg.trim_history = true;
    pf::NoteDecoder trimmed(cfg);
    size_t most_kept = 0;
    for (int f = 0; f < frames; f += 16) {
        const size_t at = size_t(f) * notes;
        plain.add_frames(&on[at], &off[at], &mpe[at], &vel[at], 16);
        trimmed.add_frames(&on[at], &off[at], &mpe[at], &vel[at], 16);
        most_kept = std::max(most_kept, trimmed.frames_kept());
    }
    plain.finish();
    trimmed.finish();
    const auto a = plain.notes(), b = trimmed.notes();
    CHECK(a.size() > 1000);  // the data really does make notes
    CHECK(a.size() == b.size());
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].pitch == b[i].pitch && a[i].onset == b[i].onset && a[i].offset == b[i].offset &&
               a[i].velocity == b[i].velocity;
    CHECK(same);
    CHECK(plain.frames_kept() == size_t(frames));
    CHECK(most_kept < 4000);  // far less than the 20000 frames it was given
}

}  // namespace

int main() {
    for (const auto& t : registry()) {
        const int before = g_failures;
        std::printf("%-55s", t.name);
        t.fn();
        std::printf("%s\n", g_failures == before ? "ok" : "FAILED");
    }
#ifndef PF_USE_SPSC
    std::printf("(SPSC ring buffer tests skipped: this build has -DPF_USE_SPSC=OFF)\n");
#endif
    std::printf("%s\n", g_failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_failures ? 1 : 0;
}
