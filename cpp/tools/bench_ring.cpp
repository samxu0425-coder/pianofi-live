// Ring buffer benchmark: one producer, one consumer, like the audio thread and
// the inference thread. Reports throughput and the time each push() takes
// (that cost lands on the audio thread, so its tail is what matters).
//
//   bench_ring [seconds]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "pf/ring_buffer.h"

using Clock = std::chrono::steady_clock;

template <class RB>
void run(const char* name, double seconds) {
    RB rb(1 << 14);
    const size_t block = 160;  // 10 ms at 16 kHz, a typical audio callback
    std::atomic<bool> stop{false};
    std::vector<double> push_ns;
    push_ns.reserve(1 << 24);
    size_t pushed = 0, popped = 0, full = 0;

    std::thread consumer([&] {
        std::vector<float> out(4096);
        while (!stop.load(std::memory_order_relaxed) || rb.size() > 0) popped += rb.pop(out.data(), out.size());
    });
    std::vector<float> in(block, 0.5f);
    const auto end = Clock::now() + std::chrono::duration<double>(seconds);
    while (Clock::now() < end) {
        const auto t0 = Clock::now();
        const size_t n = rb.push(in.data(), block);
        const auto t1 = Clock::now();
        push_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        pushed += n;
        if (n < block) ++full;
    }
    stop = true;
    consumer.join();

    std::sort(push_ns.begin(), push_ns.end());
    auto pct = [&](double p) { return push_ns[std::min(push_ns.size() - 1, size_t(p * push_ns.size()))]; };
    std::printf("%-8s %8.1f M samples/s   push p50 %6.0f ns  p99 %7.0f ns  p99.9 %8.0f ns  max %9.0f ns  full %zu\n",
                name, pushed / seconds / 1e6, pct(0.5), pct(0.99), pct(0.999), push_ns.back(), full);
    if (pushed != popped) std::printf("  ERROR: pushed %zu but popped %zu\n", pushed, popped);
}

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 2.0;
    run<pf::MutexRingBuffer>("mutex", seconds);
#ifdef PF_USE_SPSC
    run<pf::SpscRingBuffer>("spsc", seconds);
#else
    std::printf("spsc     (not built: this build has -DPF_USE_SPSC=OFF)\n");
#endif
}
