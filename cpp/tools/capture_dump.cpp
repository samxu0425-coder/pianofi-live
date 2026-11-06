// Record live audio to a wav file, to check what the capture path delivers.
//
//   capture_dump loopback|mic SECONDS RATE OUT.wav [DEVICE]
//
// RATE 0 = the device's own rate. DEVICE = part of a device name (default: the Windows default).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "pf/capture.h"

namespace {

struct Recording {
    std::vector<float> samples;  // sized before capture starts: the callback never allocates
    std::atomic<size_t> count{0};
};

void on_audio(const float* in, size_t n, void* user) {
    auto& rec = *static_cast<Recording*>(user);
    const size_t have = rec.count.load(std::memory_order_relaxed);
    n = std::min(n, rec.samples.size() - have);
    std::memcpy(rec.samples.data() + have, in, n * sizeof(float));
    rec.count.store(have + n, std::memory_order_release);
}

template <class T>
void put(std::ofstream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }

void write_wav(const std::string& path, const float* data, size_t n, int rate) {
    std::ofstream f(path, std::ios::binary);
    const uint32_t bytes = uint32_t(n * sizeof(float));
    f.write("RIFF", 4); put<uint32_t>(f, 36 + bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); put<uint32_t>(f, 16); put<uint16_t>(f, 3);  // 3 = IEEE float
    put<uint16_t>(f, 1); put<uint32_t>(f, uint32_t(rate)); put<uint32_t>(f, uint32_t(rate) * 4);
    put<uint16_t>(f, 4); put<uint16_t>(f, 32);
    f.write("data", 4); put<uint32_t>(f, bytes);
    f.write(reinterpret_cast<const char*>(data), bytes);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "usage: capture_dump loopback|mic SECONDS RATE OUT.wav [DEVICE]\n");
        return 2;
    }
    try {
        const bool loopback = std::string(argv[1]) == "loopback";
        const double seconds = std::atof(argv[2]);
        Recording rec;
        pf::Capture capture(loopback, argc == 6 ? argv[5] : "", std::atoi(argv[3]), on_audio, &rec);
        rec.samples.resize(size_t((seconds + 1) * capture.sample_rate()));
        std::printf("%s, delivering %d Hz mono\n", capture.description().c_str(), capture.sample_rate());
        capture.start();
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        capture.stop();
        write_wav(argv[4], rec.samples.data(), rec.count.load(), capture.sample_rate());
        std::printf("%zu samples -> %s\n", rec.count.load(), argv[4]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
