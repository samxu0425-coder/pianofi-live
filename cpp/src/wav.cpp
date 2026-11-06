#include "pf/wav.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace pf {
namespace {

uint16_t u16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

}  // namespace

Wav read_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 12 || std::memcmp(data.data(), "RIFF", 4) != 0 || std::memcmp(data.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error(path + " is not a RIFF/WAVE file");

    int tag = 0, channels = 0, rate = 0, bits = 0;
    const uint8_t* pcm = nullptr;
    size_t pcm_size = 0;
    size_t pos = 12;
    while (pos + 8 <= data.size()) {
        const uint8_t* chunk = data.data() + pos;
        size_t size = u32(chunk + 4);
        size_t body_size = std::min(size, data.size() - pos - 8);
        const uint8_t* body = chunk + 8;
        if (std::memcmp(chunk, "fmt ", 4) == 0 && body_size >= 16) {
            tag = u16(body);
            channels = u16(body + 2);
            rate = int(u32(body + 4));
            bits = u16(body + 14);
            if (tag == 0xFFFE && body_size >= 26) tag = u16(body + 24);  // WAVE_FORMAT_EXTENSIBLE
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            pcm = body;
            pcm_size = body_size;
        }
        pos += 8 + size + (size & 1);
    }
    if (!pcm || channels <= 0) throw std::runtime_error(path + ": missing fmt or data chunk");

    const int bytes = bits / 8;
    if (bytes <= 0) throw std::runtime_error(path + ": bad bit depth");
    const size_t frames = pcm_size / (size_t(bytes) * channels);
    Wav w;
    w.sample_rate = rate;
    w.channels = channels;
    w.mono.assign(frames, 0.0f);

    auto sample = [&](const uint8_t* p) -> double {
        if (tag == 1 && bits == 8) return (double(p[0]) - 128.0) / 128.0;
        if (tag == 1 && bits == 16) return double(int16_t(u16(p))) / 32768.0;
        if (tag == 1 && bits == 24) {
            int32_t v = int32_t(uint32_t(p[0]) << 8 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 24);
            return double(v) / 2147483648.0;
        }
        if (tag == 1 && bits == 32) return double(int32_t(u32(p))) / 2147483648.0;
        if (tag == 3 && bits == 32) {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        if (tag == 3 && bits == 64) {
            double v;
            std::memcpy(&v, p, 8);
            return v;
        }
        throw std::runtime_error(path + ": unsupported format tag " + std::to_string(tag) + ", " +
                                 std::to_string(bits) + " bits");
    };

    for (size_t i = 0; i < frames; ++i) {
        // Match Python: convert each channel to float32, then average in float32.
        float acc = 0.0f;
        for (int c = 0; c < channels; ++c)
            acc += float(sample(pcm + (i * channels + c) * bytes));
        w.mono[i] = acc / float(channels);
    }
    return w;
}

}  // namespace pf
