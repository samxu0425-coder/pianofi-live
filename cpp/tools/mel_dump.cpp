// Writes the streaming mel spectrogram of a 16 kHz wav as raw float32
// [frames][256], for py/check_mel.py to compare against Pipeline.wav2feature.
//
//   mel_dump input_16k.wav out.f32
#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <vector>

#include "pf/mel.h"
#include "pf/wav.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: mel_dump input_16k.wav out.f32\n");
        return 2;
    }
    try {
        pf::Wav w = pf::read_wav(argv[1]);
        if (w.sample_rate != 16000) {
            std::fprintf(stderr, "%s is %d Hz; run py/prep_audio.py first\n", argv[1], w.sample_rate);
            return 1;
        }
        pf::MelStream ms;
        std::vector<float> f(256);
        std::ofstream out(argv[2], std::ios::binary);
        size_t frames = 0;
        // Push in 10 ms blocks to exercise the streaming path.
        for (size_t i = 0; i < w.mono.size(); i += 160) {
            ms.push(w.mono.data() + i, std::min<size_t>(160, w.mono.size() - i));
            while (ms.next_frame(f.data())) { out.write(reinterpret_cast<const char*>(f.data()), 256 * 4); ++frames; }
        }
        ms.finish();
        while (ms.next_frame(f.data())) { out.write(reinterpret_cast<const char*>(f.data()), 256 * 4); ++frames; }
        std::printf("%zu frames\n", frames);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
