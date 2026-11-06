// Minimal WAV reader: PCM 8/16/24/32-bit and float 32/64, any channel count.
// Returns mono float samples in [-1, 1] (channels averaged, like
// torch.mean(wave, dim=0) in AMT-APC's wav2feature).
#pragma once

#include <string>
#include <vector>

namespace pf {

struct Wav {
    std::vector<float> mono;
    int sample_rate = 0;
    int channels = 0;
};

// Throws std::runtime_error with a readable message on failure.
Wav read_wav(const std::string& path);

}  // namespace pf
