// Streaming log-mel spectrogram matching AMT-APC's Pipeline.wav2feature, i.e.
// torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048,
//   win_length=2048, hop_length=256, pad_mode="constant", n_mels=256,
//   norm="slaney") followed by log(mel + 1e-8).
//
// torchaudio defaults that matter: center=True (frame t is centred on sample
// t*hop, zero-padded by n_fft/2 on both ends), periodic Hann window,
// power=2, HTK mel scale, f_min=0, f_max=sr/2.
//
// Streaming: frame t needs samples up to t*hop + n_fft/2, so it is ready
// 1024 samples (64 ms) after its centre arrives. After finish(), the
// remaining frames are produced with zero padding, giving 1 + N/hop frames in
// total, same as torchaudio.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace pf {

struct MelConfig {
    int sample_rate = 16000;
    int n_fft = 2048;
    int hop = 256;
    int n_mels = 256;
    float log_offset = 1e-8f;
};

class MelStream {
public:
    explicit MelStream(const MelConfig& cfg = {});

    void push(const float* samples, size_t n);
    void finish();  // no more samples: pad the end with zeros

    // Writes the next frame (n_mels floats) to out and returns true, or
    // returns false if the next frame still needs more samples.
    bool next_frame(float* out);

    size_t frames_done() const { return next_; }
    size_t samples_seen() const { return base_ + samples_.size(); }
    const MelConfig& config() const { return cfg_; }

    // Exposed for tests.
    const std::vector<float>& filterbank() const { return fb_; }  // [n_freqs][n_mels]
    static void fft(std::vector<std::complex<double>>& a);       // in place, size power of 2

private:
    void compute(size_t t, float* out);

    MelConfig cfg_;
    int n_freqs_;
    std::vector<double> window_;
    std::vector<float> fb_;              // [n_freqs][n_mels], torchaudio layout
    std::vector<int> fb_lo_, fb_hi_;     // nonzero freq range per mel band
    std::vector<float> samples_;         // samples [base_, base_ + size)
    size_t base_ = 0;
    size_t next_ = 0;                    // next frame index to produce
    bool finished_ = false;
    std::vector<std::complex<double>> buf_;
    std::vector<double> power_;
};

}  // namespace pf
