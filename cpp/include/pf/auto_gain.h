// Automatic input level for live audio.
//
// The model was trained on music at normal recorded level and finds fewer
// notes when the same music is quieter. Live audio arrives at whatever volume
// the PC or the mic happens to be at, so the engine scales it: the loudest
// sample of the last 10 s is brought to full scale (1.0).
//
// The gain is chosen once per pass. It drops at once when louder audio
// arrives, rises slowly when the audio gets quieter, and holds still during
// silence, where there is nothing to measure.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pf {

struct AutoGainConfig {
    double window_sec = 10.0;      // how far back to look for the loudest sample
    double rise_db_per_sec = 6.0;  // how fast the gain may go up
    float max_gain = 32.0f;        // +30 dB: never amplify more than this
    float silence = 1e-3f;         // below this peak there is nothing to measure
};

class AutoGain {
public:
    // chunk_sec: how much audio each add_peak() call covers.
    explicit AutoGain(double chunk_sec, const AutoGainConfig& cfg = {})
        : cfg_(cfg), window_(size_t(cfg.window_sec / chunk_sec + 0.5)) {}

    // peak: the largest |sample| in the next chunk of audio.
    void add_peak(float peak) { peaks_.push_back(peak); }

    // Gain for a pass that has heard chunks [0, end). elapsed_sec is the audio
    // time since the previous call.
    float update(size_t end, double elapsed_sec) {
        end = std::min(end, first_ + peaks_.size());
        const size_t begin = std::max(first_, end > window_ ? end - window_ : 0);
        float peak = 0.0f;
        for (size_t i = begin; i < end; ++i) peak = std::max(peak, peaks_[i - first_]);
        // Passes only move forward, so chunks before this window are never read again.
        if (begin > first_ + 4096) {
            peaks_.erase(peaks_.begin(), peaks_.begin() + (begin - first_));
            first_ = begin;
        }
        if (peak < cfg_.silence) return gain_;

        const float wanted = std::min(1.0f / peak, cfg_.max_gain);
        if (!measured_ || wanted < gain_) {
            gain_ = wanted;  // first reading, or louder audio: change at once
        } else {
            const double max_rise = std::pow(10.0, cfg_.rise_db_per_sec * elapsed_sec / 20.0);
            gain_ = std::min(wanted, float(gain_ * max_rise));
        }
        measured_ = true;
        return gain_;
    }

    float gain() const { return gain_; }

private:
    AutoGainConfig cfg_;
    size_t window_;             // window length in chunks
    std::vector<float> peaks_;  // one per chunk, for chunks [first_, first_ + size)
    size_t first_ = 0;
    float gain_ = 1.0f;
    bool measured_ = false;
};

}  // namespace pf
