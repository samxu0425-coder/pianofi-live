#include "pf/mel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {
namespace {

constexpr double kPi = 3.14159265358979323846;

double hz_to_mel_htk(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double mel_to_hz_htk(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

}  // namespace

MelStream::MelStream(const MelConfig& cfg) : cfg_(cfg) {
    if (cfg_.n_fft & (cfg_.n_fft - 1)) throw std::invalid_argument("n_fft must be a power of 2");
    n_freqs_ = cfg_.n_fft / 2 + 1;

    // Periodic Hann window (torch.hann_window default).
    window_.resize(cfg_.n_fft);
    for (int n = 0; n < cfg_.n_fft; ++n)
        window_[n] = 0.5 - 0.5 * std::cos(2.0 * kPi * n / cfg_.n_fft);

    // torchaudio.functional.melscale_fbanks(n_freqs, 0, sr/2, n_mels, sr,
    // norm="slaney", mel_scale="htk").
    const int M = cfg_.n_mels;
    std::vector<double> all_freqs(n_freqs_);
    const double f_max = cfg_.sample_rate / 2;  // sample_rate // 2 in torchaudio
    for (int i = 0; i < n_freqs_; ++i) all_freqs[i] = f_max * i / (n_freqs_ - 1);
    const double m_min = hz_to_mel_htk(0.0), m_max = hz_to_mel_htk(f_max);
    std::vector<double> f_pts(M + 2);
    for (int i = 0; i < M + 2; ++i) f_pts[i] = mel_to_hz_htk(m_min + (m_max - m_min) * i / (M + 1));

    fb_.assign(size_t(n_freqs_) * M, 0.0f);
    fb_lo_.assign(M, n_freqs_);
    fb_hi_.assign(M, 0);
    for (int m = 0; m < M; ++m) {
        const double f_diff_lo = f_pts[m + 1] - f_pts[m];
        const double f_diff_hi = f_pts[m + 2] - f_pts[m + 1];
        const double enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
        for (int k = 0; k < n_freqs_; ++k) {
            const double down = (all_freqs[k] - f_pts[m]) / f_diff_lo;      // -slopes[:, :-2] / f_diff[:-1]
            const double up = (f_pts[m + 2] - all_freqs[k]) / f_diff_hi;     // slopes[:, 2:] / f_diff[1:]
            const double v = std::max(0.0, std::min(down, up)) * enorm;
            fb_[size_t(k) * M + m] = float(v);
            if (v > 0) {
                fb_lo_[m] = std::min(fb_lo_[m], k);
                fb_hi_[m] = std::max(fb_hi_[m], k + 1);
            }
        }
    }

    buf_.resize(cfg_.n_fft);
    power_.resize(n_freqs_);
    samples_.reserve(size_t(cfg_.sample_rate) * 60);
}

void MelStream::fft(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {  // bit reversal
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * kPi / double(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const auto u = a[i + k];
                const auto v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

void MelStream::push(const float* samples, size_t n) {
    if (finished_) throw std::logic_error("push after finish");
    // Drop samples no future frame needs, so memory stays bounded.
    const long long keep_from = (long long)next_ * cfg_.hop - cfg_.n_fft / 2;
    if (keep_from > (long long)base_ + (long long)cfg_.sample_rate * 10) {
        const size_t drop = size_t(keep_from) - base_;
        samples_.erase(samples_.begin(), samples_.begin() + drop);
        base_ += drop;
    }
    samples_.insert(samples_.end(), samples, samples + n);
}

void MelStream::finish() { finished_ = true; }

bool MelStream::next_frame(float* out) {
    const size_t total = base_ + samples_.size();
    const size_t center = next_ * size_t(cfg_.hop);
    if (finished_) {
        if (next_ > total / size_t(cfg_.hop)) return false;  // 1 + N/hop frames in all
    } else if (center + cfg_.n_fft / 2 > total) {
        return false;
    }
    compute(next_, out);
    ++next_;
    return true;
}

void MelStream::compute(size_t t, float* out) {
    const long long start = (long long)t * cfg_.hop - cfg_.n_fft / 2;
    const long long total = (long long)(base_ + samples_.size());
    for (int n = 0; n < cfg_.n_fft; ++n) {
        const long long idx = start + n;
        double x = 0.0;  // pad_mode="constant": zeros outside the signal
        if (idx >= 0 && idx < total) {
            if (idx < (long long)base_) throw std::logic_error("sample already dropped");
            x = samples_[size_t(idx - (long long)base_)];
        }
        buf_[n] = std::complex<double>(x * window_[n], 0.0);
    }
    fft(buf_);
    for (int k = 0; k < n_freqs_; ++k) power_[k] = std::norm(buf_[k]);
    const int M = cfg_.n_mels;
    for (int m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int k = fb_lo_[m]; k < fb_hi_[m]; ++k) acc += power_[k] * fb_[size_t(k) * M + m];
        out[m] = float(std::log(float(acc) + cfg_.log_offset));
    }
}

}  // namespace pf
