// Streaming port of AMT-APC's AMT.mpe2note (mode_offset="shorter",
// mode_velocity="ignore_zero") plus note2midi's min-duration filter.
//
// Frames arrive in order. An onset or offset peak is decided as soon as the
// first differing neighbour on each side is known (the Python version scans
// left and right until the value changes). A note is emitted as NoteOn the
// moment its onset peak is decided, and finalised (NoteOff) once the rule
// that ends it is decided. After finish(), notes() returns exactly what
// mpe2note + the min_duration filter returns on the same arrays.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace pf {

struct Note {
    int pitch = 0;       // MIDI note number
    double onset = 0;    // seconds
    double offset = 0;   // seconds
    int velocity = 0;
};

struct DecoderConfig {
    int num_note = 88;
    int note_min = 21;
    double hop_sec = 256.0 / 16000.0;
    float thr_onset = 0.5f;
    float thr_offset = 1.0f;
    float thr_frame = 0.5f;
    double min_duration = 0.08;
    // Drop frames no pending decision needs, so a long live session doesn't
    // grow without limit. Off by default: with it on, a peak that sits on a
    // run of more than kTrimSlack equal values could be judged differently.
    bool trim_history = false;
};

class NoteDecoder {
public:
    // Called when a note starts / is finalised. frame = output frame index.
    using NoteOnFn = std::function<void(const Note&, size_t frame)>;
    using NoteOffFn = std::function<void(const Note&)>;

    explicit NoteDecoder(const DecoderConfig& cfg = {});
    void on_note_on(NoteOnFn fn) { note_on_ = std::move(fn); }
    void on_note_off(NoteOffFn fn) { note_off_ = std::move(fn); }

    // Each array is [count][num_note], row-major.
    void add_frames(const float* onset, const float* offset, const float* mpe,
                    const int64_t* velocity, size_t count);
    void finish();

    size_t frames() const { return n_; }
    size_t frames_kept() const { return n_ - base_; }  // how many are still in memory
    static constexpr size_t kTrimSlack = 64;           // frames kept behind each scan position
    // Final notes, sorted by onset then pitch, min_duration filter applied.
    std::vector<Note> notes() const;

private:
    struct Peak {
        size_t loc;
        double time;
    };
    struct PitchState {
        size_t onset_scan = 0;   // all onset peaks below this index are decided
        size_t offset_scan = 0;  // same for offset peaks
        std::vector<Peak> onsets;   // decided onset peaks, in order
        std::vector<Peak> offsets;  // decided offset peaks, in order
        size_t next_onset = 0;      // index into onsets of the oldest unfinished note
        size_t offset_cursor = 0;   // offsets before this are behind every pending note
        long long last_note = -1;   // index into notes_ of the last kept note
    };

    void advance(int j);
    void scan_peaks(int j, const std::vector<float>& a, float thr, size_t& scan, std::vector<Peak>& out);
    bool try_finish_note(int j);
    void trim();
    // Frame i of an array. Frames before base_ have been dropped.
    float at(const std::vector<float>& a, size_t i, int j) const { return a[(i - base_) * cfg_.num_note + j]; }
    int velocity_at(size_t i, int j) const { return int(velocity_[(i - base_) * cfg_.num_note + j]); }

    DecoderConfig cfg_;
    size_t n_ = 0;     // frames received so far
    size_t base_ = 0;  // frames dropped from the front of the arrays
    bool finished_ = false;
    std::vector<float> onset_, offset_, mpe_;
    std::vector<int64_t> velocity_;
    std::vector<PitchState> st_;
    std::vector<Note> notes_;
    NoteOnFn note_on_;
    NoteOffFn note_off_;
};

}  // namespace pf
