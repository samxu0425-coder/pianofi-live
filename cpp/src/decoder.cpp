#include "pf/decoder.h"

#include <algorithm>

namespace pf {

NoteDecoder::NoteDecoder(const DecoderConfig& cfg) : cfg_(cfg), st_(cfg.num_note) {}

void NoteDecoder::add_frames(const float* onset, const float* offset, const float* mpe,
                             const int64_t* velocity, size_t count) {
    const size_t k = count * cfg_.num_note;
    onset_.insert(onset_.end(), onset, onset + k);
    offset_.insert(offset_.end(), offset, offset + k);
    mpe_.insert(mpe_.end(), mpe, mpe + k);
    velocity_.insert(velocity_.end(), velocity, velocity + k);
    n_ += count;
    for (int j = 0; j < cfg_.num_note; ++j) advance(j);
}

void NoteDecoder::finish() {
    finished_ = true;
    for (int j = 0; j < cfg_.num_note; ++j) advance(j);
}

// Python: a frame i with a[i] >= thr is a peak if, scanning left, the first
// different value is smaller (or there is none) and the same scanning right.
void NoteDecoder::scan_peaks(int j, const std::vector<float>& a, float thr, size_t& scan,
                             std::vector<Peak>& out) {
    while (scan < n_) {
        const size_t i = scan;
        const float v = at(a, i, j);
        if (v >= thr) {
            bool left = true;
            for (size_t ii = i; ii-- > 0;) {
                const float u = at(a, ii, j);
                if (v > u) { left = true; break; }
                if (v < u) { left = false; break; }
            }
            bool right = true, decided = false;
            for (size_t ii = i + 1; ii < n_; ++ii) {
                const float u = at(a, ii, j);
                if (v > u) { right = true; decided = true; break; }
                if (v < u) { right = false; decided = true; break; }
            }
            if (!decided && !finished_) return;  // plateau runs to the newest frame: wait
            if (left && right) {
                // Python: no interpolation at the first or the last frame.
                const double h = cfg_.hop_sec;
                double time = double(i) * h;
                if (i != 0 && i + 1 < n_) {
                    const double p = at(a, i - 1, j), c = v, q = at(a, i + 1, j);
                    if (p == q) time = double(i) * h;
                    else if (p > q) time = double(i) * h - h * 0.5 * (p - q) / (c - q);
                    else time = double(i) * h + h * 0.5 * (q - p) / (c - p);
                }
                out.push_back({i, time});
            }
        }
        ++scan;
    }
}

void NoteDecoder::advance(int j) {
    PitchState& s = st_[j];
    const size_t before = s.onsets.size();
    scan_peaks(j, onset_, cfg_.thr_onset, s.onset_scan, s.onsets);
    scan_peaks(j, offset_, cfg_.thr_offset, s.offset_scan, s.offsets);
    if (note_on_) {
        for (size_t k = before; k < s.onsets.size(); ++k) {
            const Peak& p = s.onsets[k];
            const int vel = int(velocity_[p.loc * cfg_.num_note + j]);
            if (vel > 0) note_on_(Note{j + cfg_.note_min, p.time, p.time, vel}, p.loc);
        }
    }
    while (try_finish_note(j)) {}
}

bool NoteDecoder::try_finish_note(int j) {
    PitchState& s = st_[j];
    if (s.next_onset >= s.onsets.size()) return false;
    const Peak on = s.onsets[s.next_onset];
    const size_t i = on.loc;
    const double h = cfg_.hop_sec;

    // loc_next / time_next: next decided onset, or the end of the stream.
    bool have_next = s.next_onset + 1 < s.onsets.size();
    size_t loc_next = 0;
    double time_next = 0;
    if (have_next) {
        loc_next = s.onsets[s.next_onset + 1].loc;
        time_next = s.onsets[s.next_onset + 1].time;
    } else if (finished_) {
        have_next = true;
        loc_next = n_;
        time_next = double(n_ - 1) * h;
    }

    // First frame in (i, loc_next) with mpe < thr. Without a known next
    // onset, a drop at m only counts once onsets are decided through m (so no
    // onset can appear in (i, m]).
    const size_t mpe_limit = have_next ? loc_next : std::min(n_, s.onset_scan);
    long long m = -1;
    for (size_t ii = i + 1; ii < mpe_limit; ++ii) {
        if (at(mpe_, ii, j) < cfg_.thr_frame) { m = (long long)ii; break; }
    }
    // The end rule also depends on the first offset peak after i, up to the
    // frame where the note ends. Offsets must be decided that far.
    const size_t end_loc = m >= 0 ? size_t(m) : loc_next;
    if (m < 0 && !have_next) return false;
    if (s.offset_scan <= end_loc && !finished_) return false;

    while (s.offset_cursor < s.offsets.size() && s.offsets[s.offset_cursor].loc <= i) ++s.offset_cursor;
    const bool flag_offset = s.offset_cursor < s.offsets.size();
    size_t loc_offset = i + 1;
    double time_offset = 0;
    if (flag_offset) {
        loc_offset = s.offsets[s.offset_cursor].loc;
        time_offset = s.offsets[s.offset_cursor].time;
    }
    // Cap at the next onset. Only matters when it is known: with an mpe drop
    // at m < loc_next, the "shorter" rule picks the drop anyway.
    if (have_next && loc_offset > loc_next) {
        loc_offset = loc_next;
        time_offset = time_next;
    }
    const bool flag_mpe = m >= 0;
    double off;
    if (!flag_offset && !flag_mpe) off = time_next;
    else if (flag_offset && !flag_mpe) off = time_offset;
    else if (!flag_offset && flag_mpe) off = double(m) * h;
    else off = (loc_offset <= size_t(m)) ? time_offset : double(m) * h;

    const int vel = int(velocity_[i * cfg_.num_note + j]);
    if (vel > 0) {
        Note note{j + cfg_.note_min, on.time, off, vel};
        // Python trims the previous note of this pitch if the new one starts
        // before it ends.
        if (s.last_note >= 0 && note.onset < notes_[size_t(s.last_note)].offset)
            notes_[size_t(s.last_note)].offset = note.onset;
        notes_.push_back(note);
        s.last_note = (long long)notes_.size() - 1;
        if (note_off_) note_off_(note);
    }
    ++s.next_onset;
    return true;
}

std::vector<Note> NoteDecoder::notes() const {
    std::vector<Note> out;
    for (const Note& n : notes_)
        if (n.offset - n.onset >= cfg_.min_duration) out.push_back(n);
    std::stable_sort(out.begin(), out.end(), [](const Note& a, const Note& b) { return a.pitch < b.pitch; });
    std::stable_sort(out.begin(), out.end(), [](const Note& a, const Note& b) { return a.onset < b.onset; });
    return out;
}

}  // namespace pf
