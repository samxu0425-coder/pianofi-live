// Live transcription engine.
//
//   audio thread ──push──▶ ring buffer ──pop──▶ inference thread
//   (paced like a          (AudioRingBuffer)    mel frames ─▶ frame model ─▶ h cache
//    sound card)                                h cache window ─▶ time model ─▶ note decoder
//
// The audio thread plays a wav in 10 ms blocks at real-time speed (or as fast
// as possible with realtime=false, for accuracy runs). It never blocks: if
// the ring buffer is full, samples are dropped and counted as overruns, like
// a real audio callback.
#pragma once

#include <string>
#include <vector>

#include "pf/decoder.h"
#include "pf/model.h"

namespace pf {

struct EngineConfig {
    int stride = 32;              // output frames per pass (16 ms each)
    int delay = 0;                // emit output this many frames back from the newest (B heads only)
    char heads = 'b';             // 'b': time model (what batch uses), 'a': per-frame heads only
    bool cache = true;            // reuse per-frame h across passes
    bool realtime = true;         // pace the audio thread like a sound card
    bool events = false;          // print one JSON line per note event to stdout
    int block = 160;              // samples per audio callback (10 ms)
    size_t ring_capacity = 1 << 15;
    std::string dump_dir;         // if set, write raw outputs for py/check_decoder.py
};

struct Metrics {
    std::string provider;
    int stride = 0;
    int delay = 0;
    char heads = 'b';
    bool cache = true;
    bool realtime = true;
    double audio_sec = 0;
    size_t passes = 0;
    size_t notes = 0;
    size_t overrun_samples = 0;
    double wall_sec = 0;
    double compute_sec = 0;
    std::vector<double> frame_ms, time_ms, decode_ms, pass_ms;  // per pass
    std::vector<double> frame_lag_ms;   // per output frame: output ready - its audio arrived
    std::vector<double> note_lag_ms;    // per note-on: decided - its onset audio arrived
    std::vector<double> push_jitter_ms; // audio thread: actual push - scheduled push
};

class Engine {
public:
    Engine(Model& model, const std::vector<float>& sv, const EngineConfig& cfg);
    // samples: 16 kHz mono. Returns final notes; fills metrics.
    std::vector<Note> run(const std::vector<float>& samples, Metrics& metrics);

private:
    Model& model_;
    std::vector<float> sv_;
    EngineConfig cfg_;
};

}  // namespace pf
