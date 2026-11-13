// ONNX Runtime wrapper for the two graphs written by py/export_onnx.py.
//
//   frame.onnx  spec [1, 256, S + 64], sv [1, 24]
//               -> h [S, 88, 256], onset_a/offset_a/mpe_a [S, 88], velocity_a [S, 88] int64
//   time.onnx   h [512, 88, 256] -> onset/offset/mpe [512, 88], velocity [512, 88] int64
//
// All outputs are written straight into caller-owned buffers (no copies).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace pf {

constexpr int kBins = 256;
constexpr int kNotes = 88;
constexpr int kHid = 256;
constexpr int kMargin = 32;
constexpr int kWindow = 512;
constexpr int kStyleDim = 24;
constexpr size_t kHSize = size_t(kNotes) * kHid;  // floats of h per frame

struct ModelOptions {
    std::string frame_path;
    std::string time_path;
    bool cuda = false;
    int threads = 0;  // 0 = ONNX Runtime default
    // Not empty: ONNX Runtime writes a per-node timing profile for each graph when the
    // model is destroyed, to <profile>_frame_<time>.json and <profile>_time_<time>.json.
    std::string profile;
    // TensorRT provider in front of CUDA (needs cuda = true and the TensorRT 10 DLLs on PATH).
    // Engines are built on the first run and kept in trt_cache; a build takes minutes.
    bool trt = false;
    bool trt_fp16 = false;       // let TensorRT pick FP16 layers (it takes the FP32 graphs)
    int trt_opt_frames = 80;     // the usual frame.onnx input length: stride + 64
    int trt_max_frames = 608;    // the longest: the first pass, stride + delay + 96
    std::string trt_cache;
};

class Model {
public:
    explicit Model(const ModelOptions& opt);
    ~Model();

    // spec: [256][in_frames] (bin-major, like the model input), in_frames = S + 64.
    // Outputs: h [S][88*256], A heads [S][88] (any of them may be nullptr... except h).
    void run_frame(const float* spec, int in_frames, const float* sv, float* h,
                   float* onset_a, float* offset_a, float* mpe_a, int64_t* velocity_a);

    // h: [512][88*256] contiguous. Outputs [512][88].
    void run_time(const float* h, float* onset, float* offset, float* mpe, int64_t* velocity);

    const char* provider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pf
