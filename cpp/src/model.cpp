#include "pf/model.h"

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace pf {

namespace {
#ifdef _WIN32
std::wstring to_ort_path(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#else
std::string to_ort_path(const std::string& s) { return s; }
#endif

// ONNX Runtime's CUDA MatMul uses NVIDIA's TF32 mode, which multiplies with
// reduced precision. That moves outputs up to 1.3e-2 away from batch (torch), so
// turn it off. Must run before the CUDA libraries load.
void disable_tf32() {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
}
}  // namespace

struct Model::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "pianofi_live"};
    Ort::SessionOptions so;
    std::unique_ptr<Ort::Session> frame, time;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    bool cuda = false;
    // Scratch for outputs the caller does not want.
    std::vector<float> scratch_f;
    std::vector<int64_t> scratch_i;
};

Model::Model(const ModelOptions& opt) : impl_(std::make_unique<Impl>()) {
    auto& so = impl_->so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (opt.threads > 0) so.SetIntraOpNumThreads(opt.threads);
    if (opt.cuda) {
        disable_tf32();
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        so.AppendExecutionProvider_CUDA(cuda_opts);  // throws if the CUDA EP is missing
        impl_->cuda = true;
    }
    impl_->frame = std::make_unique<Ort::Session>(impl_->env, to_ort_path(opt.frame_path).c_str(), so);
    impl_->time = std::make_unique<Ort::Session>(impl_->env, to_ort_path(opt.time_path).c_str(), so);
}

Model::~Model() = default;

const char* Model::provider() const { return impl_->cuda ? "CUDA" : "CPU"; }

void Model::run_frame(const float* spec, int in_frames, const float* sv, float* h, float* onset_a,
                      float* offset_a, float* mpe_a, int64_t* velocity_a) {
    auto& m = *impl_;
    const int s = in_frames - 2 * kMargin;
    if (s <= 0) throw std::invalid_argument("run_frame: need more than 64 input frames");
    const size_t n = size_t(s) * kNotes;
    if (!onset_a || !offset_a || !mpe_a || !velocity_a) {
        if (m.scratch_f.size() < 3 * n) m.scratch_f.resize(3 * n);
        if (m.scratch_i.size() < n) m.scratch_i.resize(n);
        if (!onset_a) onset_a = m.scratch_f.data();
        if (!offset_a) offset_a = m.scratch_f.data() + n;
        if (!mpe_a) mpe_a = m.scratch_f.data() + 2 * n;
        if (!velocity_a) velocity_a = m.scratch_i.data();
    }

    const std::array<int64_t, 3> spec_shape{1, kBins, in_frames};
    const std::array<int64_t, 2> sv_shape{1, kStyleDim};
    const std::array<int64_t, 3> h_shape{s, kNotes, kHid};
    const std::array<int64_t, 2> head_shape{s, kNotes};

    Ort::Value inputs[] = {
        Ort::Value::CreateTensor<float>(m.mem, const_cast<float*>(spec), size_t(kBins) * in_frames, spec_shape.data(), 3),
        Ort::Value::CreateTensor<float>(m.mem, const_cast<float*>(sv), kStyleDim, sv_shape.data(), 2),
    };
    Ort::Value outputs[] = {
        Ort::Value::CreateTensor<float>(m.mem, h, size_t(s) * kHSize, h_shape.data(), 3),
        Ort::Value::CreateTensor<float>(m.mem, onset_a, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<float>(m.mem, offset_a, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<float>(m.mem, mpe_a, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(m.mem, velocity_a, n, head_shape.data(), 2),
    };
    const char* in_names[] = {"spec", "sv"};
    const char* out_names[] = {"h", "onset_a", "offset_a", "mpe_a", "velocity_a"};
    m.frame->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, outputs, 5);
}

void Model::run_time(const float* h, float* onset, float* offset, float* mpe, int64_t* velocity) {
    auto& m = *impl_;
    const std::array<int64_t, 3> h_shape{kWindow, kNotes, kHid};
    const std::array<int64_t, 2> head_shape{kWindow, kNotes};
    const size_t n = size_t(kWindow) * kNotes;
    Ort::Value input = Ort::Value::CreateTensor<float>(m.mem, const_cast<float*>(h), kWindow * kHSize, h_shape.data(), 3);
    Ort::Value outputs[] = {
        Ort::Value::CreateTensor<float>(m.mem, onset, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<float>(m.mem, offset, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<float>(m.mem, mpe, n, head_shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(m.mem, velocity, n, head_shape.data(), 2),
    };
    const char* in_names[] = {"h"};
    const char* out_names[] = {"onset", "offset", "mpe", "velocity"};
    m.time->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, outputs, 4);
}

}  // namespace pf
