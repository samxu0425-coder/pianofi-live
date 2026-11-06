#include "pf/capture.h"

#include <stdexcept>

// Leave out the parts of miniaudio this project doesn't use.
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)  // third-party code: don't report its warnings
#endif
#include "miniaudio.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace pf {

struct Capture::Impl {
    ma_device device;
    Callback callback;
    void* user;
};

Capture::Capture(bool loopback, Callback callback, void* user) : impl_(std::make_unique<Impl>()) {
    impl_->callback = callback;
    impl_->user = user;

    ma_device_config config = ma_device_config_init(loopback ? ma_device_type_loopback : ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = 16000;
    config.periodSizeInMilliseconds = 10;
    config.dataCallback = [](ma_device* device, void*, const void* input, ma_uint32 frames) {
        auto* impl = static_cast<Impl*>(device->pUserData);
        impl->callback(static_cast<const float*>(input), frames, impl->user);
    };
    config.pUserData = impl_.get();
    // On Windows miniaudio tries WASAPI first, and loopback only exists on WASAPI.
    const ma_result result = ma_device_init(nullptr, &config, &impl_->device);
    if (result != MA_SUCCESS) throw std::runtime_error(std::string("can't open audio device: ") + ma_result_description(result));
}

Capture::~Capture() { ma_device_uninit(&impl_->device); }

void Capture::start() {
    const ma_result result = ma_device_start(&impl_->device);
    if (result != MA_SUCCESS) throw std::runtime_error(std::string("can't start audio device: ") + ma_result_description(result));
}

void Capture::stop() { ma_device_stop(&impl_->device); }

std::string Capture::description() const {
    const auto& capture = impl_->device.capture;
    return std::string(capture.name) + " (" + ma_get_backend_name(impl_->device.pContext->backend) + ", device format " +
           std::to_string(capture.internalSampleRate) + " Hz, " + std::to_string(capture.internalChannels) + " channels)";
}

}  // namespace pf
