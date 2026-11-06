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
namespace {

std::runtime_error audio_error(const std::string& what, ma_result result) {
    return std::runtime_error(what + ": " + ma_result_description(result));
}

// The devices a Capture can open: outputs for loopback, microphones otherwise.
// The list belongs to the context and is valid until the context is used again.
void list_devices(ma_context& context, bool loopback, ma_device_info*& infos, ma_uint32& count) {
    ma_device_info *outputs, *inputs;
    ma_uint32 n_outputs, n_inputs;
    const ma_result result = ma_context_get_devices(&context, &outputs, &n_outputs, &inputs, &n_inputs);
    if (result != MA_SUCCESS) throw audio_error("can't list audio devices", result);
    infos = loopback ? outputs : inputs;
    count = loopback ? n_outputs : n_inputs;
}

}  // namespace

struct Capture::Impl {
    ma_context context;
    ma_device device;
    // Loopback only. Windows delivers no loopback samples while nothing on the PC
    // is playing, which would make the engine's sample clock fall behind real
    // time. Playing silence ourselves keeps the samples coming.
    ma_device silence;
    bool has_silence = false;
    Callback callback;
    void* user;
};

Capture::Capture(bool loopback, const std::string& device, int sample_rate, Callback callback, void* user)
    : impl_(std::make_unique<Impl>()) {
    impl_->callback = callback;
    impl_->user = user;
    // On Windows miniaudio tries WASAPI first, and loopback only exists on WASAPI.
    ma_result result = ma_context_init(nullptr, 0, nullptr, &impl_->context);
    if (result != MA_SUCCESS) throw audio_error("can't start the audio system", result);

    try {
        // Find the device by name. No name means the Windows default, which is a null id.
        ma_device_id id{};
        const ma_device_id* chosen = nullptr;
        if (!device.empty()) {
            ma_device_info* infos;
            ma_uint32 count;
            list_devices(impl_->context, loopback, infos, count);
            for (ma_uint32 i = 0; i < count && !chosen; ++i) {
                if (std::string(infos[i].name).find(device) != std::string::npos) {
                    id = infos[i].id;
                    chosen = &id;
                }
            }
            if (!chosen)
                throw std::runtime_error(std::string("no ") + (loopback ? "output device" : "microphone") + " has \"" +
                                         device + "\" in its name. --list-devices shows the names.");
        }

        ma_device_config config = ma_device_config_init(loopback ? ma_device_type_loopback : ma_device_type_capture);
        config.capture.pDeviceID = chosen;
        config.capture.format = ma_format_f32;
        config.capture.channels = 1;
        config.sampleRate = ma_uint32(sample_rate);
        config.periodSizeInMilliseconds = 10;
        config.dataCallback = [](ma_device* dev, void*, const void* input, ma_uint32 frames) {
            auto* impl = static_cast<Impl*>(dev->pUserData);
            impl->callback(static_cast<const float*>(input), frames, impl->user);
        };
        config.pUserData = impl_.get();
        result = ma_device_init(&impl_->context, &config, &impl_->device);
        if (result != MA_SUCCESS) throw audio_error("can't open audio device", result);

        if (loopback) {
            ma_device_config silent = ma_device_config_init(ma_device_type_playback);
            silent.playback.pDeviceID = chosen;  // the same output we are listening to
            silent.dataCallback = [](ma_device*, void*, const void*, ma_uint32) {};  // miniaudio hands us a zeroed buffer
            result = ma_device_init(&impl_->context, &silent, &impl_->silence);
            if (result != MA_SUCCESS) {
                ma_device_uninit(&impl_->device);
                throw audio_error("can't open the silent playback stream", result);
            }
            impl_->has_silence = true;
        }
    } catch (...) {
        ma_context_uninit(&impl_->context);
        throw;
    }
}

Capture::~Capture() {
    ma_device_uninit(&impl_->device);
    if (impl_->has_silence) ma_device_uninit(&impl_->silence);
    ma_context_uninit(&impl_->context);
}

std::vector<std::string> Capture::device_names(bool loopback) {
    ma_context context;
    const ma_result result = ma_context_init(nullptr, 0, nullptr, &context);
    if (result != MA_SUCCESS) throw audio_error("can't start the audio system", result);
    std::vector<std::string> names;
    try {
        ma_device_info* infos;
        ma_uint32 count;
        list_devices(context, loopback, infos, count);
        for (ma_uint32 i = 0; i < count; ++i)
            names.push_back(std::string(infos[i].name) + (infos[i].isDefault ? "  (default)" : ""));
    } catch (...) {
        ma_context_uninit(&context);
        throw;
    }
    ma_context_uninit(&context);
    return names;
}

void Capture::start() {
    ma_result result = impl_->has_silence ? ma_device_start(&impl_->silence) : MA_SUCCESS;
    if (result == MA_SUCCESS) result = ma_device_start(&impl_->device);
    if (result != MA_SUCCESS) throw audio_error("can't start audio device", result);
}

void Capture::stop() {
    ma_device_stop(&impl_->device);
    if (impl_->has_silence) ma_device_stop(&impl_->silence);
}

int Capture::sample_rate() const { return int(impl_->device.sampleRate); }

std::string Capture::description() const {
    const auto& capture = impl_->device.capture;
    return std::string(capture.name) + " (" + ma_get_backend_name(impl_->device.pContext->backend) + ", device format " +
           std::to_string(capture.internalSampleRate) + " Hz, " + std::to_string(capture.internalChannels) + " channels)";
}

}  // namespace pf
