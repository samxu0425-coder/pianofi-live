// Live audio capture with miniaudio: a microphone, or whatever the PC is
// playing (WASAPI loopback). Delivers mono float samples.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace pf {

class Capture {
public:
    // Runs on the audio thread: no allocation, no locks, no I/O.
    using Callback = void (*)(const float* samples, size_t n, void* user);

    // device: part of a device name, or empty for the Windows default. For
    //   loopback it names the output device to listen to, otherwise a microphone.
    // sample_rate 0 = the device's own rate, so nothing is resampled.
    // Throws if the device can't be found or opened.
    Capture(bool loopback, const std::string& device, int sample_rate, Callback callback, void* user);
    ~Capture();
    void start();
    void stop();  // after this returns, the callback won't run again
    int sample_rate() const;  // rate of the samples the callback gets
    std::string description() const;

    // Names of the devices --device can pick from: outputs for loopback, else microphones.
    static std::vector<std::string> device_names(bool loopback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pf
