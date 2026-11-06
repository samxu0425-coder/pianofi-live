// Live audio capture with miniaudio: the default mic, or whatever the PC is
// playing (WASAPI loopback). Always delivers 16 kHz mono float samples.
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace pf {

class Capture {
public:
    // Runs on the audio thread: no allocation, no locks, no I/O.
    using Callback = void (*)(const float* samples, size_t n, void* user);

    Capture(bool loopback, Callback callback, void* user);  // throws if the device won't open
    ~Capture();
    void start();
    void stop();  // after this returns, the callback won't run again
    std::string description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pf
