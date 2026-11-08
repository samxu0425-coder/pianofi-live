// A count of engines running on this machine, so two don't share one GPU by accident.
//
// Engines take numbered slot files in the temp folder and lock each exclusively
// (LockFileEx on Windows, flock on Linux). The OS drops a lock when its process
// ends, however it ends, so a crashed engine never leaves a stale slot. An engine
// that finds all `max_engines` slots taken refuses to start. Containers have their
// own temp folders, so engines in different containers never see each other: in
// the cloud the container scheduler is what limits engines per GPU.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace pf {

class EngineLock {
public:
    // max_engines <= 0 means no limit. Throws when every slot is taken.
    explicit EngineLock(int max_engines) {
        if (max_engines <= 0) return;
        const auto dir = std::filesystem::temp_directory_path() / "pianofi_live_slots";
        std::filesystem::create_directories(dir);
        for (int slot = 0; slot < max_engines; ++slot) {
            if (try_lock((dir / ("slot-" + std::to_string(slot))).string())) return;
        }
        throw std::runtime_error(
            "another pianofi_live is already running on this machine (" + std::to_string(max_engines) +
            " allowed). One engine per GPU keeps the timings honest; pass --max-engines N to allow more, or 0 for no limit.");
    }

    ~EngineLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);  // also releases the lock
#else
        if (fd_ >= 0) close(fd_);  // also releases the flock
#endif
    }

    EngineLock(const EngineLock&) = delete;
    EngineLock& operator=(const EngineLock&) = delete;

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;

    bool try_lock(const std::string& path) {
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        OVERLAPPED ov{};
        if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) {
            CloseHandle(h);
            return false;
        }
        handle_ = h;
        return true;
    }
#else
    int fd_ = -1;

    bool try_lock(const std::string& path) {
        int fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd < 0) return false;
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd);
            return false;
        }
        fd_ = fd;
        return true;
    }
#endif
};

}  // namespace pf
