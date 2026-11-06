// Ring buffers that hand audio samples from the audio thread (producer) to the
// inference thread (consumer).
//
// Two implementations with the same interface:
//   MutexRingBuffer  baseline, guarded by a std::mutex.
//   SpscRingBuffer   lock-free single-producer / single-consumer.
//
// The engine uses SpscRingBuffer, or MutexRingBuffer when built with
// -DPF_USE_SPSC=OFF. bench_ring compares the two.
//
// Interface (both classes):
//   explicit Buffer(size_t capacity)       capacity is rounded up to a power of 2
//   size_t push(const float* src, size_t n) producer only; returns samples written
//                                           (less than n when full: never blocks)
//   size_t pop(float* dst, size_t n)        consumer only; returns samples read
//                                           (less than n when empty: never blocks)
//   size_t size() const                     samples currently stored (approximate
//                                           when read from the other thread)
//   size_t capacity() const
#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace pf {

inline size_t round_up_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

class MutexRingBuffer {
public:
    explicit MutexRingBuffer(size_t capacity)
        : buf_(round_up_pow2(capacity)), mask_(buf_.size() - 1) {}

    size_t push(const float* src, size_t n) {
        std::lock_guard<std::mutex> lock(mu_);
        size_t free_space = buf_.size() - (head_ - tail_);
        if (n > free_space) n = free_space;
        for (size_t i = 0; i < n; ++i) buf_[(head_ + i) & mask_] = src[i];
        head_ += n;
        return n;
    }

    size_t pop(float* dst, size_t n) {
        std::lock_guard<std::mutex> lock(mu_);
        size_t avail = head_ - tail_;
        if (n > avail) n = avail;
        for (size_t i = 0; i < n; ++i) dst[i] = buf_[(tail_ + i) & mask_];
        tail_ += n;
        return n;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return head_ - tail_;
    }

    size_t capacity() const { return buf_.size(); }

private:
    std::vector<float> buf_;
    size_t mask_;
    size_t head_ = 0;  // total samples ever pushed
    size_t tail_ = 0;  // total samples ever popped
    mutable std::mutex mu_;
};

// Same behavior as MutexRingBuffer, no mutex. Safe for exactly one producer
// thread and one consumer thread, because each counter has a single writer.
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(size_t capacity)
        : buf_(round_up_pow2(capacity)), mask_(buf_.size() - 1) {}

    size_t push(const float* src, size_t n);
    size_t pop(float* dst, size_t n);
    size_t size() const;
    size_t capacity() const { return buf_.size(); }

private:
    std::vector<float> buf_;
    size_t mask_;
    // Separate cache lines, so the two threads don't slow each other down.
    alignas(64) std::atomic<size_t> head_{0};  // total samples ever pushed; only the producer writes it
    alignas(64) std::atomic<size_t> tail_{0};  // total samples ever popped; only the consumer writes it
};

#ifdef PF_USE_SPSC
using AudioRingBuffer = SpscRingBuffer;
#else
using AudioRingBuffer = MutexRingBuffer;
#endif

}  // namespace pf
