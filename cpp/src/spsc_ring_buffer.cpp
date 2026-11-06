// Lock-free single-producer / single-consumer ring buffer.
// Only compiled with -DPF_USE_SPSC=ON.
//
// head_ and tail_ count forever and are never wrapped by hand: `& mask_` turns a
// count into an array index, and head - tail is the number of stored samples
// even if size_t overflows, because unsigned arithmetic wraps around.
#include "pf/ring_buffer.h"

namespace pf {

size_t SpscRingBuffer::push(const float* src, size_t n) {
    // Relaxed: only this thread writes head_, so it already knows the value.
    const size_t head = head_.load(std::memory_order_relaxed);
    // Acquire: pairs with pop's release, so the slots it freed are really free.
    const size_t tail = tail_.load(std::memory_order_acquire);

    const size_t free_space = buf_.size() - (head - tail);
    if (n > free_space) n = free_space;  // full: write what fits, never wait
    for (size_t i = 0; i < n; ++i) buf_[(head + i) & mask_] = src[i];

    // Release: the consumer can't see the new head before the samples above.
    head_.store(head + n, std::memory_order_release);
    return n;
}

size_t SpscRingBuffer::pop(float* dst, size_t n) {
    // The mirror image of push, with the two counters swapped.
    const size_t tail = tail_.load(std::memory_order_relaxed);
    const size_t head = head_.load(std::memory_order_acquire);

    const size_t available = head - tail;
    if (n > available) n = available;  // empty: read what's there, never wait
    for (size_t i = 0; i < n; ++i) dst[i] = buf_[(tail + i) & mask_];

    tail_.store(tail + n, std::memory_order_release);
    return n;
}

size_t SpscRingBuffer::size() const {
    // Read tail first. head can only have grown since, so head - tail never goes
    // negative. The result can be briefly out of date, as the interface allows.
    const size_t tail = tail_.load(std::memory_order_acquire);
    const size_t head = head_.load(std::memory_order_acquire);
    return head - tail;
}

}  // namespace pf
