#pragma once
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "mixdex/utils/aliases.hpp"
#include "mixdex/utils/omp_macros.hpp"

namespace mixdex {

enum class Schedule { Static, Dynamic, Guided };

namespace detail {

// Capture one worker exception and rethrow it after the parallel loop.
class ExceptionSink {
public:
    bool failed() const noexcept {
        return failed_.load(std::memory_order_relaxed);
    }

    template <class F>
    void run(F&& f) noexcept {
        if (failed())
            return;
        try {
            f();
        } catch (...) {
            capture();
        }
    }

    void rethrow_if_failed() {
        if (first_)
            std::rethrow_exception(first_);
    }

private:
    void capture() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_)
            first_ = std::current_exception();
        failed_.store(true, std::memory_order_relaxed);
    }

    std::mutex mutex_;  // Serializes exception capture.
    std::exception_ptr first_;
    std::atomic<bool> failed_{false};  // Skips new work; callbacks already running may finish.
};

}  // namespace detail

// Run [0, n) with signed callback indices; propagate worker exceptions after the loop.
// Negative counts and counts outside the signed index range throw.
template <Schedule S = Schedule::Dynamic, class N, class F>
void parallel_for(N n, F&& f) {
    static_assert(std::is_integral_v<N> && !std::is_same_v<N, bool>,
                  "parallel_for requires an integer count other than bool");
    using I = std::common_type_t<int, std::make_signed_t<N>>;  // Widen small counters.
    if constexpr (std::is_signed_v<N>) {
        if (n < 0)
            throw std::invalid_argument("parallel_for count must be nonnegative");
    }
    if (!std::in_range<I>(+n))
        throw std::length_error("parallel_for count exceeds the signed index range");
    const I count = static_cast<I>(n);
    detail::ExceptionSink sink;
    if constexpr (S == Schedule::Static) {
        MIXDEX_OMP_STATIC
        for (I i = 0; i < count; ++i)
            sink.run([&] {
                f(i);
            });
    } else if constexpr (S == Schedule::Guided) {
        MIXDEX_OMP_GUIDED
        for (I i = 0; i < count; ++i)
            sink.run([&] {
                f(i);
            });
    } else {
        MIXDEX_OMP_DYNAMIC
        for (I i = 0; i < count; ++i)
            sink.run([&] {
                f(i);
            });
    }
    sink.rethrow_if_failed();
}

// Partition boundary: total >= 0, parts > 0, and 0 <= part <= parts.
// Widen the product before division to avoid overflow in the index type.
inline Idx chunk_begin(Idx total, int parts, int part) {
    if (total < 0 || parts <= 0 || part < 0 || part > parts)
        throw std::invalid_argument("invalid partition bounds");
    return static_cast<Idx>(static_cast<Count>(total) * part / parts);
}

}  // namespace mixdex
