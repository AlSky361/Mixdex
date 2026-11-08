#pragma once
#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mixdex/utils/aliases.hpp"

namespace mixdex::detail {

inline Idx checked_index_size(std::size_t size,
                              const char* message = "size exceeds the index range") {
    if (!std::in_range<Idx>(size))
        throw std::length_error(message);
    return static_cast<Idx>(size);
}

inline std::size_t checked_size_product(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("buffer size overflow");
    return a * b;
}

inline void add_count(Count& value, Count delta = 1) {
    if (delta > std::numeric_limits<Count>::max() - value)
        throw std::overflow_error("counter overflow");
    value += delta;
}

inline void increment_count(std::atomic<Count>& value) {
    Count current = value.load(std::memory_order_relaxed);
    do {
        if (current == std::numeric_limits<Count>::max())
            throw std::overflow_error("counter overflow");
    } while (!value.compare_exchange_weak(current, current + 1,
                                          std::memory_order_relaxed));
}

} // namespace mixdex::detail
