#pragma once
#include <cmath>
#include <span>
#include <stdexcept>

namespace mixdex::detail {

inline void require_finite(std::span<const double> values, const char* message) {
    for (double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(message);
}

}  // namespace mixdex::detail
