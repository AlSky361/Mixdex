#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mixdex::detail {

// Every feature weight is bounded by this power of two.
inline int weight_bound_exponent(double inv_weight_sum, bool weighted) noexcept {
    if (!weighted)
        return 0;
    const double below = std::nextafter(inv_weight_sum, 0.0);
    assert(below > 0.0);
    int exponent = 0;
    std::frexp(below, &exponent);
    return 1 - exponent;
}

// count fits 32 bits, so count * 2^exponent is exact whenever finite.
inline bool dyadic_sum_fits(std::size_t count, int exponent, double inv_weight_sum) noexcept {
    const double upper = std::ldexp(static_cast<double>(count), exponent);
    return std::isfinite(upper) && std::isfinite(upper * inv_weight_sum);
}

// The extra binary order also covers internal node distances used by radii.
inline bool distance_range_fits(double max_delta,
                                std::size_t dimensions,
                                double inv_weight_sum,
                                bool weighted) noexcept {
    if (!std::isfinite(max_delta))
        return false;
    // At most 31 count bits + 258 difference + 257 weight + 256 inverse bits.
    if (max_delta <= 0x1p256 && inv_weight_sum >= 0x1p-256 && inv_weight_sum <= 0x1p256)
        return true;
    int exponent = 0;
    std::frexp(std::max(1.0, max_delta), &exponent);
    if (exponent + 1 >= std::numeric_limits<double>::max_exponent)
        return false;
    return dyadic_sum_fits(
        dimensions, exponent + 1 + weight_bound_exponent(inv_weight_sum, weighted), inv_weight_sum);
}

// Check the remaining category additions before returning a numeric cutoff.
inline bool distance_tail_fits(double numeric_sum,
                               std::size_t categories,
                               double inv_weight_sum,
                               bool weighted) noexcept {
    int exponent = 0;
    std::frexp(numeric_sum, &exponent);
    return dyadic_sum_fits(categories + 1,
                           std::max(exponent, weight_bound_exponent(inv_weight_sum, weighted)),
                           inv_weight_sum);
}

}  // namespace mixdex::detail
