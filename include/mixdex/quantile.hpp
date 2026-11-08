#pragma once
#include <span>
#include <stdexcept>
#include <vector>

#include "mixdex/types.hpp"

namespace mixdex {

enum class QuantileMethod { InvertedCDF, MedianUnbiased };

/// Compute q quantiles in O(q); input must be sorted and finite (not scanned).
/// Input length must fit int; probabilities must be finite and in [0, 1].
/// Empty probabilities return empty; otherwise input must be nonempty.
std::vector<double> quantiles(std::span<const double> x_sorted,
                              const std::vector<double>& ps,
                              QuantileMethod method);

/// Match sorted-input quantiles without full sorting; x_work is reordered.
/// Validate finite values and all arguments before reordering.
/// Expected O(n) per order statistic plus O(q log q); O(q) extra storage.
std::vector<double> quantiles_select(std::span<double> x_work,
                                     const std::vector<double>& ps,
                                     QuantileMethod method);

constexpr QuantileMethod method_for(NumericType type) {
    switch (type) {
        case NumericType::Continuous:
            return QuantileMethod::MedianUnbiased;
        case NumericType::Discrete:
            return QuantileMethod::InvertedCDF;
    }
    throw std::invalid_argument("unknown numeric type");
}

}  // namespace mixdex
