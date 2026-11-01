#pragma once
#include <cassert>
#include <span>
#include <stdexcept>
#include <vector>

#include "mixdex/types.hpp"

namespace mixdex {

enum class QuantileMethod { InvertedCDF, MedianUnbiased };

/// Compute quantiles from sorted input.
/// Input length must fit int; probabilities must be finite and in [0, 1].
std::vector<double> quantiles(std::span<const double> x_sorted,
                              const std::vector<double>& ps,
                              QuantileMethod method);

/// Match sorted-input quantiles without full sorting; x_work is reordered.
/// Expected linear work per requested order statistic.
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
    assert(false && "method_for: неизвестный NumericType");
    return QuantileMethod::MedianUnbiased;
}

}  // namespace mixdex
