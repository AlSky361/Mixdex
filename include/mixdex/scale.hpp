#pragma once
#include <cstddef>
#include <span>
#include <vector>

#include "mixdex/types.hpp"

namespace mixdex {

/// Use the first width exceeding kScaleRelEps * max(abs(min), abs(max)).
/// Dataset rejects constants; scaling also rejects numerically unusable widths.
enum class ScaleLevel : int {
    Iqr = 0,    // Interquartile range: q75 - q25.
    Idr = 1,    // Interdecile range: q90 - q10.
    Range = 2,  // Full range: max - min.
};

struct ScaleParams {
    double median;
    double scale;  // Selected interval width.
    ScaleLevel level;
};

/// Fit a positive scale; reject empty, nonfinite, constant or numerically degenerate input.
ScaleParams fit_scale_params(std::span<const double> values, NumericType type);

std::vector<double> scale_feature(std::span<const double> values, NumericType type);

/// Own fitted scales; apply the same transform to stored and external rows.
class Scaler {
public:
    Scaler() = default;

    static Scaler fit(const Dataset& data);

    Dataset transform(const Dataset& data) const;

    /// Transform m row-major numeric rows using the fitted schema.
    std::vector<double> transform_rows(std::span<const double> num, std::size_t m) const;

    Idx n_num() const noexcept {
        return static_cast<Idx>(median_.size());
    }
    const std::vector<double>& medians() const noexcept {
        return median_;
    }
    // Selected feature widths; not necessarily IQR.
    const std::vector<double>& scales() const noexcept {
        return scale_;
    }
    const std::vector<ScaleLevel>& levels() const noexcept {
        return level_;
    }

private:
    std::vector<double> median_;
    std::vector<double> scale_;
    std::vector<ScaleLevel> level_;
};

Dataset scale(const Dataset& data);

}  // namespace mixdex
