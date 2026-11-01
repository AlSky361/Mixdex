#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mixdex::constants {

constexpr double kEpsZero = 1e-12;  // Default absolute comparison tolerance.
constexpr double kEpsEq = 1e-9;     // Default relative comparison tolerance.

constexpr double kInf = std::numeric_limits<double>::infinity();

inline constexpr double kZeroLbEps = 1e-9;  // Counts near-zero bounds in MST diagnostics.

inline constexpr double kLofLrdRegularizer = 1e-10;  // Avoids a zero denominator in LOF density.

// Round-to-nearest uses u = epsilon/2; gamma_m = m*u/(1-m*u) requires m*u < 1.
// Higham, Accuracy and Stability of Numerical Algorithms (2002):
// https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/
inline constexpr double kUnitRoundoff = std::numeric_limits<double>::epsilon() / 2.0;

// Project operation allowances beyond the feature count in rounding guards.
inline constexpr int kBallBoundRoundingOps = 6;  // Point-to-node bounds.
inline constexpr int kNodePairRoundingOps = 8;   // Node-to-node bounds.

inline constexpr int kKUnset = -1;  // Requests automatic k selection in Approximate mode.

// Grow the smallest connected k by this factor, capped at k_max.
inline constexpr double kApproxKMargin = 2.0;  // Project margin for Approximate mode.

inline constexpr int kNoiseLabel = -1;  // DBSCAN/HDBSCAN noise.
inline constexpr int kUnlabeled = -1;   // Unlabeled kNN input.

// Hyndman-Fan (1996), type 8: m(p) = (p + 1) / 3.
// https://doi.org/10.1080/00031305.1996.10473566
inline constexpr double kQuantileType8Divisor = 3.0;
inline const std::vector<double> kQuartileProbs{0.25, 0.5, 0.75};  // Quartile probability levels.

// Scaling tries Q75-Q25 (interquartile), Q90-Q10 (interdecile), then the full range.
// Quantile order: min, Q10, Q25, median, Q75, Q90, max.
inline const std::vector<double> kScaleProbs{0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0};

// Accept a width only if it exceeds kScaleRelEps * max(abs(min), abs(max)).
// This relative test detects numerically unusable scales, including nonconstant data.
inline constexpr double kScaleRelEps = 1e-12;  // Project tolerance, not machine epsilon.

inline constexpr std::size_t kCacheLineBytes = 64;  // Chosen alignment for per-thread storage.
inline constexpr double kPercent = 100.0;  // Converts a fraction to a percentage in diagnostics.
inline constexpr int kPairHashShift = 32;  // Separates vertex indices in edge-pair hashes.

// Initial node reservation is factor * estimated leaves + slack; storage can grow.
inline constexpr int kTreeNodesPerLeafBound = 2;  // Reservation factor.
inline constexpr int kTreeNodesReserveSlack = 4;  // Extra reserved nodes.

constexpr int kDefaultLeafSize = 40;  // Default leaf capacity for indexed search.

// Approximate comparisons use project tolerances, not exact or bitwise equality.
// C++20 does not guarantee constexpr std::fabs; keep header definitions inline.
inline bool f_equal(double a, double b, double rel_eps = kEpsEq, double abs_eps = kEpsZero) {
    double diff = std::fabs(a - b);
    double scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= std::max(rel_eps * scale, abs_eps);
}

inline bool f_zero(double a, double eps = kEpsZero) {
    return std::fabs(a) <= eps;
}

}  // namespace mixdex::constants
