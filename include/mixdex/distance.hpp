#pragma once

#include "mixdex/types.hpp"

namespace mixdex {

// Kernels require valid row indices and complete queries matching the Dataset schema.
// Debug assertions check these preconditions; Python validates indices at its boundary.

/// Weighted numeric L1, scalar Hamming and set Jaccard, divided by total feature weight.
[[nodiscard]] double distance(const Dataset& data, int i, int j) noexcept;

/// Compare a complete external row using the Dataset schema and preprocessing.
[[nodiscard]] double distance(const Dataset& data, const RowView& q, int j) noexcept;

/// May return a partial distance above bound; values at or below bound are exact.
[[nodiscard]] double distance_bounded(const Dataset& data,
                                      const RowView& q,
                                      int j,
                                      double bound) noexcept;

/// Numeric contribution only, normalized by the total weight of all features.
[[nodiscard]] double distance_num_partial(const Dataset& data, int i, int j) noexcept;

}  // namespace mixdex
