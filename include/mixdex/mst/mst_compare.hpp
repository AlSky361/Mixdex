#pragma once

#include <vector>

#include "mixdex/utils/graph.hpp"

namespace mixdex {

/// Fraction of reference edges found by endpoint pair; weights are ignored.
double mst_edge_recall(const std::vector<Edge>& candidate, const std::vector<Edge>& reference);

}  // namespace mixdex
