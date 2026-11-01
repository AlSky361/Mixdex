#pragma once
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"

namespace mixdex {

/// GLOSH outlier scores from point departures and cluster density levels.
std::vector<double> compute_glosh(const std::vector<double>& core_dist,
                                  const std::vector<CondRow>& rows,
                                  const std::vector<int>& cl_parent,
                                  int n,
                                  int next_cl);

}  // namespace mixdex
