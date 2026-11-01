#pragma once
#include <vector>

#include "mixdex/density/hierarchy.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {

struct HDBSCANResult {
    std::vector<int> labels;  // -1 denotes noise.
    std::vector<double> outlier_scores;
    std::vector<double> cluster_stabilities;  // Indexed by nonnegative cluster label.
};

/// Build the density hierarchy, select clusters and compute GLOSH scores.
/// k_opt controls Approximate mode; kKUnset requests automatic selection.
HDBSCANResult hdbscan(const Dataset& data,
                      int m_pts,
                      int min_cluster_size,
                      HDBSCANMode mode,
                      SearchMethod method,
                      int leaf_size = constants::kDefaultLeafSize,
                      int k_opt = constants::kKUnset,
                      bool deterministic_mst = true);

}  // namespace mixdex
