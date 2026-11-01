#pragma once
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {

/// Density clustering; min_pts includes the query itself, and -1 marks noise.
std::vector<int> dbscan(const NeighbourSearch& search, double eps, int min_pts);

std::vector<int> dbscan(const Dataset& data,
                        double eps,
                        int min_pts,
                        SearchMethod method = SearchMethod::BallTree,
                        int leaf_size = constants::kDefaultLeafSize);

}  // namespace mixdex
