#pragma once
#include <cstddef>
#include <span>
#include <vector>

#include "mixdex/index/neighbours.hpp"

namespace mixdex {

/// Distance to the kth other row; larger values indicate more isolated points.
std::vector<double> knn_distance_scores(const NeighbourSearch& search, int k);

std::vector<double> knn_distance_scores(const NeighbourSearch& search,
                                        std::span<const double> num,
                                        std::span<const int> cat,
                                        std::size_t m,
                                        int k);

}  // namespace mixdex
