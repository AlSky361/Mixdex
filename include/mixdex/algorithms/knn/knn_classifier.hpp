#pragma once
#include <cstddef>
#include <span>
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {

using constants::kUnlabeled;

// Ignore kUnlabeled (-1); equal vote totals choose the smallest label.
// Weighted votes use inverse distance; zero-distance neighbors take precedence.
std::vector<int> knn_classify(const NeighbourSearch& search,
                              const std::vector<int>& labels,
                              int k,
                              bool distance_weighted = false);

std::vector<int> knn_classify(const Dataset& data,
                              const std::vector<int>& labels,
                              int k,
                              bool distance_weighted = false,
                              SearchMethod method = SearchMethod::BallTree,
                              int leaf_size = constants::kDefaultLeafSize);

std::vector<int> knn_predict(const NeighbourSearch& search,
                             const std::vector<int>& labels,
                             std::span<const double> num,
                             std::span<const int> cat,
                             std::size_t m,
                             int k,
                             bool distance_weighted = false);

std::vector<int> knn_predict(const Dataset& data,
                             const std::vector<int>& labels,
                             std::span<const double> num,
                             std::span<const int> cat,
                             std::size_t m,
                             int k,
                             bool distance_weighted = false,
                             SearchMethod method = SearchMethod::BallTree,
                             int leaf_size = constants::kDefaultLeafSize);

}  // namespace mixdex
