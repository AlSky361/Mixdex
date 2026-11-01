#pragma once
#include <utility>
#include <vector>

#include "mixdex/types.hpp"
#include "mixdex/utils/graph.hpp"

namespace mixdex {

template <typename Search>
Graph build_knn_graph(const Search& search, int k);

// Use the first k entries per list; preserve build_knn_graph(search, k) edge order.
inline Graph graph_from_knn_lists(const std::vector<std::vector<std::pair<int, double>>>& lists,
                                  int k);

template <typename Metric>
std::vector<Edge> build_full_mreach_edges(const Dataset& data,
                                          const std::vector<double>& core,
                                          Metric metric_distance);

}  // namespace mixdex

#include "mixdex/index/graph/knn_graph/knn_graph.ipp"
