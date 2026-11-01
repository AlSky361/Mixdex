#pragma once
#include <algorithm>

#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/index/graph/knn_graph/knn_graph.hpp"
#include "mixdex/utils/graph.hpp"
#include "mixdex/utils/parallel.hpp"

namespace mixdex {

template <typename Search>
Graph build_knn_graph(const Search& search, int k) {
    const int n = search.n_rows();
    std::vector<std::vector<std::pair<int, double>>> local(n);

    parallel_for<Schedule::Static>(n, [&](int i) {
        local[i] = search.knn(i, k);
    });

    Graph g(n, k);
    for (int i = 0; i < n; ++i)
        for (const auto& [j, w] : local[i]) g.add_edge(i, j, w);

    return g;
}

inline Graph graph_from_knn_lists(const std::vector<std::vector<std::pair<int, double>>>& lists,
                                  int k) {
    const int n = static_cast<int>(lists.size());
    Graph g(n, k);
    for (int i = 0; i < n; ++i) {
        const size_t take = std::min(static_cast<size_t>(k), lists[i].size());
        for (size_t t = 0; t < take; ++t) g.add_edge(i, lists[i][t].first, lists[i][t].second);
    }
    return g;
}

template <typename Metric>
std::vector<Edge> build_full_mreach_edges(const Dataset& data,
                                          const std::vector<double>& core,
                                          Metric metric_distance) {
    const int n = data.n_rows();
    const int n_edges = n * (n - 1) / 2;

    std::vector<Edge> edges(n_edges);

    parallel_for<Schedule::Guided>(n, [&](int i) {
        int base = i * (2 * n - i - 1) / 2;
        for (int j = i + 1; j < n; ++j)
            edges[base + (j - i - 1)] = Edge(i, j, mreach_distance(core, i, j, metric_distance));
    });

    return edges;
}

}  // namespace mixdex
