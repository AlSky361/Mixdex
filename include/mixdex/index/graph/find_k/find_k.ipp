#pragma once
#include <algorithm>
#include <cmath>
#include <optional>

#include "mixdex/index/graph/find_k/find_k.hpp"
#include "mixdex/index/graph/knn_graph/knn_graph.hpp"
#include "mixdex/utils/graph.hpp"
#include "mixdex/utils/uf.hpp"

namespace mixdex {

inline int min_connected_k(const KnnLists& lists, int m_pts) {
    const int n = static_cast<int>(lists.size());
    if (n == 0 || m_pts <= 0)
        return -1;

    size_t k_layers = 0;
    for (const auto& l : lists) k_layers = std::max(k_layers, l.size());

    UF uf(n);
    for (size_t k = 0; k < k_layers; ++k) {
        for (int i = 0; i < n; ++i)
            if (k < lists[i].size())
                uf.unite(i, lists[i][k].first);

        if (static_cast<int>(k) + 1 >= m_pts && uf.n_comp() == 1)
            return static_cast<int>(k) + 1;
    }
    return -1;
}

// Connectivity is monotone in k: double the list size and reuse its prefix layers.
// This avoids computing k_max neighbors when the graph connects much earlier.
template <typename Search>
int find_k_connected(const Search& search, int m_pts, int k_max) {
    const int n = search.n_rows();
    if (n == 0 || m_pts <= 0 || k_max < m_pts || k_max > n)
        return -1;

    int k_try = m_pts;
    while (true) {
        const int k = min_connected_k(search.knn_batch(k_try), m_pts);
        if (k >= 0)
            return k;
        if (k_try == k_max)
            return -1;
        k_try = (k_try > k_max / 2) ? k_max : 2 * k_try;
    }
}

template <typename Search>
std::optional<AutoK> auto_select_k(const Search& search, int m_pts, int k_max, double margin) {
    const int n = search.n_rows();
    if (n == 0 || m_pts <= 0 || k_max < m_pts || k_max > n)
        return std::nullopt;

    int k_try = std::min(k_max, 2 * m_pts);
    while (true) {
        KnnLists lists = search.knn_batch(k_try);
        const int k_star = min_connected_k(lists, m_pts);
        if (k_star >= 0) {
            const int k_opt = std::min(k_max, static_cast<int>(std::ceil(margin * k_star)));
            if (k_opt > k_try)
                lists = search.knn_batch(k_opt);  // Connectivity also holds at larger k.
            return AutoK{k_star, k_opt, std::move(lists)};
        }
        if (k_try == k_max)
            return std::nullopt;
        k_try = (k_try > k_max / 2) ? k_max : 2 * k_try;
    }
}

}  // namespace mixdex
