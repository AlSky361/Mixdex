#pragma once
#include <algorithm>
#include <vector>

#include "mixdex/index/brute_force/brute_force.hpp"
#include "mixdex/utils/omp_macros.hpp"

namespace mixdex {

template <typename Metric>
std::vector<std::pair<int, double>> BruteForceSearch<Metric>::knn(int query_idx, int k) const {
    check_k(k);
    FixedMaxHeap heap(k);
    const int n = data.n_rows();

    for (int i = 0; i < n; ++i) {
        if (i == query_idx)
            continue;
        double d = dist(query_idx, i);
        heap.try_insert(d, i);
    }

    return heap.sorted();
}

template <typename Metric>
std::vector<std::pair<int, double>> BruteForceSearch<Metric>::radius_query(int query_idx,
                                                                           double r) const {
    check_radius(r);
    std::vector<std::pair<int, double>> result;
    const int n = data.n_rows();

    for (int i = 0; i < n; ++i) {
        if (i == query_idx)
            continue;
        double d = dist(query_idx, i);
        if (d <= r)
            result.emplace_back(i, d);
    }

    sort_neighbours(result);
    return result;
}

template <typename Metric>
std::vector<std::pair<int, double>> BruteForceSearch<Metric>::knn(const RowView& q, int k) const {
    check_row(q);
    check_external_ready();
    check_k(k);
    FixedMaxHeap heap(k);
    const int n = data.n_rows();
    for (int i = 0; i < n; ++i) heap.try_insert(distance(data, q, i), i);
    return heap.sorted();
}

template <typename Metric>
std::vector<std::pair<int, double>> BruteForceSearch<Metric>::radius_query(const RowView& q,
                                                                           double r) const {
    check_row(q);
    check_external_ready();
    check_radius(r);
    std::vector<std::pair<int, double>> result;
    const int n = data.n_rows();
    for (int i = 0; i < n; ++i) {
        const double d = distance(data, q, i);
        if (d <= r)
            result.emplace_back(i, d);
    }
    sort_neighbours(result);
    return result;
}

}  // namespace mixdex
