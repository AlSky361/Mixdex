#pragma once
#include <optional>
#include <utility>
#include <vector>

namespace mixdex {

// Neighbor lists contain (index, distance) pairs in ascending distance order.
using KnnLists = std::vector<std::vector<std::pair<int, double>>>;

// Return the smallest connected k >= m_pts, or -1 if all available layers fail.
inline int min_connected_k(const KnnLists& lists, int m_pts);

// Return the smallest connected k in [m_pts, k_max], or -1; use the search metric.
template <typename Search>
int find_k_connected(const Search& search, int m_pts, int k_max);

// Choose Approximate k while reusing neighbor lists.
struct AutoK {
    int k_star;      // Smallest connected k.
    int k_opt;       // min(k_max, ceil(margin * k_star)).
    KnnLists lists;  // At least k_opt neighbors per point, sorted by distance.
};

// Double k_try from min(k_max, 2*m_pts) until connected; reuse lists if possible.
// Return nullopt for invalid limits or disconnection at k_max.
template <typename Search>
std::optional<AutoK> auto_select_k(const Search& search, int m_pts, int k_max, double margin);

}  // namespace mixdex

#include "find_k.ipp"
