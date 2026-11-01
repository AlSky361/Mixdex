#pragma once
#include <stdexcept>
#include <vector>

#include "mixdex/density/core/core.hpp"

namespace mixdex {

template <typename Search>
std::vector<double> core_distances(const Search& search, int m_pts) {
    if (m_pts < 1)
        throw std::invalid_argument("core_distances: m_pts must be >= 1");

    const int n = search.n_rows();
    if (n == 0)
        return {};
    if (m_pts == 1)
        return std::vector<double>(n, 0.0);

    return search.kth_distance_all(m_pts - 1);
}

}  // namespace mixdex
