#pragma once
#include <vector>

namespace mixdex {

// Core distance is the (m_pts - 1)th other-neighbor distance; m_pts=1 gives zero.
template <typename Search>
std::vector<double> core_distances(const Search& search, int m_pts);

}  // namespace mixdex

#include "mixdex/density/core/core.ipp"
