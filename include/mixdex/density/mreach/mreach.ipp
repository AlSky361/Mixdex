#pragma once
#include <algorithm>
#include <vector>

#include "mixdex/density/mreach/mreach.hpp"

namespace mixdex {

template <typename Metric>
double mreach_distance(const std::vector<double>& core_dist, int i, int j, Metric metric) {
    double d = metric(i, j);

    return std::max({d, core_dist[i], core_dist[j]});
}

}  // namespace mixdex
