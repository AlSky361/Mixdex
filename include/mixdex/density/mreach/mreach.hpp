#pragma once
#include <vector>

namespace mixdex {

// Mutual reachability is max(core[i], core[j], metric(i,j)); indices must be valid.
template <typename Metric>
double mreach_distance(const std::vector<double>& core_dist, int i, int j, Metric metric);

}  // namespace mixdex

#include "mixdex/density/mreach/mreach.ipp"
