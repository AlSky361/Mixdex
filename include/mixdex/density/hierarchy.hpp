#pragma once
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/graph.hpp"

namespace mixdex {

enum class MSTMode { Exact, Approximate };
using HDBSCANMode = MSTMode;

/// Reuse core distances and a mutual-reachability MST for DBSCAN* cuts.
class DensityHierarchy {
public:
    // Approximate uses a kNN graph; automatic k selection stops at floor(sqrt(n)).
    // k_opt is ignored in Exact mode.
    DensityHierarchy(const Dataset& data,
                     int m_pts,
                     HDBSCANMode mode,
                     SearchMethod method,
                     int leaf_size = constants::kDefaultLeafSize,
                     int k_opt = constants::kKUnset,
                     bool deterministic_mst = true);

    int n() const noexcept {
        return static_cast<int>(core_.size());
    }
    int m_pts() const noexcept {
        return m_pts_;
    }
    const std::vector<double>& core_distances() const noexcept {
        return core_;
    }

    const std::vector<Edge>& mst_edges() const noexcept {
        return mst_;
    }

    std::vector<int> dbscan_star(double eps, int min_cluster_size = 1) const;

private:
    int m_pts_;
    std::vector<double> core_;
    std::vector<Edge> mst_;
};

}  // namespace mixdex
