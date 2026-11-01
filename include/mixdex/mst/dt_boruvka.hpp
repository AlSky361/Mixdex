#pragma once

#include <vector>

#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/utils/aliases.hpp"
#include "mixdex/utils/graph.hpp"

namespace mixdex {

/// Deterministic mode resolves ties by (weight, min(u,v), max(u,v)) across threads.
/// Otherwise equal-bound candidates may be pruned; the MST remains minimal,
/// but its tied edges can vary with thread scheduling.
std::vector<Edge> dual_tree_boruvka(const BallTree& tree, bool deterministic = true);

struct DTBoruvkaStats {
    Count total_calls = 0;
    Count pruned_calls = 0;
    Count zero_lb_calls = 0;
    Count leaf_pairs = 0;
    Count metric_calls = 0;
    Count fast_skipped = 0;
    int iterations = 0;

    double pruned_ratio() const noexcept {
        return total_calls > 0
                   ? static_cast<double>(pruned_calls) / static_cast<double>(total_calls)
                   : 0.0;
    }
    double zero_lb_ratio() const noexcept {
        return total_calls > 0
                   ? static_cast<double>(zero_lb_calls) / static_cast<double>(total_calls)
                   : 0.0;
    }
};

struct DTBoruvkaResult {
    std::vector<Edge> edges;
    DTBoruvkaStats stats;
};

DTBoruvkaResult dual_tree_boruvka_with_stats(const BallTree& tree, bool deterministic = true);

}  // namespace mixdex
