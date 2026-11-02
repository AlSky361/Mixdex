// Method references: docs/references.md#spanning-trees-and-dual-tree-search
#include <cinttypes>
#include <cstdio>

#include "dt_boruvka_internal.hpp"

namespace mixdex {
namespace {

constexpr bool kVerbose = false;

[[maybe_unused]] void print_iteration(const detail::ThreadCounters& c,
                                      Count previous_calls,
                                      int iteration,
                                      int added,
                                      int size,
                                      int n) {
    printf(
        "[DTBoruvka] iter=%d  added=%d  mst=%d/%d  "
        "calls=%" PRIu64 "  pruned=%" PRIu64
        " (%.1f%%)  "
        "zero_lb=%" PRIu64
        " (%.1f%%)  "
        "leaf_pairs=%" PRIu64 "  metric_calls=%" PRIu64 "  fast_skipped=%" PRIu64 "\n",
        iteration,
        added,
        size,
        n - 1,
        c.total_calls - previous_calls,
        c.pruned_calls,
        c.total_calls > 0 ? constants::kPercent * static_cast<double>(c.pruned_calls) /
                                static_cast<double>(c.total_calls)
                          : 0.0,
        c.zero_lb_calls,
        c.total_calls > 0 ? constants::kPercent * static_cast<double>(c.zero_lb_calls) /
                                static_cast<double>(c.total_calls)
                          : 0.0,
        c.leaf_pairs,
        c.metric_calls,
        c.fast_skipped);
}

DTBoruvkaStats statistics(const detail::ThreadCounters& c, int iterations) {
    DTBoruvkaStats result;
    result.total_calls = c.total_calls;
    result.pruned_calls = c.pruned_calls;
    result.zero_lb_calls = c.zero_lb_calls;
    result.leaf_pairs = c.leaf_pairs;
    result.metric_calls = c.metric_calls;
    result.fast_skipped = c.fast_skipped;
    result.iterations = iterations;
    return result;
}

std::vector<Edge> build_mst(const BallTree& tree, bool deterministic, DTBoruvkaStats* stats) {
    const int n = tree.n_rows();
    if (n == 0)
        return {};
    std::vector<Edge> edges;
    edges.reserve(n - 1);
    detail::DTBState state(tree, deterministic);
    detail::ThreadCounters counts;
    int iteration = 0;
    while (static_cast<int>(edges.size()) < n - 1) {
        const Count previous_calls = counts.total_calls;
        detail::boruvka_iteration(state);
        counts = detail::total_counters(state.counters);
        const int added = state.merge_and_collect(edges);
        if constexpr (kVerbose)
            print_iteration(
                counts, previous_calls, iteration, added, static_cast<int>(edges.size()), n);
        if (added == 0)
            break;
        ++iteration;
    }
    if (stats)
        *stats = statistics(counts, iteration);
    return edges;
}

}  // namespace

std::vector<Edge> dual_tree_boruvka(const BallTree& tree, bool deterministic) {
    return build_mst(tree, deterministic, nullptr);
}

DTBoruvkaResult dual_tree_boruvka_with_stats(const BallTree& tree, bool deterministic) {
    DTBoruvkaResult result;
    result.edges = build_mst(tree, deterministic, &result.stats);
    return result;
}

}  // namespace mixdex
