#include "dt_boruvka_internal.hpp"
#include "mixdex/utils/checked_arithmetic.hpp"

namespace mixdex::detail {

ThreadCounters total_counters(const CounterPool& pool) {
    ThreadCounters t;
    for (int i = 0; i < pool.threads(); ++i) {
        const ThreadCounters& c = pool.slot(i);
        if (c.overflow)
            throw std::overflow_error("counter overflow");
        detail::add_count(t.total_calls, c.total_calls);
        detail::add_count(t.pruned_calls, c.pruned_calls);
        detail::add_count(t.zero_lb_calls, c.zero_lb_calls);
        detail::add_count(t.leaf_pairs, c.leaf_pairs);
        detail::add_count(t.metric_calls, c.metric_calls);
        detail::add_count(t.fast_skipped, c.fast_skipped);
    }
    return t;
}

DTBState::DTBState(const BallTree& t, bool det) :
    tree(t),
    deterministic(det),
    uf(t.n_rows()),
    best_edge(t.n_rows(), Edge(-1, -1, kInf)),
    comp_best_w(static_cast<std::size_t>(t.n_rows())),
    component_locks(static_cast<std::size_t>(t.n_rows())),
    counters(MIXDEX_OMP_MAX_THREADS()) {
    reset_best_w();
}

void DTBState::reset_best_w() noexcept {
    for (std::atomic<double>& w : comp_best_w) w.store(kInf, std::memory_order_relaxed);
}

int DTBState::merge_and_collect(std::vector<Edge>& mst_edges) {
    int added = 0;
    for (int c = 0; c < n(); ++c) {
        if (uf.find_compress(c) != c)
            continue;
        Edge& e = best_edge[c];
        if (e.u < 0)
            continue;
        if (uf.same(e.u, e.v))
            continue;
        mst_edges.push_back(e);
        uf.unite(e.u, e.v);
        ++added;
    }

    uf.flatten();
    std::fill(best_edge.begin(), best_edge.end(), Edge(-1, -1, kInf));
    reset_best_w();
    return added;
}

}  // namespace mixdex::detail
