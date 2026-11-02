#pragma once
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <tuple>
#include <vector>

#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/omp_macros.hpp"
#include "mixdex/utils/per_thread.hpp"
#include "mixdex/utils/uf_members.hpp"

namespace mixdex::detail {
using constants::kInf;

struct ThreadCounters {
    Count total_calls{0};
    Count pruned_calls{0};
    Count zero_lb_calls{0};
    Count leaf_pairs{0};
    Count metric_calls{0};
    Count fast_skipped{0};
    bool overflow{false};

    // Report after the task region; exceptions must not escape OpenMP tasks.
    void increment(Count& value) noexcept {
        if (value == std::numeric_limits<Count>::max())
            overflow = true;
        else
            ++value;
    }
};

using CounterPool = PerThread<ThreadCounters>;

struct DTBState {
    const BallTree& tree;
    // Deterministic mode retains equal-bound candidates for tie resolution.
    // Otherwise equality pruning may select a different tied MST.
    const bool deterministic;
    UFMembers uf;

    std::vector<Edge> best_edge;

    std::vector<std::atomic<double>> comp_best_w;
    std::vector<std::mutex> component_locks;

    CounterPool counters;

    DTBState(const BallTree& t, bool det);

    double best_w(int c) const noexcept {
        return comp_best_w[static_cast<std::size_t>(c)].load(std::memory_order_relaxed);
    }

    void reset_best_w() noexcept;

    int n() const noexcept {
        return tree.n_rows();
    }

    // Retain finite equal weights for deterministic index tie-breaking; infinity cannot improve.
    // Other modes may prune ties and return different, equally minimal trees.
    bool may_improve(double candidate_w, double incumbent_w) const noexcept {
        if (candidate_w < incumbent_w)
            return true;
        return deterministic && candidate_w == incumbent_w && candidate_w < kInf;
    }

    static bool better_candidate(const Edge& incumbent, int i, int j, double w) {
        if (w != incumbent.w)
            return w < incumbent.w;
        int lo = std::min(i, j), hi = std::max(i, j);
        int inc_lo = std::min(incumbent.u, incumbent.v);
        int inc_hi = std::max(incumbent.u, incumbent.v);
        return std::tie(lo, hi) < std::tie(inc_lo, inc_hi);
    }

    void update_component(int c, int i, int j, double w) {
        if (!may_improve(w, best_w(c)))
            return;
        std::lock_guard lock(component_locks[static_cast<std::size_t>(c)]);
        if (better_candidate(best_edge[c], i, j, w)) {
            best_edge[c] = {i, j, w};
            comp_best_w[static_cast<std::size_t>(c)].store(w, std::memory_order_relaxed);
        }
    }

    void try_update(int i, int j, double w) {
        // UF is immutable during the round; merge only after all tasks join.
        const int ci = uf.find_root(i);
        const int cj = uf.find_root(j);
        if (ci == cj)
            return;
        // Lock one component at a time: no nested locks or cross-component invariant.
        update_component(ci, i, j, w);
        update_component(cj, i, j, w);
    }

    bool can_prune(int low, int high, int rlow, int rhigh, double lb) {
        const auto& idx = tree.indices();
        for (int i = low; i < high; ++i) {
            int ci = uf.find_root(idx[i]);
            if (may_improve(lb, best_w(ci)))
                return false;
        }
        for (int i = rlow; i < rhigh; ++i) {
            int ci = uf.find_root(idx[i]);
            if (may_improve(lb, best_w(ci)))
                return false;
        }
        return true;
    }

    int merge_and_collect(std::vector<Edge>& mst_edges);
};

ThreadCounters total_counters(const CounterPool& pool);
void boruvka_iteration(DTBState& state);

}  // namespace mixdex::detail
