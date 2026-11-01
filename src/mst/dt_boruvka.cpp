#include "mixdex/mst/dt_boruvka.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <numeric>
#include <tuple>
#include <vector>

#include "mixdex/distance.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/omp_macros.hpp"
#include "mixdex/utils/per_thread.hpp"
#include "mixdex/utils/uf_members.hpp"

using namespace mixdex::constants;

namespace mixdex::debug {

constexpr bool DTB_VERBOSE = false;

}

namespace mixdex {

struct ThreadCounters {
    Count total_calls{0};
    Count pruned_calls{0};
    Count zero_lb_calls{0};
    Count leaf_pairs{0};
    Count metric_calls{0};
    Count fast_skipped{0};
};

using CounterPool = PerThread<ThreadCounters>;

static ThreadCounters total_counters(const CounterPool& pool) noexcept {
    ThreadCounters t;
    for (int i = 0; i < pool.threads(); ++i) {
        const ThreadCounters& c = pool.slot(i);
        t.total_calls += c.total_calls;
        t.pruned_calls += c.pruned_calls;
        t.zero_lb_calls += c.zero_lb_calls;
        t.leaf_pairs += c.leaf_pairs;
        t.metric_calls += c.metric_calls;
        t.fast_skipped += c.fast_skipped;
    }
    return t;
}

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

    DTBState(const BallTree& t, bool det) :
        tree(t),
        deterministic(det),
        uf(t.n_rows()),
        best_edge(t.n_rows(), Edge(-1, -1, kInf)),
        comp_best_w(static_cast<std::size_t>(t.n_rows())),
        component_locks(static_cast<std::size_t>(t.n_rows())),
        counters(MIXDEX_OMP_MAX_THREADS()) {
        reset_best_w();
    }

    double best_w(int c) const noexcept {
        return comp_best_w[static_cast<std::size_t>(c)].load(std::memory_order_relaxed);
    }

    void reset_best_w() noexcept {
        for (std::atomic<double>& w : comp_best_w) w.store(kInf, std::memory_order_relaxed);
    }

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

    int merge_and_collect(std::vector<Edge>& mst_edges) {
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
};

static void sort3(int qs[3], int rs[3], double lbs[3]) {
    for (int i = 1; i < 3; ++i) {
        double lb = lbs[i];
        int q = qs[i], r = rs[i];
        int j = i - 1;
        while (j >= 0 && lbs[j] > lb) {
            lbs[j + 1] = lbs[j];
            qs[j + 1] = qs[j];
            rs[j + 1] = rs[j];
            --j;
        }
        lbs[j + 1] = lb;
        qs[j + 1] = q;
        rs[j + 1] = r;
    }
}

static void sort4(int qs[4], int rs[4], double lbs[4]) {
    for (int i = 1; i < 4; ++i) {
        double lb = lbs[i];
        int q = qs[i], r = rs[i];
        int j = i - 1;
        while (j >= 0 && lbs[j] > lb) {
            lbs[j + 1] = lbs[j];
            qs[j + 1] = qs[j];
            rs[j + 1] = rs[j];
            --j;
        }
        lbs[j + 1] = lb;
        qs[j + 1] = q;
        rs[j + 1] = r;
    }
}

static void process_leaf_pair(int qnode_idx, int rnode_idx, DTBState& state, bool same_node) {
    const BallTree& tree = state.tree;
    const auto& nodes = tree.nodes();
    const auto& indices = tree.indices();
    const auto& cd = *tree.core_dist();

    const BallTree::Node& qnode = nodes[qnode_idx];
    const BallTree::Node& rnode = nodes[rnode_idx];

    const int qsz = qnode.high - qnode.low;
    const int rsz = rnode.high - rnode.low;

    ThreadCounters& cnt = state.counters.local();
    ++cnt.leaf_pairs;

    for (int qi = 0; qi < qsz; ++qi) {
        int q = indices[qnode.low + qi];
        int cq = state.uf.find_root(q);
        double thr = state.best_w(cq);
        double cdq = cd[q];

        int ri_begin = same_node ? qi + 1 : 0;

        for (int ri = ri_begin; ri < rsz; ++ri) {
            int r = indices[rnode.low + ri];
            int cr = state.uf.find_root(r);
            if (cr == cq)
                continue;

            double thr_r = state.best_w(cr);

            double lb_fast = std::max(cdq, cd[r]);
            if (!state.may_improve(lb_fast, thr) && !state.may_improve(lb_fast, thr_r)) {
                ++cnt.fast_skipped;
                continue;
            }

            ++cnt.metric_calls;
            double w = tree.call_metric(q, r);
            if (state.may_improve(w, thr) || state.may_improve(w, thr_r)) {
                state.try_update(q, r, w);
                cq = state.uf.find_root(q);
                thr = state.best_w(cq);
            }
        }
    }
}

static void dtb_recurse(
    int qnode_idx, int rnode_idx, DTBState& state, int depth, int max_spawn_depth) {
    const BallTree& tree = state.tree;
    const auto& nodes = tree.nodes();

    const BallTree::Node& qnode = nodes[qnode_idx];
    const BallTree::Node& rnode = nodes[rnode_idx];

    ThreadCounters& cnt = state.counters.local();
    ++cnt.total_calls;

    double lb = tree.lower_bound_mreach_node(qnode_idx, rnode_idx);

    if (lb < kZeroLbEps)
        ++cnt.zero_lb_calls;

    if (state.can_prune(qnode.low, qnode.high, rnode.low, rnode.high, lb)) {
        ++cnt.pruned_calls;
        return;
    }

    if (qnode.is_leaf() && rnode.is_leaf()) {
        process_leaf_pair(qnode_idx, rnode_idx, state, qnode_idx == rnode_idx);
        return;
    }

    const bool spawn_ok = depth < max_spawn_depth;
    const int next_depth = depth + 1;

    if (qnode.is_leaf()) {
        MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
        dtb_recurse(qnode_idx, rnode.left, state, next_depth, max_spawn_depth);
        MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
        dtb_recurse(qnode_idx, rnode.right, state, next_depth, max_spawn_depth);
        if (spawn_ok) {
            MIXDEX_OMP_TASKWAIT
        }
        return;
    }
    if (rnode.is_leaf()) {
        MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
        dtb_recurse(qnode.left, rnode_idx, state, next_depth, max_spawn_depth);
        MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
        dtb_recurse(qnode.right, rnode_idx, state, next_depth, max_spawn_depth);
        if (spawn_ok) {
            MIXDEX_OMP_TASKWAIT
        }
        return;
    }

    if (qnode_idx == rnode_idx) {
        int qs[3] = {qnode.left, qnode.left, qnode.right};
        int rs[3] = {qnode.left, qnode.right, qnode.right};
        double lbs[3] = {
            tree.lower_bound_mreach_node(qnode.left, qnode.left),
            tree.lower_bound_mreach_node(qnode.left, qnode.right),
            tree.lower_bound_mreach_node(qnode.right, qnode.right),
        };
        sort3(qs, rs, lbs);
        for (int k = 0; k < 3; ++k) {
            MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
            dtb_recurse(qs[k], rs[k], state, next_depth, max_spawn_depth);
        }
        if (spawn_ok) {
            MIXDEX_OMP_TASKWAIT
        }
    } else {
        int qs[4] = {qnode.left, qnode.left, qnode.right, qnode.right};
        int rs[4] = {rnode.left, rnode.right, rnode.left, rnode.right};
        double lbs[4] = {
            tree.lower_bound_mreach_node(qnode.left, rnode.left),
            tree.lower_bound_mreach_node(qnode.left, rnode.right),
            tree.lower_bound_mreach_node(qnode.right, rnode.left),
            tree.lower_bound_mreach_node(qnode.right, rnode.right),
        };
        sort4(qs, rs, lbs);
        for (int k = 0; k < 4; ++k) {
            MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
            dtb_recurse(qs[k], rs[k], state, next_depth, max_spawn_depth);
        }
        if (spawn_ok) {
            MIXDEX_OMP_TASKWAIT
        }
    }
}

static int compute_max_spawn_depth(int threads) {
    if (threads <= 1)
        return 0;
    int depth = 0;
    while ((1 << depth) < threads) ++depth;
    return depth + 2;
}

static void boruvka_iteration(DTBState& state) {
    const auto& nodes = state.tree.nodes();
    const int root = state.tree.root();

    if (nodes[root].is_leaf()) {
        dtb_recurse(root, root, state, 0, 0);
        return;
    }

    const int threads = state.counters.threads();
    const int max_spawn_depth = compute_max_spawn_depth(threads);

    int ql = nodes[root].left;
    int qr = nodes[root].right;

    int qs[3] = {ql, ql, qr};
    int rs[3] = {ql, qr, qr};

    MIXDEX_OMP_PARALLEL_TASKS_N(threads, {
        for (int k = 0; k < 3; ++k) {
            int q = qs[k];
            int r = rs[k];
            MIXDEX_OMP_TASK_SHARED(state)
            dtb_recurse(q, r, state, 1, max_spawn_depth);
        }
        MIXDEX_OMP_TASKWAIT
    })
}

std::vector<Edge> dual_tree_boruvka(const BallTree& tree, bool deterministic) {
    const int n = tree.n_rows();
    if (n == 0)
        return {};

    std::vector<Edge> mst_edges;
    mst_edges.reserve(n - 1);

    DTBState state(tree, deterministic);

    int iteration = 0;
    while (static_cast<int>(mst_edges.size()) < n - 1) {
        const Count calls_before = total_counters(state.counters).total_calls;
        boruvka_iteration(state);
        int added = state.merge_and_collect(mst_edges);

        if constexpr (debug::DTB_VERBOSE) {
            const ThreadCounters c = total_counters(state.counters);
            printf(
                "[DTBoruvka] iter=%d  added=%d  mst=%d/%d  "
                "calls=%lld  pruned=%lld (%.1f%%)  "
                "zero_lb=%lld (%.1f%%)  "
                "leaf_pairs=%lld  metric_calls=%lld  fast_skipped=%lld\n",
                iteration,
                added,
                static_cast<int>(mst_edges.size()),
                n - 1,
                c.total_calls - calls_before,
                c.pruned_calls,
                c.total_calls > 0 ? kPercent * static_cast<double>(c.pruned_calls) /
                                        static_cast<double>(c.total_calls)
                                  : 0.0,
                c.zero_lb_calls,
                c.total_calls > 0 ? kPercent * static_cast<double>(c.zero_lb_calls) /
                                        static_cast<double>(c.total_calls)
                                  : 0.0,
                c.leaf_pairs,
                c.metric_calls,
                c.fast_skipped);
        }

        if (added == 0)
            break;
        ++iteration;
    }

    return mst_edges;
}

DTBoruvkaResult dual_tree_boruvka_with_stats(const BallTree& tree, bool deterministic) {
    const int n = tree.n_rows();
    DTBoruvkaResult result;
    if (n == 0)
        return result;

    result.edges.reserve(n - 1);

    DTBState state(tree, deterministic);

    int iteration = 0;
    while (static_cast<int>(result.edges.size()) < n - 1) {
        boruvka_iteration(state);
        int added = state.merge_and_collect(result.edges);
        if (added == 0)
            break;
        ++iteration;
    }

    const ThreadCounters c = total_counters(state.counters);
    result.stats.total_calls = c.total_calls;
    result.stats.pruned_calls = c.pruned_calls;
    result.stats.zero_lb_calls = c.zero_lb_calls;
    result.stats.leaf_pairs = c.leaf_pairs;
    result.stats.metric_calls = c.metric_calls;
    result.stats.fast_skipped = c.fast_skipped;
    result.stats.iterations = iteration;

    return result;
}

}  // namespace mixdex
