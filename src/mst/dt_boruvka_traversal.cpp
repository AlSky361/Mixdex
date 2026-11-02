// Method references: docs/references.md#spanning-trees-and-dual-tree-search
#include "dt_boruvka_internal.hpp"

namespace mixdex::detail {
using constants::kZeroLbEps;
namespace {

template <std::size_t N>
void sort_pairs(int (&qs)[N], int (&rs)[N], double (&lbs)[N]) {
    for (int i = 1; i < static_cast<int>(N); ++i) {
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

void process_leaf_pair(int qnode_idx, int rnode_idx, DTBState& state, bool same_node) {
    const BallTree& tree = state.tree;
    const auto& nodes = tree.nodes();
    const auto& indices = tree.indices();
    const auto& cd = *tree.core_dist();

    const BallTree::Node& qnode = nodes[qnode_idx];
    const BallTree::Node& rnode = nodes[rnode_idx];

    const int qsz = qnode.high - qnode.low;
    const int rsz = rnode.high - rnode.low;

    ThreadCounters& cnt = state.counters.local();
    cnt.increment(cnt.leaf_pairs);

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
                cnt.increment(cnt.fast_skipped);
                continue;
            }

            cnt.increment(cnt.metric_calls);
            double w = tree.call_metric(q, r);
            if (state.may_improve(w, thr) || state.may_improve(w, thr_r)) {
                state.try_update(q, r, w);
                cq = state.uf.find_root(q);
                thr = state.best_w(cq);
            }
        }
    }
}

void dtb_recurse(int qnode_idx, int rnode_idx, DTBState& state, int depth, int max_spawn_depth);

template <std::size_t N>
void visit_children(
    int (&qs)[N], int (&rs)[N], DTBState& state, int depth, int max_spawn_depth, bool rank) {
    if (rank) {
        double lbs[N];
        for (std::size_t k = 0; k < N; ++k)
            lbs[k] = state.tree.lower_bound_mreach_node(qs[k], rs[k]);
        sort_pairs(qs, rs, lbs);
    }
    const bool spawn_ok = depth < max_spawn_depth;
    const int next_depth = depth + 1;
    for (std::size_t k = 0; k < N; ++k) {
        MIXDEX_OMP_TASK_IF_SHARED(spawn_ok, state)
        dtb_recurse(qs[k], rs[k], state, next_depth, max_spawn_depth);
    }
    if (spawn_ok) {
        MIXDEX_OMP_TASKWAIT
    }
}

void dtb_recurse(int qnode_idx, int rnode_idx, DTBState& state, int depth, int max_spawn_depth) {
    const auto& nodes = state.tree.nodes();
    const auto& qnode = nodes[qnode_idx];
    const auto& rnode = nodes[rnode_idx];
    ThreadCounters& cnt = state.counters.local();
    cnt.increment(cnt.total_calls);
    const double lb = state.tree.lower_bound_mreach_node(qnode_idx, rnode_idx);
    if (lb < kZeroLbEps)
        cnt.increment(cnt.zero_lb_calls);
    if (state.can_prune(qnode.low, qnode.high, rnode.low, rnode.high, lb)) {
        cnt.increment(cnt.pruned_calls);
        return;
    }
    if (qnode.is_leaf() && rnode.is_leaf()) {
        process_leaf_pair(qnode_idx, rnode_idx, state, qnode_idx == rnode_idx);
    } else if (qnode.is_leaf()) {
        int qs[2] = {qnode_idx, qnode_idx};
        int rs[2] = {rnode.left, rnode.right};
        visit_children(qs, rs, state, depth, max_spawn_depth, false);
    } else if (rnode.is_leaf()) {
        int qs[2] = {qnode.left, qnode.right};
        int rs[2] = {rnode_idx, rnode_idx};
        visit_children(qs, rs, state, depth, max_spawn_depth, false);
    } else if (qnode_idx == rnode_idx) {
        int qs[3] = {qnode.left, qnode.left, qnode.right};
        int rs[3] = {qnode.left, qnode.right, qnode.right};
        visit_children(qs, rs, state, depth, max_spawn_depth, true);
    } else {
        int qs[4] = {qnode.left, qnode.left, qnode.right, qnode.right};
        int rs[4] = {rnode.left, rnode.right, rnode.left, rnode.right};
        visit_children(qs, rs, state, depth, max_spawn_depth, true);
    }
}

int compute_max_spawn_depth(int threads) {
    if (threads <= 1)
        return 0;
    int depth = 0;
    while ((1 << depth) < threads) ++depth;
    return depth + 2;
}

}  // namespace

void boruvka_iteration(DTBState& state) {
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

}  // namespace mixdex::detail
