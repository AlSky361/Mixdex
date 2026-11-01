// Private condensed-tree state and equal-weight step handlers.
// The fast path uses bounded BFS for a single cut; the slow path rebuilds components.
#pragma once
#include <cstddef>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/utils/graph.hpp"
#include "mixdex/utils/rollback_uf.hpp"

namespace mixdex {

#ifdef MIXDEX_PROFILE_FASTPATH
extern long long g_fp_hot_steps, g_fp_reason_multicluster, g_fp_reason_multinonself,
    g_fp_reason_group_gt2, g_fp_reason_self_unrelated, g_fp_reason_cap_bothbig, g_fp_reason_ok;
extern long long g_fp_multinonself_edgesum, g_fp_multinonself_sizesum, g_fp_multinonself_count;
#endif

namespace detail {

// References to state owned by build_condensed_tree.
struct CondTreeState {
    const std::vector<Edge>& edges;
    int n;
    int min_cluster_size;
    const std::vector<double>& core;
    Graph& g;
    RollbackUF& uf;
    std::vector<int>& cluster;
    std::vector<std::vector<int>>& cl_members;
    std::vector<CondRow>& rows;
    std::vector<double>& lambda_birth;
    int& next_cl;
    std::vector<char>& shed;  // A shed point already has its output row.
};

// Equal-weight edges [i, j), with lambda = 1/w.
struct StepInfo {
    std::size_t i;
    std::size_t j;
    double w;
    double lambda;
};

// Ordered maps keep row and cluster-ID assignment stable across standard libraries.
using ClusterPoints = std::map<int, const std::vector<int>*>;

// Write each point departure once; shed flags prevent later edges from duplicating it.
// Keep cluster[p] so extraction can find the final cluster.
inline void shed_point(CondTreeState& st, int cl_id, int p, double lambda) {
    st.rows.push_back({cl_id, p, lambda, 1});
    st.shed[p] = 1;
}

// Return true if handled; false leaves state unchanged for the slow path.
bool try_fast_path(CondTreeState& st,
                   const StepInfo& step,
                   const std::unordered_set<int>& affected_clusters,
                   const ClusterPoints& cluster_points);

void slow_path(CondTreeState& st, const StepInfo& step, const ClusterPoints& cluster_points);

}  // namespace detail
}  // namespace mixdex
