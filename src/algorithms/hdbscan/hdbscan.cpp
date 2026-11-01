// Method references: docs/references.md#density-clustering
#include "mixdex/algorithms/hdbscan/hdbscan.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/algorithms/hdbscan/glosh.hpp"
#include "mixdex/density/hierarchy.hpp"
#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/index/brute_force/brute_force.hpp"
#include "mixdex/index/graph/knn_graph/knn_graph.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/mst/boruvka.hpp"
#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/omp_macros.hpp"
#include "mixdex/utils/uf_members.hpp"

using namespace mixdex::constants;

namespace mixdex {

namespace {

void append_self_edges(std::vector<Edge>& edges, const std::vector<double>& core) {
    for (int i = 0; i < static_cast<int>(core.size()); ++i) {
        if (core[i] > 0.0)
            edges.emplace_back(i, i, core[i]);
    }
}

bool edge_order(const Edge& a, const Edge& b) {
    if (a.w != b.w)
        return a.w > b.w;

    bool a_self = (a.u == a.v);
    bool b_self = (b.u == b.v);

    if (a_self != b_self)
        return b_self;
    if (a.u != b.u)
        return a.u < b.u;

    return a.v < b.v;
}

}  // namespace

HDBSCANResult hdbscan(const Dataset& data,
                      int m_pts,
                      int min_cluster_size,
                      HDBSCANMode mode,
                      SearchMethod method,
                      int leaf_size,
                      int k_opt,
                      bool deterministic_mst) {
    const int n = data.n_rows();
    HDBSCANResult result;
    result.labels.assign(n, constants::kNoiseLabel);
    result.outlier_scores.assign(n, 1.0);
    result.cluster_stabilities.clear();

    if (n == 0)
        return result;

    DensityHierarchy hier(data, m_pts, mode, method, leaf_size, k_opt, deterministic_mst);
    const std::vector<double>& core = hier.core_distances();
    std::vector<Edge> mst_edges = hier.mst_edges();

    append_self_edges(mst_edges, core);
    std::stable_sort(mst_edges.begin(), mst_edges.end(), edge_order);

    std::vector<CondRow> rows;
    std::vector<double> lambda_birth;
    std::vector<int> final_cluster;
    int next_cl = n + 1;

    build_condensed_tree(
        mst_edges, n, min_cluster_size, core, rows, lambda_birth, next_cl, final_cluster);

    extract_clusters(
        rows, lambda_birth, n, next_cl, final_cluster, result.labels, result.cluster_stabilities);

    const int n_cl = next_cl - n;
    std::vector<int> cl_parent(n_cl, -1);

    for (const auto& r : rows) {
        if (r.child_cl < n)
            continue;
        int chi = r.child_cl - n;
        int pai = r.parent_cl - n;
        if (chi >= 0 && chi < n_cl && pai >= 0 && pai < n_cl)
            cl_parent[chi] = pai;
    }

    result.outlier_scores = compute_glosh(core, rows, cl_parent, n, next_cl);
    return result;
}

}  // namespace mixdex
