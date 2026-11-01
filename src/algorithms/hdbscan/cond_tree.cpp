// Condensed-tree construction; flat-cluster extraction lives in cond_tree_extract.cpp.
#include "mixdex/algorithms/hdbscan/cond_tree.hpp"

#include <cassert>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cond_tree_internal.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/graph.hpp"
#include "mixdex/utils/rollback_uf.hpp"

using namespace mixdex::constants;

namespace mixdex {
namespace {

using Step = std::pair<std::size_t, std::size_t>;  // Half-open range [i, j) of equal-weight edges.

// Connectivity searches exclude self-loops.
Graph build_graph(const std::vector<Edge>& edges, int n) {
    Graph g(n);
    for (const auto& e : edges) {
        if (e.u != e.v)
            g.add_edge(e.u, e.v, e.w);
    }
    return g;
}

// Group descending edges by equal weight.
std::vector<Step> group_equal_weight_edges(const std::vector<Edge>& edges) {
    std::vector<Step> steps;
    std::size_t i = 0;
    while (i < edges.size()) {
        const double w = edges[i].w;
        std::size_t j = i;
        while (j < edges.size() && f_equal(edges[j].w, w)) ++j;
        steps.emplace_back(i, j);
        i = j;
    }
    return steps;
}

// Merge from light to heavy; undo_log[k] restores the state before group k.
std::vector<std::size_t> build_undo_log(const std::vector<Edge>& edges,
                                        const std::vector<Step>& steps,
                                        RollbackUF& uf) {
    std::vector<std::size_t> undo_log(steps.size() + 1, 0);
    for (std::size_t k = steps.size(); k-- > 0;) {
        undo_log[k] = uf.snapshot();
        for (std::size_t e = steps[k].first; e < steps[k].second; ++e) {
            if (edges[e].u != edges[e].v)
                uf.unite(edges[e].u, edges[e].v);
        }
    }
    return undo_log;
}

// Reserve up to 2n-1 clusters: push_back must not invalidate member references.
std::vector<std::vector<int>> make_cluster_members(int n) {
    std::vector<std::vector<int>> cl_members;
    cl_members.reserve(2 * static_cast<std::size_t>(n));
    cl_members.emplace_back(static_cast<std::size_t>(n));
    std::iota(cl_members[0].begin(), cl_members[0].end(), 0);
    return cl_members;
}

void remove_step_edges(Graph& g, const std::vector<Edge>& edges, const Step& s) {
    for (std::size_t k = s.first; k < s.second; ++k) {
        if (edges[k].u != edges[k].v)
            g.remove_edge(edges[k].u, edges[k].v);
    }
}

// Skip shed endpoints: they no longer belong to live clusters.
std::unordered_set<int> collect_affected_clusters(const std::vector<Edge>& edges,
                                                  const Step& s,
                                                  const std::vector<int>& cluster,
                                                  const std::vector<char>& shed) {
    std::unordered_set<int> affected;
    for (std::size_t k = s.first; k < s.second; ++k) {
        const auto& e = edges[k];
        if (shed[e.u] || shed[e.v])
            continue;
        if (e.u != e.v) {
            if (cluster[e.u] >= 0)
                affected.insert(cluster[e.u]);
            if (cluster[e.v] >= 0)
                affected.insert(cluster[e.v]);
        } else if (cluster[e.u] >= 0) {
            affected.insert(cluster[e.u]);
        }
    }
    return affected;
}

// Member views for affected clusters.
detail::ClusterPoints collect_cluster_points(const std::unordered_set<int>& affected,
                                             const std::vector<std::vector<int>>& cl_members,
                                             int n) {
    detail::ClusterPoints cluster_points;
    for (int cl_id : affected) {
        const int idx = cl_id - n;
        if (idx >= 0 && idx < static_cast<int>(cl_members.size()))
            cluster_points[cl_id] = &cl_members[idx];
    }
    return cluster_points;
}

}  // namespace

void build_condensed_tree(const std::vector<Edge>& reversed_MSText,
                          int n,
                          int min_cluster_size,
                          const std::vector<double>& core,
                          std::vector<CondRow>& rows,
                          std::vector<double>& lambda_birth,
                          int& next_cl,
                          std::vector<int>& final_cluster) {
    rows.clear();
    lambda_birth.clear();
    next_cl = n + 1;
    lambda_birth.push_back(0.0);

    const auto& edges = reversed_MSText;

    Graph g = build_graph(edges, n);
    RollbackUF uf(n);
    const std::vector<Step> steps = group_equal_weight_edges(edges);
    const std::vector<std::size_t> undo_log = build_undo_log(edges, steps, uf);

    std::vector<int> cluster(n, n);
    std::vector<char> shed(static_cast<std::size_t>(n), 0);
    std::vector<std::vector<int>> cl_members = make_cluster_members(n);
    const std::size_t cl_capacity0 = cl_members.capacity();

    for (std::size_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
        const Step& s = steps[step_idx];
        const double w = edges[s.first].w;
        const double lambda = (w > kEpsZero ? 1.0 / w : kInf);

        remove_step_edges(g, edges, s);
        uf.rollback_to(undo_log[step_idx]);

        const std::unordered_set<int> affected = collect_affected_clusters(edges, s, cluster, shed);
        const detail::ClusterPoints cluster_points =
            collect_cluster_points(affected, cl_members, n);

        detail::CondTreeState st{edges,
                                 n,
                                 min_cluster_size,
                                 core,
                                 g,
                                 uf,
                                 cluster,
                                 cl_members,
                                 rows,
                                 lambda_birth,
                                 next_cl,
                                 shed};
        const detail::StepInfo step{s.first, s.second, w, lambda};

        if (!detail::try_fast_path(st, step, affected, cluster_points))
            detail::slow_path(st, step, cluster_points);
    }
    assert(cl_members.capacity() == cl_capacity0 && "cl_members reallocated");
    (void)cl_capacity0;
    final_cluster = cluster;
}

}  // namespace mixdex
