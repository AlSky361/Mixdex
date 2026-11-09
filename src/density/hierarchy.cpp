#include "mixdex/density/hierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "mixdex/density/core/core.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/index/graph/find_k/find_k.hpp"
#include "mixdex/index/graph/knn_graph/knn_graph.hpp"
#include "mixdex/mst/boruvka.hpp"
#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/utils/uf.hpp"

namespace mixdex {

DensityHierarchy::DensityHierarchy(const Dataset& data,
                                   int m_pts,
                                   HDBSCANMode mode,
                                   SearchMethod method,
                                   int leaf_size,
                                   int k_opt,
                                   bool deterministic_mst) :
    m_pts_(m_pts) {
    if (m_pts < 1)
        throw std::invalid_argument("DensityHierarchy: m_pts must be >= 1");
    const int n = data.n_rows();
    if (n == 0)
        return;

    std::unique_ptr<MutualReachabilitySearch> search = make_mreach_search(data, method, leaf_size);

    core_ = mixdex::core_distances(*search, m_pts);

    switch (mode) {
        case HDBSCANMode::Exact: {
            search->set_core_dist(core_);

            auto* bt = dynamic_cast<BallTree*>(search.get());
            if (!bt)
                throw std::invalid_argument(
                    "hdbscan Exact: DTBoruvka requires "
                    "SearchMethod::BallTree; got a different neighbor search method.");

            mst_ = dual_tree_boruvka(*bt, deterministic_mst);
            break;
        }

        case HDBSCANMode::Approximate: {
            const int k_max = static_cast<int>(std::sqrt(static_cast<double>(n)));
            Graph g_mr;

            if (k_opt == constants::kKUnset) {
                // Reuse mutual-reachability neighbor lists while searching prefix connectivity.
                if (k_max < m_pts)
                    throw std::invalid_argument(
                        "hdbscan Approximate: sqrt(n)=" + std::to_string(k_max) +
                        " < m_pts=" + std::to_string(m_pts) +
                        "; automatic k_opt selection is unavailable. Use Exact.");

                search->set_core_dist(core_);
                std::optional<AutoK> sel =
                    auto_select_k(*search, m_pts, k_max, constants::kApproxKMargin);
                if (!sel)
                    throw std::invalid_argument(
                        "hdbscan Approximate: kNN graph is disconnected for every k <= sqrt(n)=" +
                        std::to_string(k_max) +
                        " (automatic k_opt selection found no connected graph). For an exact MST, "
                        "use Exact.");
                g_mr = graph_from_knn_lists(sel->lists, sel->k_opt);
            } else {
                if (k_opt < m_pts)
                    throw std::invalid_argument(
                        "hdbscan Approximate: k_opt=" + std::to_string(k_opt) +
                        " < m_pts=" + std::to_string(m_pts) +
                        ". k_opt must be in [m_pts, sqrt(n)=" + std::to_string(k_max) + "]");

                if (k_opt > k_max)
                    throw std::invalid_argument(
                        "hdbscan Approximate: k_opt=" + std::to_string(k_opt) +
                        " > sqrt(n)=" + std::to_string(k_max) + ". For an exact MST, use Exact.");

                search->set_core_dist(core_);
                g_mr = build_knn_graph(*search, k_opt);

                if (!g_mr.is_connected()) {
                    const std::vector<int> comp = g_mr.compute_components();
                    const int n_comp =
                        comp.empty() ? 0 : *std::max_element(comp.begin(), comp.end()) + 1;
                    throw std::invalid_argument(
                        "hdbscan Approximate: kNN graph at k_opt=" + std::to_string(k_opt) +
                        " is disconnected (components: " + std::to_string(n_comp) +
                        "); its spanning forest is not a tree. Increase k_opt (at most "
                        "sqrt(n)=" +
                        std::to_string(k_max) +
                        "), omit k_opt for automatic selection, or use Exact.");
                }
            }

            mst_ = boruvka(g_mr);
            break;
        }
    }
}

std::vector<int> DensityHierarchy::dbscan_star(double eps, int min_cluster_size) const {
    if (!(eps >= 0.0))
        throw std::invalid_argument("dbscan_star: eps must be >= 0");
    if (min_cluster_size < 1)
        throw std::invalid_argument("dbscan_star: min_cluster_size must be >= 1");

    const int n = this->n();
    UF uf(n);
    for (const Edge& e : mst_)
        if (e.w <= eps)
            uf.unite(e.u, e.v);

    std::vector<int> comp_size(n, 0);
    for (int i = 0; i < n; ++i)
        if (core_[i] <= eps)
            ++comp_size[uf.find_compress(i)];

    std::vector<int> labels(n, -1);
    std::vector<int> root_label(n, -1);
    int next = 0;
    for (int i = 0; i < n; ++i) {
        if (core_[i] > eps)
            continue;
        const int r = uf.find_compress(i);
        if (comp_size[r] < min_cluster_size)
            continue;
        if (root_label[r] < 0)
            root_label[r] = next++;
        labels[i] = root_label[r];
    }
    return labels;
}

}  // namespace mixdex
