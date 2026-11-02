// Method references: docs/references.md#spanning-trees-and-dual-tree-search
#include <algorithm>

#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/utils/constants.hpp"

using namespace mixdex::constants;

namespace mixdex {

void BallTree::set_core_dist(const std::vector<double>& cd) {
    check_core_distances(cd);
    // Prepare both buffers before changing the active metric.
    std::vector<double> core = cd;
    std::vector<MReachNode> summaries(nodes_.size());
    const auto ordinary = [this](int i, int j) {
        return mixdex::distance(data_, i, j);
    };
    for (int ni = static_cast<int>(nodes_.size()) - 1; ni >= 0; --ni) {
        const Node& node = nodes_[ni];
        auto& summary = summaries[ni];
        summary.min_core = node.is_leaf() ? kInf
                                          : std::min(summaries[node.left].min_core,
                                                     summaries[node.right].min_core);
        for (int pos = node.low; pos < node.high; ++pos) {
            const int point = indices_[pos];
            if (node.is_leaf())
                summary.min_core = std::min(summary.min_core, core[point]);
            summary.radius =
                std::max(summary.radius, mreach_distance(core, node.center_idx, point, ordinary));
        }
    }
    core_copy_ = std::move(core);
    mreach_nodes_ = std::move(summaries);
    metric_type_ = MetricType::MReach;
}

namespace {

inline double rounding_gamma(Idx m) {
    const double mu = static_cast<double>(m) * kUnitRoundoff;
    return mu / (1.0 - mu);
}

double distance_lb(const Dataset& data,
                   const RowView& q_row,
                   const BallTree::Node& node,
                   std::span<const double> bbox_lo,
                   std::span<const double> bbox_hi) {
    const Idx n_num = data.n_num();
    const Idx n_cat = data.n_cat();
    const Idx d = n_num + n_cat;
    if (d == 0)
        return 0.0;

    const double inv_W = data.has_weights() ? data.inv_weights_sum() : 1.0 / static_cast<double>(d);
    const auto q_num = q_row.num;
    const auto q_cat = q_row.cat;

    double bbox_sum = 0.0;
    for (Idx k = 0; k < n_num; ++k) {
        const double q = q_num[k];
        const double delta = std::max({0.0, q - bbox_hi[k], bbox_lo[k] - q});
        bbox_sum += data.weight(k) * delta;
    }

    double cat_sum = 0.0;
    for (Idx k = 0; k < n_cat; ++k) {
        if (!data.category_sets().contains(k) && !node.cat_present[k].contains(q_cat[k])) {
            const double w = data.weight(n_num + k);
            bbox_sum += w;
            cat_sum += w;
        }
    }
    const double lb_bbox = bbox_sum * inv_W;

    const auto c_num = data.row_num_span(node.center_idx);
    double ball_sum = 0.0;
    for (Idx k = 0; k < n_num; ++k) ball_sum += data.weight(k) * std::abs(q_num[k] - c_num[k]);
    const double ball = ball_sum * inv_W;
    const double cat = cat_sum * inv_W;
    const double slack = rounding_gamma(d + kBallBoundRoundingOps) * (ball + node.radius_num + cat);
    const double lb_ball = std::max(0.0, ball - node.radius_num - slack) + cat;

    return std::max(lb_ball, lb_bbox);
}

}  // namespace

double BallTree::lower_bound_row(const RowView& q, int node_idx) const {
    return distance_lb(data_, q, nodes()[node_idx], bbox_lo(node_idx), bbox_hi(node_idx));
}

double BallTree::lower_bound(int query_idx, int node_idx) const {
    return lower_bound_row(data_.row(query_idx), node_idx);
}

double BallTree::lb_node(const RowView& q, int query_idx, int node_idx) const {
    return core_dist() ? lower_bound_mreach(query_idx, node_idx) : lower_bound_row(q, node_idx);
}

double BallTree::lower_bound_mreach(int query_idx, int node_idx) const {
    double lb_g = lower_bound(query_idx, node_idx);
    double lb_cq = (*core_dist())[query_idx];
    double lb_cn = mreach_nodes_[node_idx].min_core;
    return std::max({lb_g, lb_cq, lb_cn});
}

double BallTree::lower_bound_mreach_node(int qnode_idx, int rnode_idx) const {
    const BallTree::Node& qnode = nodes()[qnode_idx];
    const BallTree::Node& rnode = nodes()[rnode_idx];
    const MReachNode& qmr = mreach_nodes_[qnode_idx];
    const MReachNode& rmr = mreach_nodes_[rnode_idx];

    const Idx n_num = data_.n_num();
    const Idx n_cat = data_.n_cat();
    const Idx d = n_num + n_cat;
    const double min_core = std::max(qmr.min_core, rmr.min_core);
    if (d == 0)
        return min_core;

    const double inv_W =
        data_.has_weights() ? data_.inv_weights_sum() : 1.0 / static_cast<double>(d);

    auto q_lo = bbox_lo(qnode_idx);
    auto q_hi = bbox_hi(qnode_idx);
    auto r_lo = bbox_lo(rnode_idx);
    auto r_hi = bbox_hi(rnode_idx);

    double bbox_sum = 0.0;
    for (Idx k = 0; k < n_num; ++k) {
        const double delta = std::max({0.0, r_lo[k] - q_hi[k], q_lo[k] - r_hi[k]});
        bbox_sum += data_.weight(k) * delta;
    }

    double cat_sum = 0.0;
    for (Idx k = 0; k < n_cat; ++k) {
        if (!data_.category_sets().contains(k) &&
            !qnode.cat_present[k].intersects(rnode.cat_present[k])) {
            const double w = data_.weight(n_num + k);
            bbox_sum += w;
            cat_sum += w;
        }
    }
    const double lb_bbox = bbox_sum * inv_W;
    const double cat = cat_sum * inv_W;

    const double gamma = rounding_gamma(d + kNodePairRoundingOps);

    const double d_centers_num = distance_num_partial(data_, qnode.center_idx, rnode.center_idx);
    const double slack1 = gamma * (d_centers_num + qnode.radius_num + rnode.radius_num + cat);
    const double lb1 =
        std::max(0.0, d_centers_num - qnode.radius_num - rnode.radius_num - slack1) + cat;

    const double d_centers_mreach = call_metric(qnode.center_idx, rnode.center_idx);
    const double slack2 = gamma * (d_centers_mreach + qmr.radius + rmr.radius);
    // Jaccard adds division rounding not covered by the legacy full-metric ball guard.
    const double lb2 = data_.category_sets().empty()
                           ? std::max(0.0, d_centers_mreach - qmr.radius - rmr.radius - slack2)
                           : 0.0;

    return std::max({lb1, lb2, lb_bbox, min_core});
}

}  // namespace mixdex
