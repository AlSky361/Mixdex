#pragma once
#include <functional>
#include <span>
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/fm_heap.hpp"
#include "mixdex/utils/omp_macros.hpp"

namespace mixdex {

/// Borrow Dataset storage; prune nodes using distance bounds (docs/distance-bounds.md).
class BallTree : public MutualReachabilitySearch {
public:
    struct Node {
        int center_idx{-1};

        double radius_num{0.0};  // Numeric contribution, normalized by total feature weight.

        int left{-1};
        int right{-1};
        int low{0};   // First row position in indices_.
        int high{0};  // Exclusive end position in indices_.

        NodeCatPresent cat_present;

        bool is_leaf() const noexcept {
            return left == -1 && right == -1;
        }
    };

    struct MReachNode {
        double radius{0.0};
        double min_core{0.0};
    };

    explicit BallTree(const Dataset& data, int leaf_size);
    std::vector<std::pair<int, double>> knn_filtered(
        const RowView& q,
        int k,
        std::span<const unsigned char> active,
        std::span<const int> excluded = {}) const override;

    std::vector<std::pair<int, double>> knn(int query_idx, int k) const override;
    std::vector<std::pair<int, double>> radius_query(int query_idx, double r) const override;
    std::vector<std::pair<int, double>> knn(const RowView& q, int k) const override;
    std::vector<std::pair<int, double>> radius_query(const RowView& q, double r) const override;
    bool has_neighbours_within(int query_idx, double r, int at_least) const override;
    void radius_ids(int query_idx, double r, std::vector<int>& out) const override;
    double kth_distance(int query_idx, int k) const override;
    double kth_distance(const RowView& q, int k) const override;
    const Dataset& dataset() const noexcept override {
        return data_;
    }
    void check_external_ready() const override {
        if (core_dist())
            throw std::invalid_argument("external queries are not supported in mreach mode");
    }

    int n_rows() const noexcept override {
        return data_.n_rows();
    }
    void set_core_dist(const std::vector<double>& cd) override;
    const std::vector<double>* core_dist() const noexcept {
        return metric_type_ == MetricType::MReach ? &core_copy_ : nullptr;
    }

    double lower_bound(int query_idx, int node_idx) const;
    double lower_bound_row(const RowView& q, int node_idx) const;
    double lower_bound_mreach(int query_idx, int node_idx) const;
    double lower_bound_mreach_node(int qnode_idx, int rnode_idx) const;

    double call_metric(int i, int j) const;

    const std::vector<Node>& nodes() const noexcept {
        return nodes_;
    }

    const std::vector<MReachNode>& mreach_nodes() const noexcept {
        return mreach_nodes_;
    }
    const std::vector<int>& indices() const noexcept {
        return indices_;
    }
    int root() const noexcept {
        return root_;
    }

    std::span<const double> bbox_lo(int node_idx) const noexcept {
        const std::size_t n_num = static_cast<std::size_t>(data_.n_num());
        return {bbox_lo_flat_.data() + static_cast<std::size_t>(node_idx) * n_num, n_num};
    }
    std::span<const double> bbox_hi(int node_idx) const noexcept {
        const std::size_t n_num = static_cast<std::size_t>(data_.n_num());
        return {bbox_hi_flat_.data() + static_cast<std::size_t>(node_idx) * n_num, n_num};
    }

private:
    const Dataset& data_;
    int leaf_size_;
    MetricType metric_type_{MetricType::Composite};
    std::vector<int> indices_;
    std::vector<Node> nodes_;
    std::vector<MReachNode> mreach_nodes_;
    std::vector<double> bbox_lo_flat_;
    std::vector<double> bbox_hi_flat_;
    int root_{0};
    std::vector<double> core_copy_;

    int build(int low, int high);

    double lb_node(const RowView& q, int query_idx, int node_idx) const;

    double dist_to(const RowView& q, int query_idx, int j, double bound) const;
    void knn_search(int node_idx, const RowView& q, int query_idx, FixedMaxHeap& heap) const;
    void radius_search(int node_idx,
                       const RowView& q,
                       int query_idx,
                       double r,
                       std::vector<std::pair<int, double>>& result) const;

    template <class Sink>
    void radius_traverse(
        int node_idx, const RowView& q, int query_idx, double r, Sink&& sink) const;
};

}  // namespace mixdex

#include "mixdex/index/ball_tree/ball_tree.ipp"
