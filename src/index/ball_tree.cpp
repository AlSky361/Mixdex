// Method references: docs/references.md#spanning-trees-and-dual-tree-search
#include "mixdex/index/ball_tree/ball_tree.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/utils/constants.hpp"

using namespace mixdex::constants;

namespace mixdex {
namespace {

bool best_num_axis_scored(const Dataset& data,
                          const std::vector<int>& idx,
                          int low,
                          int high,
                          int* out_axis,
                          double* out_score) {
    const int n_num = data.n_num();
    if (n_num == 0)
        return false;
    const int n = high - low;
    const double pair_count = static_cast<double>(n) * (n - 1) / 2.0;

    int best_axis = 0;
    double best_score = -1.0;

    std::vector<double> vals(n);

    for (int k = 0; k < n_num; ++k) {
        auto col = data.col_num_span(k);

        for (int i = 0; i < n; ++i) {
            vals[i] = col[idx[low + i]];
        }
        std::sort(vals.begin(), vals.end());

        double pairwise_sum = 0.0;
        for (int m = 1; m <= n; ++m) {
            pairwise_sum += (2.0 * m - n - 1) * vals[m - 1];
        }

        double w_k = data.has_weights() ? data.weight(k) : 1.0;
        double score = (pair_count > 0.0) ? (w_k * pairwise_sum / pair_count) : 0.0;

        if (score > best_score) {
            best_score = score;
            best_axis = k;
        }
    }

    *out_axis = best_axis;
    *out_score = best_score;
    return true;
}

bool best_oth_axis_scored(const Dataset& data,
                          const std::vector<int>& idx,
                          int low,
                          int high,
                          int* out_axis,
                          double* out_score) {
    const int n_cat = data.n_cat();
    if (n_cat == 0)
        return false;
    const int n = high - low;
    const int n_num = data.n_num();

    int best_axis = -1;
    double best_score = -1.0;

    std::vector<int> counts;
    for (int k = 0; k < n_cat; ++k) {
        auto col = data.col_cat_span(k);
        std::size_t card = data.categories_size(k);
        counts.assign(card, 0);
        for (int i = low; i < high; ++i) {
            int c = col[idx[i]];
            if (c >= 0 && static_cast<std::size_t>(c) < card)
                counts[static_cast<std::size_t>(c)]++;
        }

        double gini = 1.0;
        for (int c : counts) {
            double p = static_cast<double>(c) / n;
            gini -= p * p;
        }

        double w_k = data.has_weights() ? data.weight(n_num + k) : 1.0;
        double score = w_k * gini;

        if (score > best_score) {
            best_score = score;
            best_axis = k;
        }
    }

    if (best_axis < 0)
        return false;

    *out_axis = best_axis;
    *out_score = best_score;
    return true;
}

void fill_cat_present(
    const Dataset& data, const std::vector<int>& idx, BallTree::Node& node, int low, int high) {
    node.cat_present.build(data, idx, low, high);
}

void fill_bbox(const Dataset& data,
               const std::vector<int>& idx,
               std::span<double> node_lo,
               std::span<double> node_hi,
               int low,
               int high) {
    const int n_num = data.n_num();

    for (int k = 0; k < n_num; ++k) {
        auto col = data.col_num_span(k);
        double mn = col[idx[low]];
        double mx = mn;
        for (int i = low + 1; i < high; ++i) {
            double v = col[idx[i]];
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        node_lo[k] = mn;
        node_hi[k] = mx;
    }
}

int find_center(const Dataset& data, const std::vector<int>& idx, int low, int high) {
    const int n_num = data.n_num();
    const int n_cat = data.n_cat();

    const int n = high - low;

    std::vector<double> mean(n_num, 0.0);
    for (int i = low; i < high; ++i) {
        auto row = data.row_num_span(idx[i]);
        for (int k = 0; k < n_num; ++k) mean[k] += row[k];
    }
    for (double& v : mean) v /= n;

    std::vector<int> mode(n_cat, -1);
    std::vector<int> counts;

    for (int k = 0; k < n_cat; ++k) {
        auto col = data.col_cat_span(k);
        counts.assign(data.categories_size(k), 0);

        if (counts.empty())
            continue;

        for (int i = low; i < high; ++i) {
            int c = col[idx[i]];
            if (c >= 0 && static_cast<std::size_t>(c) < counts.size())
                counts[static_cast<std::size_t>(c)]++;
        }
        int best = 0;
        for (std::size_t c = 1; c < counts.size(); ++c)
            if (counts[c] > counts[static_cast<std::size_t>(best)])
                best = static_cast<int>(c);

        mode[k] = best;
    }

    int best_pt = idx[low];
    double best_dist = kInf;

    for (int i = low; i < high; ++i) {
        auto row = data.row_num_span(idx[i]);
        double d = 0.0;

        for (int k = 0; k < n_num; ++k) d += data.weight(k) * std::abs(row[k] - mean[k]);

        auto oth = data.row_cat_span(idx[i]);
        for (int k = 0; k < n_cat; ++k)
            if (oth[k] != mode[k])
                d += data.weight(n_num + k);

        if (d < best_dist) {
            best_dist = d;
            best_pt = idx[i];
        }
    }

    return best_pt;
}

void merge_bbox(std::span<double> parent_lo,
                std::span<double> parent_hi,
                std::span<const double> left_lo,
                std::span<const double> left_hi,
                std::span<const double> right_lo,
                std::span<const double> right_hi) {
    const int n_num = static_cast<int>(left_lo.size());
    for (int k = 0; k < n_num; ++k) {
        parent_lo[k] = std::min(left_lo[k], right_lo[k]);
        parent_hi[k] = std::max(left_hi[k], right_hi[k]);
    }
}

void merge_cat_present(BallTree::Node& parent,
                       const BallTree::Node& left,
                       const BallTree::Node& right) {
    parent.cat_present = NodeCatPresent::merge(left.cat_present, right.cat_present);
}

}  // namespace

int BallTree::build(int low, int high) {
    if (nodes_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("BallTree: node count exceeds int capacity");
    const int ni = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    nodes_[ni].low = low;
    nodes_[ni].high = high;

    const int n_num = static_cast<int>(data_.n_num());
    bbox_lo_flat_.resize(bbox_lo_flat_.size() + n_num);
    bbox_hi_flat_.resize(bbox_hi_flat_.size() + n_num);
    std::span<double> node_lo(bbox_lo_flat_.data() + static_cast<std::size_t>(ni) * n_num, n_num);
    std::span<double> node_hi(bbox_hi_flat_.data() + static_cast<std::size_t>(ni) * n_num, n_num);

    const int center = find_center(data_, indices_, low, high);
    nodes_[ni].center_idx = center;

    double radius_num = 0.0;
    for (int i = low; i < high; ++i) {
        double d_num = distance_num_partial(data_, center, indices_[i]);
        if (d_num > radius_num)
            radius_num = d_num;
    }
    nodes_[ni].radius_num = radius_num;

    fill_bbox(data_, indices_, node_lo, node_hi, low, high);
    fill_cat_present(data_, indices_, nodes_[ni], low, high);

    if (high - low <= leaf_size_)
        return ni;

    const int mid = low + (high - low) / 2;

    int num_axis = 0, cat_axis = 0;
    double num_score = -1.0, cat_score = -1.0;
    bool has_num = best_num_axis_scored(data_, indices_, low, high, &num_axis, &num_score);
    bool has_cat = best_oth_axis_scored(data_, indices_, low, high, &cat_axis, &cat_score);

    bool use_num = has_num && (!has_cat || num_score >= cat_score);

    if (use_num) {
        auto col = data_.col_num_span(num_axis);
        std::nth_element(indices_.begin() + low,
                         indices_.begin() + mid,
                         indices_.begin() + high,
                         [&](int a, int b) {
                             return col[a] < col[b];
                         });
    } else if (has_cat) {
        auto col = data_.col_cat_span(cat_axis);
        std::nth_element(indices_.begin() + low,
                         indices_.begin() + mid,
                         indices_.begin() + high,
                         [&](int a, int b) {
                             return col[a] < col[b];
                         });
    }

    const int left_ni = build(low, mid);
    const int right_ni = build(mid, high);

    std::span<double> parent_lo(bbox_lo_flat_.data() + static_cast<std::size_t>(ni) * n_num, n_num);
    std::span<double> parent_hi(bbox_hi_flat_.data() + static_cast<std::size_t>(ni) * n_num, n_num);

    nodes_[ni].left = left_ni;
    nodes_[ni].right = right_ni;

    merge_bbox(parent_lo,
               parent_hi,
               bbox_lo(left_ni),
               bbox_hi(left_ni),
               bbox_lo(right_ni),
               bbox_hi(right_ni));
    merge_cat_present(nodes_[ni], nodes_[left_ni], nodes_[right_ni]);

    return ni;
}

BallTree::BallTree(const Dataset& data, int leaf_size) : data_(data), leaf_size_(leaf_size) {
    if (leaf_size <= 0)
        throw std::invalid_argument("BallTree: leaf_size <= 0");

    const int n = data.n_rows();
    if (n == 0)
        return;

    indices_.resize(n);
    std::iota(indices_.begin(), indices_.end(), 0);

    const std::size_t n_leaves = (static_cast<std::size_t>(n) + leaf_size - 1) / leaf_size;
    const int n_num = static_cast<int>(data.n_num());
    const std::size_t max_nodes =
        std::min(kTreeNodesPerLeafBound * n_leaves + kTreeNodesReserveSlack,
                 static_cast<std::size_t>(std::numeric_limits<int>::max()));
    nodes_.reserve(max_nodes);
    bbox_lo_flat_.reserve(static_cast<std::size_t>(max_nodes) * n_num);
    bbox_hi_flat_.reserve(static_cast<std::size_t>(max_nodes) * n_num);

    root_ = build(0, n);
}

}  // namespace mixdex
