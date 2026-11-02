#pragma once
#include <atomic>
#include <cstdlib>
#include <limits>

#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/utils/aliases.hpp"
#include "mixdex/utils/checked_arithmetic.hpp"

namespace mixdex {

// Count kNN node visits only when MIXDEX_PROFILE_KNN is set at startup.
inline std::atomic<Count> g_knn_nodes_visited{0};
inline bool knn_profile_enabled() {
    static bool enabled = std::getenv("MIXDEX_PROFILE_KNN") != nullptr;
    return enabled;
}

inline double BallTree::call_metric(int i, int j) const {
    const auto distance = [this](int a, int b) {
        return mixdex::distance(data_, a, b);
    };
    if (metric_type_ == MetricType::MReach)
        return mreach_distance(*core_dist(), i, j, distance);
    return distance(i, j);
}

inline double BallTree::dist_to(const RowView& q, int query_idx, int j, double bound) const {
    if (metric_type_ == MetricType::MReach) {
        const auto distance = [this, &q, bound](int, int b) {
            return distance_bounded(data_, q, b, bound);
        };
        return mreach_distance(*core_dist(), query_idx, j, distance);
    }
    return distance_bounded(data_, q, j, bound);
}

inline void BallTree::knn_search(int node_idx,
                                 const RowView& q,
                                 int query_idx,
                                 FixedMaxHeap& heap) const {
    struct StackEntry {
        int node_idx;
        bool has_guard;
        double guard_lb;
    };
    thread_local std::vector<StackEntry> stack;
    stack.clear();
    stack.push_back({node_idx, false, 0.0});

    while (!stack.empty()) {
        StackEntry entry = stack.back();
        stack.pop_back();

        if (entry.has_guard && heap.full() && entry.guard_lb > heap.max_distance())
            continue;

        if (knn_profile_enabled())
            detail::increment_count(g_knn_nodes_visited);

        int ni = entry.node_idx;
        const Node& node = nodes()[ni];

        double lb = lb_node(q, query_idx, ni);
        if (heap.full() && lb > heap.max_distance())
            continue;

        if (node.is_leaf()) {
            for (int i = node.low; i < node.high; ++i) {
                int idx = indices()[i];
                if (idx == query_idx)
                    continue;
                const double bound =
                    heap.full() ? heap.max_distance() : std::numeric_limits<double>::infinity();
                double d = dist_to(q, query_idx, idx, bound);
                heap.try_insert(d, idx);
            }
            continue;
        }

        double lb_left = lb_node(q, query_idx, node.left);
        double lb_right = lb_node(q, query_idx, node.right);

        int first = (lb_left <= lb_right) ? node.left : node.right;
        int second = (lb_left <= lb_right) ? node.right : node.left;
        double lb_second = (lb_left <= lb_right) ? lb_right : lb_left;

        stack.push_back({second, true, lb_second});
        stack.push_back({first, false, 0.0});
    }
}

template <class Sink>
inline void BallTree::radius_traverse(
    int node_idx, const RowView& q, int query_idx, double r, Sink&& sink) const {
    thread_local std::vector<int> stack;
    stack.clear();
    stack.push_back(node_idx);

    while (!stack.empty()) {
        int ni = stack.back();
        stack.pop_back();

        const Node& node = nodes()[ni];

        double lb = lb_node(q, query_idx, ni);
        if (lb > r)
            continue;

        if (node.is_leaf()) {
            for (int i = node.low; i < node.high; ++i) {
                int idx = indices()[i];
                if (idx == query_idx)
                    continue;
                double d = dist_to(q, query_idx, idx, r);
                if (d <= r && !sink(idx, d))
                    return;
            }
            continue;
        }

        stack.push_back(node.left);
        stack.push_back(node.right);
    }
}

inline void BallTree::radius_search(int node_idx,
                                    const RowView& q,
                                    int query_idx,
                                    double r,
                                    std::vector<std::pair<int, double>>& result) const {
    radius_traverse(node_idx, q, query_idx, r, [&result](int idx, double d) {
        result.emplace_back(idx, d);
        return true;
    });
}

namespace detail {
inline FixedMaxHeap& thread_heap(int k) {
    thread_local FixedMaxHeap heap(0);
    heap.reset(static_cast<std::size_t>(k));
    return heap;
}
}  // namespace detail

inline std::vector<std::pair<int, double>> BallTree::knn(int query_idx, int k) const {
    check_k(k);
    FixedMaxHeap& heap = detail::thread_heap(k);
    knn_search(root(), data_.row(query_idx), query_idx, heap);
    return heap.sorted();
}

inline double BallTree::kth_distance(int query_idx, int k) const {
    check_k(k);
    FixedMaxHeap& heap = detail::thread_heap(k);
    knn_search(root(), data_.row(query_idx), query_idx, heap);
    return heap.farthest();
}

inline bool BallTree::has_neighbours_within(int query_idx, double r, int at_least) const {
    check_radius(r);
    if (at_least <= 0)
        return true;
    int count = 0;
    radius_traverse(root(), data_.row(query_idx), query_idx, r, [&count, at_least](int, double) {
        return ++count < at_least;
    });
    return count >= at_least;
}

inline void BallTree::radius_ids(int query_idx, double r, std::vector<int>& out) const {
    check_radius(r);
    out.clear();
    radius_traverse(root(), data_.row(query_idx), query_idx, r, [&out](int idx, double) {
        out.push_back(idx);
        return true;
    });
}

inline std::vector<std::pair<int, double>> BallTree::radius_query(int query_idx, double r) const {
    check_radius(r);
    std::vector<std::pair<int, double>> result;
    radius_search(root(), data_.row(query_idx), query_idx, r, result);
    sort_neighbours(result);
    return result;
}

inline std::vector<std::pair<int, double>> BallTree::knn(const RowView& q, int k) const {
    check_row(q);
    check_external_ready();
    check_k(k);
    FixedMaxHeap& heap = detail::thread_heap(k);
    knn_search(root(), q, -1, heap);
    return heap.sorted();
}

inline double BallTree::kth_distance(const RowView& q, int k) const {
    check_row(q);
    check_external_ready();
    check_k(k);
    FixedMaxHeap& heap = detail::thread_heap(k);
    knn_search(root(), q, -1, heap);
    return heap.farthest();
}

inline std::vector<std::pair<int, double>> BallTree::radius_query(const RowView& q,
                                                                  double r) const {
    check_row(q);
    check_external_ready();
    check_radius(r);
    std::vector<std::pair<int, double>> result;
    radius_search(root(), q, -1, r, result);
    sort_neighbours(result);
    return result;
}

}  // namespace mixdex
