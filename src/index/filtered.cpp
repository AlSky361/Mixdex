#include <limits>

#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/utils/fm_heap.hpp"

namespace mixdex {
namespace {
void validate_filter(int n, std::span<const unsigned char> active, std::span<const int> excluded) {
    if (active.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument("active mask size does not match dataset");
    if (!std::is_sorted(excluded.begin(), excluded.end()))
        throw std::invalid_argument("excluded rows must be sorted");
    for (int i : excluded)
        if (i < 0 || i >= n)
            throw std::invalid_argument("excluded row out of range");
}
bool eligible(int i, std::span<const unsigned char> active, std::span<const int> excluded) {
    return active[i] && !std::binary_search(excluded.begin(), excluded.end(), i);
}
}  // namespace
std::vector<std::pair<int, double>> NeighbourSearch::knn_filtered(
    const RowView& q,
    int k,
    std::span<const unsigned char> active,
    std::span<const int> excluded) const {
    check_k(k);
    check_row(q);
    check_external_ready();
    validate_filter(n_rows(), active, excluded);
    FixedMaxHeap heap(std::min(k, n_rows()));
    for (int i = 0; i < n_rows(); ++i)
        if (eligible(i, active, excluded))
            heap.try_insert(distance(dataset(), q, i), i);
    return heap.sorted();
}
std::vector<std::pair<int, double>> BallTree::knn_filtered(const RowView& q,
                                                           int k,
                                                           std::span<const unsigned char> active,
                                                           std::span<const int> excluded) const {
    check_k(k);
    check_row(q);
    check_external_ready();
    validate_filter(n_rows(), active, excluded);
    if (root() < 0)
        return {};
    FixedMaxHeap heap(std::min(k, n_rows()));
    std::vector<int> stack{root()};
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        if (heap.full() && lower_bound_row(q, v) > heap.max_distance())
            continue;
        const auto& node = nodes()[v];
        if (node.is_leaf()) {
            for (int pos = node.low; pos < node.high; ++pos) {
                int i = indices()[pos];
                if (!eligible(i, active, excluded))
                    continue;
                double bound =
                    heap.full() ? heap.max_distance() : std::numeric_limits<double>::infinity();
                heap.try_insert(distance_bounded(dataset(), q, i, bound), i);
            }
        } else {
            bool left = lower_bound_row(q, node.left) <= lower_bound_row(q, node.right);
            stack.push_back(left ? node.right : node.left);
            stack.push_back(left ? node.left : node.right);
        }
    }
    return heap.sorted();
}
}  // namespace mixdex
