#include "mixdex/distance.hpp"

#include <cassert>
#include <cmath>

#include "mixdex/utils/aliases.hpp"

namespace mixdex {
namespace {
double categorical_component(const Dataset& data, const RowView& q, int k, Code stored) noexcept {
    if (!q.cat_sets.empty() && data.category_sets().contains(k))
        return data.categorical_distance(k, q.cat_sets[k], stored);
    return data.categorical_distance(k, q.cat[k], stored);
}
}  // namespace

double distance(const Dataset& data, const RowView& q, int j) noexcept {
    assert(j >= 0 && j < data.n_rows());
    assert(q.num.size() == static_cast<std::size_t>(data.n_num()) &&
           q.cat.size() == static_cast<std::size_t>(data.n_cat()));

    const Idx n_num = data.n_num();
    const Idx n_cat = data.n_cat();
    const Idx d = n_num + n_cat;

    if (d == 0)
        return 0.0;

    const double inv_W = data.has_weights() ? data.inv_weights_sum() : 1.0 / static_cast<double>(d);

    auto x_num_j = data.row_num_span(j);
    auto x_cat_j = data.row_cat_span(j);

    double sum = 0.0;

    for (Idx k = 0; k < n_num; ++k) {
        double w_k = data.weight(k);
        sum += w_k * std::abs(q.num[k] - x_num_j[k]);
    }

    for (Idx k = 0; k < n_cat; ++k) {
        double w_k = data.weight(n_num + k);
        sum += w_k * categorical_component(data, q, k, x_cat_j[k]);
    }

    return sum * inv_W;
}

namespace {

// Check the bound after the numeric block to avoid a branch per coordinate.
template <bool Weighted>
inline double distance_bounded_impl(
    const Dataset& data, const RowView& q, int j, double bound, double inv_W) noexcept {
    const Idx n_num = data.n_num();
    const Idx n_cat = data.n_cat();
    const auto x_num = data.row_num_span(j);
    const auto x_cat = data.row_cat_span(j);

    double sum = 0.0;
    for (Idx k = 0; k < n_num; ++k) {
        const double diff = std::abs(q.num[k] - x_num[k]);
        if constexpr (Weighted)
            sum += data.weight(k) * diff;
        else
            sum += diff;
    }
    if (n_cat > 0 && sum * inv_W > bound)
        return sum * inv_W;

    for (Idx k = 0; k < n_cat; ++k) {
        const double neq = categorical_component(data, q, k, x_cat[k]);
        if constexpr (Weighted)
            sum += data.weight(n_num + k) * neq;
        else
            sum += neq;
    }
    return sum * inv_W;
}

}  // namespace

double distance_bounded(const Dataset& data, const RowView& q, int j, double bound) noexcept {
    assert(j >= 0 && j < data.n_rows());
    assert(q.num.size() == static_cast<std::size_t>(data.n_num()) &&
           q.cat.size() == static_cast<std::size_t>(data.n_cat()));

    const Idx d = data.n_num() + data.n_cat();
    if (d == 0)
        return 0.0;

    const double inv_W = data.has_weights() ? data.inv_weights_sum() : 1.0 / static_cast<double>(d);
    return data.has_weights() ? distance_bounded_impl<true>(data, q, j, bound, inv_W)
                              : distance_bounded_impl<false>(data, q, j, bound, inv_W);
}

double distance(const Dataset& data, int i, int j) noexcept {
    assert(i >= 0 && i < data.n_rows());
    return mixdex::distance(data, data.row(i), j);
}

double distance_num_partial(const Dataset& data, int i, int j) noexcept {
    assert(i >= 0 && i < data.n_rows());
    assert(j >= 0 && j < data.n_rows());

    const Idx n_num = data.n_num();
    const Idx n_cat = data.n_cat();
    const Idx d = n_num + n_cat;

    if (d == 0)
        return 0.0;

    const double inv_W = data.has_weights() ? data.inv_weights_sum() : 1.0 / static_cast<double>(d);

    auto x_num_i = data.row_num_span(i);
    auto x_num_j = data.row_num_span(j);

    double sum = 0.0;

    for (Idx k = 0; k < n_num; ++k) {
        double w_k = data.weight(k);
        sum += w_k * std::abs(x_num_i[k] - x_num_j[k]);
    }

    return sum * inv_W;
}

}  // namespace mixdex
