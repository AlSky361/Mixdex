#include "mixdex/distance.hpp"

#include <cassert>
#include <cmath>

#include "distance_range.hpp"
#include "mixdex/utils/aliases.hpp"

namespace mixdex {
namespace {
double categorical_component(const Dataset& data, const RowView& q, int k, Code stored) noexcept {
    if (!q.cat_sets.empty() && data.category_sets().contains(k))
        return data.categorical_distance(k, q.cat_sets[k], stored);
    return data.categorical_distance(k, q.cat[k], stored);
}

// Accumulate nonnegative products without materializing an oversized numerator.
class ScaledPositiveSum {
public:
    void add_product(double a, double b) noexcept {
        if (a == 0.0 || b == 0.0)
            return;
        int a_exponent = 0, b_exponent = 0;
        const double a_fraction = std::frexp(a, &a_exponent);
        const double b_fraction = std::frexp(b, &b_exponent);
        add_scaled(a_fraction * b_fraction, a_exponent + b_exponent);
    }

    void add_difference(double weight, double a, double b) noexcept {
        const double difference = std::abs(a - b);
        if (std::isfinite(difference)) {
            add_product(weight, difference);
        } else {
            // Finite operands can overflow subtraction only with opposite signs.
            add_product(weight, std::abs(a));
            add_product(weight, std::abs(b));
        }
    }

    double normalized(double inverse) const noexcept {
        int exponent = 0;
        const double fraction = std::frexp(inverse, &exponent);
        return std::ldexp(fraction_ * fraction, exponent_ + exponent);
    }

private:
    void add_scaled(double fraction, int exponent) noexcept {
        if (fraction_ == 0.0) {
            fraction_ = fraction;
            exponent_ = exponent;
        } else if (exponent > exponent_) {
            fraction_ = std::ldexp(fraction_, exponent_ - exponent) + fraction;
            exponent_ = exponent;
        } else {
            fraction_ += std::ldexp(fraction, exponent - exponent_);
        }
        int shift = 0;
        fraction_ = std::frexp(fraction_, &shift);
        exponent_ += shift;
    }

    double fraction_{0.0};
    int exponent_{0};
};

template <bool NumericOnly = false>
double scaled_distance(const Dataset& data, const RowView& q, int j, double inv_W) noexcept {
    ScaledPositiveSum sum;
    const auto num = data.row_num_span(j);
    for (int k = 0; k < data.n_num(); ++k) sum.add_difference(data.weight(k), q.num[k], num[k]);
    if constexpr (!NumericOnly) {
        const auto cat = data.row_cat_span(j);
        for (int k = 0; k < data.n_cat(); ++k)
            sum.add_product(data.weight(data.n_num() + k),
                            categorical_component(data, q, k, cat[k]));
    }
    return sum.normalized(inv_W);
}
}  // namespace

double distance(const Dataset& data, const RowView& q, int j) noexcept {
    assert(j >= 0 && j < data.n_rows());
    assert(q.num.size() == static_cast<std::size_t>(data.n_num()) &&
           q.cat.size() == static_cast<std::size_t>(data.n_cat()));
    assert(q.cat_sets.empty() || q.cat_sets.size() == q.cat.size());

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

    const double result = sum * inv_W;
    return std::isfinite(result) ? result : scaled_distance(data, q, j, inv_W);
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
    const double numeric = sum * inv_W;
    if (!std::isfinite(numeric))
        return distance(data, q, j);
    if (n_cat > 0 && numeric > bound) {
        if (detail::distance_tail_fits(sum, static_cast<std::size_t>(n_cat), inv_W, Weighted))
            return numeric;
        return distance(data, q, j);
    }

    for (Idx k = 0; k < n_cat; ++k) {
        const double neq = categorical_component(data, q, k, x_cat[k]);
        if constexpr (Weighted)
            sum += data.weight(n_num + k) * neq;
        else
            sum += neq;
    }
    const double result = sum * inv_W;
    return std::isfinite(result) ? result : scaled_distance(data, q, j, inv_W);
}

}  // namespace

double distance_bounded(const Dataset& data, const RowView& q, int j, double bound) noexcept {
    assert(j >= 0 && j < data.n_rows());
    assert(q.num.size() == static_cast<std::size_t>(data.n_num()) &&
           q.cat.size() == static_cast<std::size_t>(data.n_cat()));
    assert(q.cat_sets.empty() || q.cat_sets.size() == q.cat.size());

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

    const double result = sum * inv_W;
    return std::isfinite(result) ? result : scaled_distance<true>(data, data.row(i), j, inv_W);
}

}  // namespace mixdex
