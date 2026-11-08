#include "mixdex/quantile.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "mixdex/utils/checked_arithmetic.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/numeric_validation.hpp"

using namespace mixdex::constants;

namespace mixdex {

namespace {

void validate_quantile_method(QuantileMethod method) {
    if (method != QuantileMethod::InvertedCDF && method != QuantileMethod::MedianUnbiased)
        throw std::invalid_argument("unknown quantile method");
}

// Quantile = (1 - gamma)*x[j0] + gamma*x[j1]; endpoints need no interpolation.
struct QuantilePlan {
    int j0;
    int j1;
    double gamma;
};

QuantilePlan plan_quantile(int n, double p, QuantileMethod method) {
    if (n == 0) {
        throw std::invalid_argument("quantiles require nonempty input");
    }
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        throw std::invalid_argument("quantile probabilities must be finite and in [0, 1]");
    }

    if (p == 0.0) {
        return {0, 0, 0.0};
    }

    if (p == 1.0) {
        return {n - 1, n - 1, 0.0};
    }

    double m = 0.0;
    switch (method) {
        case QuantileMethod::InvertedCDF:
            m = 0.0;
            break;
        case QuantileMethod::MedianUnbiased:
            m = 1.0 / kQuantileType8Divisor + p / kQuantileType8Divisor;
            break;
    }

    const double raw_j = p * n + m;
    int j = static_cast<int>(std::floor(raw_j));
    double g = raw_j - j;

    double gamma = 0.0;
    switch (method) {
        case QuantileMethod::InvertedCDF:
            gamma = g == 0.0 ? 0.0 : 1.0;
            break;
        case QuantileMethod::MedianUnbiased:
            gamma = g;
            break;
    }

    assert(gamma >= 0.0 && gamma <= 1.0);

    const int j0 = std::clamp(j - 1, 0, n - 1);
    const int j1 = std::clamp(j, 0, n - 1);
    if (gamma == 0.0 || j0 == j1)
        return {j0, j0, 0.0};
    if (gamma == 1.0)
        return {j1, j1, 0.0};
    return {j0, j1, gamma};
}

double apply_plan(const QuantilePlan& plan, double x0, double x1) {
    if (!std::isfinite(x0) || !std::isfinite(x1))
        throw std::invalid_argument("quantiles require finite values");
    if (plan.j0 == plan.j1 || x0 == x1)
        return x0;
    // Preserve ordinary rounding; lerp restores finite, bounded interpolation.
    const double value = (1.0 - plan.gamma) * x0 + plan.gamma * x1;
    if (std::isfinite(value) && x0 <= value && value <= x1)
        return value;
    return std::lerp(x0, x1, plan.gamma);
}

}  // namespace

std::vector<double> quantiles(std::span<const double> x_sorted,
                              const std::vector<double>& ps,
                              QuantileMethod method) {
    const int n =
        detail::checked_index_size(x_sorted.size(), "quantile input exceeds the index range");
    validate_quantile_method(method);
    std::vector<double> result(ps.size());
    for (std::size_t i = 0; i < ps.size(); ++i) {
        const QuantilePlan plan = plan_quantile(n, ps[i], method);
        result[i] = apply_plan(plan, x_sorted[plan.j0], x_sorted[plan.j1]);
    }
    return result;
}

std::vector<double> quantiles_select(std::span<double> x_work,
                                     const std::vector<double>& ps,
                                     QuantileMethod method) {
    const int n =
        detail::checked_index_size(x_work.size(), "quantile input exceeds the index range");
    validate_quantile_method(method);
    detail::require_finite(x_work, "quantiles_select requires finite values");

    std::vector<QuantilePlan> plans;
    plans.reserve(ps.size());
    std::vector<int> needed;
    needed.reserve(ps.size());
    for (double p : ps) {
        plans.push_back(plan_quantile(n, p, method));
        needed.push_back(plans.back().j0);
        if (plans.back().j1 != plans.back().j0)
            needed.push_back(plans.back().j1);
    }
    std::sort(needed.begin(), needed.end());
    needed.erase(std::unique(needed.begin(), needed.end()), needed.end());

    // After nth_element, search later order statistics only in the right partition.
    int lo = 0;
    for (int idx : needed) {
        std::nth_element(x_work.begin() + lo, x_work.begin() + idx, x_work.end());
        lo = idx + 1;
    }

    std::vector<double> result(ps.size());
    for (std::size_t i = 0; i < ps.size(); ++i) {
        result[i] = apply_plan(plans[i], x_work[plans[i].j0], x_work[plans[i].j1]);
    }
    return result;
}

}  // namespace mixdex
