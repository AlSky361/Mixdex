#include "mixdex/quantile.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

#include "mixdex/utils/constants.hpp"

using namespace mixdex::constants;

namespace mixdex {

namespace {

int checked_quantile_size(std::size_t size, QuantileMethod method) {
    if (!std::in_range<int>(size))
        throw std::length_error("quantile input exceeds the index range");
    if (method != QuantileMethod::InvertedCDF && method != QuantileMethod::MedianUnbiased)
        throw std::invalid_argument("unknown quantile method");
    return static_cast<int>(size);
}

// Quantile = (1 - gamma)*x[j0] + gamma*x[j1]; endpoints need no interpolation.
struct QuantilePlan {
    int j0;
    int j1;
    double gamma;
    bool direct;  // Use x[j0] without interpolation.
};

QuantilePlan plan_quantile(int n, double p, QuantileMethod method) {
    if (n == 0) {
        throw std::invalid_argument("compute_quantile: пустой массив");
    }
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        throw std::invalid_argument("compute_quantile: p не принадлежит [0,1]");
    }

    if (p < kEpsEq) {
        return {0, 0, 0.0, true};
    }

    if (p > 1.0 - kEpsEq) {
        return {n - 1, n - 1, 0.0, true};
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
            gamma = f_zero(g) ? 0.0 : 1.0;
            break;
        case QuantileMethod::MedianUnbiased:
            gamma = g;
            break;
    }

    assert(gamma >= 0 && gamma <= 1.0 && "compute_quantile: gamma не принадлежит [0; 1]");

    [[maybe_unused]] double p_left = (j - m) / n;
    [[maybe_unused]] double p_right = (j - m + 1) / n;

    assert(p_left <= p && p <= p_right && "compute_quantile: p вне [(j - m) / n; (j - m + 1) / n");

    return {std::clamp(j - 1, 0, n - 1), std::clamp(j, 0, n - 1), gamma, false};
}

double apply_plan(const QuantilePlan& plan, double x0, double x1) {
    return plan.direct ? x0 : (1.0 - plan.gamma) * x0 + plan.gamma * x1;
}

}  // namespace

std::vector<double> quantiles(std::span<const double> x_sorted,
                              const std::vector<double>& ps,
                              QuantileMethod method) {
    const int n = checked_quantile_size(x_sorted.size(), method);
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
    const int n = checked_quantile_size(x_work.size(), method);

    std::vector<QuantilePlan> plans;
    plans.reserve(ps.size());
    std::vector<int> needed;
    needed.reserve(2 * ps.size());
    for (double p : ps) {
        plans.push_back(plan_quantile(n, p, method));
        needed.push_back(plans.back().j0);
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
