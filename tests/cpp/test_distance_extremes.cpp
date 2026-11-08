#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"

using namespace mixdex;

namespace {
void check(bool condition) {
    if (!condition)
        throw std::runtime_error("extreme-distance regression");
}

void near(double actual, double expected) {
    check(std::isfinite(actual));
    check(std::abs(actual / expected - 1.0) < 2e-14);
}

Dataset numeric(std::vector<double> values) {
    const auto n = values.size();
    return Dataset(
        std::move(values), {}, n, 1, 0, {FeatureType::Numeric}, {NumericType::Continuous}, {"x"});
}

void check_cutoffs(const Dataset& data) {
    const double full = distance(data, 0, 1);
    for (double bound : {0.0, full / 2, full, std::numeric_limits<double>::infinity()}) {
        const double limited = distance_bounded(data, data.row(0), 1, bound);
        check(!std::isnan(limited));
        if (limited <= bound)
            check(limited == full);
        else
            check(full > bound);
    }
}

void check_distance_recovery() {
    const double largest = std::numeric_limits<double>::max();
    auto weighted = numeric({0, 3});
    weighted.set_weights({largest / 2});
    near(distance(weighted, 0, 1), 3);
    near(distance_num_partial(weighted, 0, 1), 3);
    check(distance(weighted, 0, 0) == 0);
    check_cutoffs(weighted);

    Dataset summed({0, 0, 1e308, 1e308},
                   {},
                   2,
                   2,
                   0,
                   {FeatureType::Numeric, FeatureType::Numeric},
                   {NumericType::Continuous, NumericType::Continuous},
                   {"a", "b"});
    near(distance(summed, 0, 1), 1e308);
    near(distance_num_partial(summed, 0, 1), 1e308);
    check_cutoffs(summed);

    Dataset difference({-largest, largest},
                       {0, 1},
                       2,
                       1,
                       1,
                       {FeatureType::Numeric, FeatureType::Categorical},
                       {NumericType::Continuous},
                       {"x", "c"});
    difference.set_weights({1, 3});
    near(distance(difference, 0, 1), largest / 2);
    near(distance_num_partial(difference, 0, 1), largest / 2);
    check_cutoffs(difference);

    Dataset tail({0, 1.5},
                 {0, 1},
                 2,
                 1,
                 1,
                 {FeatureType::Numeric, FeatureType::Categorical},
                 {NumericType::Continuous},
                 {"x", "c"});
    tail.set_weights({largest / 2, largest / 2});
    near(distance(tail, 0, 1), 1.25);
    check_cutoffs(tail);

    const auto unrepresentable = numeric({-largest, largest});
    check(std::isinf(distance(unrepresentable, 0, 1)));
    check_cutoffs(unrepresentable);
}

void check_bounds() {
    std::vector<double> values;
    for (int i = 0; i < 24; ++i) values.push_back(i);
    auto data = numeric(std::move(values));
    data.set_weights({std::ldexp(1.0, 1020)});
    BallTree tree(data, 2);
    std::vector<double> core(data.n_rows(), 1.0);
    tree.set_core_dist(core);
    for (int q = 0; q < data.n_rows(); ++q) {
        for (int ni = 0; ni < static_cast<int>(tree.nodes().size()); ++ni) {
            const auto& node = tree.nodes()[ni];
            double minimum = std::numeric_limits<double>::infinity();
            for (int pos = node.low; pos < node.high; ++pos)
                minimum = std::min(minimum, distance(data, q, tree.indices()[pos]));
            check(tree.lower_bound(q, ni) <= minimum);
        }
    }
    for (int qi = 0; qi < static_cast<int>(tree.nodes().size()); ++qi)
        for (int ri = 0; ri < static_cast<int>(tree.nodes().size()); ++ri) {
            double minimum = std::numeric_limits<double>::infinity();
            const auto& q = tree.nodes()[qi];
            const auto& r = tree.nodes()[ri];
            for (int i = q.low; i < q.high; ++i)
                for (int j = r.low; j < r.high; ++j)
                    minimum = std::min(
                        minimum,
                        std::max(1.0, distance(data, tree.indices()[i], tree.indices()[j])));
            check(tree.lower_bound_mreach_node(qi, ri) <= minimum);
        }
}
}  // namespace

int main() {
    check_distance_recovery();
    check_bounds();
    return 0;
}
