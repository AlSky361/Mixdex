#include <cstdlib>

#include "mixdex/index/ball_tree/ball_tree.hpp"

mixdex::BallTree copied_tree(const mixdex::Dataset& data) {
    mixdex::BallTree original(data, 2);
    original.set_core_dist({0.2, 0.3, 0.4});
    auto copy = original;
    return copy;
}

int main() {
    mixdex::Dataset data({0., 1., 2.},
                         {},
                         3,
                         1,
                         0,
                         {mixdex::FeatureType::Numeric},
                         {mixdex::NumericType::Continuous},
                         {"x"});
    auto tree = copied_tree(data);
    if (tree.call_metric(0, 1) != 1.0 || tree.core_dist()->at(2) != 0.4)
        return EXIT_FAILURE;
    auto moved = std::move(tree);
    return moved.call_metric(0, 1) == 1.0 && moved.core_dist()->at(2) == 0.4 ? EXIT_SUCCESS
                                                                             : EXIT_FAILURE;
}
