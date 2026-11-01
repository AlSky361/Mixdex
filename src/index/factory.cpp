#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/index/brute_force/brute_force.hpp"
#include "mixdex/index/neighbours.hpp"

namespace mixdex {

std::unique_ptr<MutualReachabilitySearch> make_mreach_search(const Dataset& data,
                                                             SearchMethod method,
                                                             int leaf_size) {
    if (method == SearchMethod::BruteForce) {
        auto metric = [&](int i, int j) {
            return mixdex::distance(data, i, j);
        };
        return std::make_unique<BruteForceSearch<decltype(metric)>>(data, metric);
    }

    return std::make_unique<BallTree>(data, leaf_size);
}

std::unique_ptr<NeighbourSearch> make_search(const Dataset& data,
                                             SearchMethod method,
                                             int leaf_size) {
    return make_mreach_search(data, method, leaf_size);
}

}  // namespace mixdex
