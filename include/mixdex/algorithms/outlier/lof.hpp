#pragma once
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {

/// Local Outlier Factor: compare local reachability density with neighboring densities.
class LOF {
public:
    LOF(const Dataset& data,
        int k,
        SearchMethod method = SearchMethod::BallTree,
        int leaf_size = constants::kDefaultLeafSize);

    LOF(std::shared_ptr<const NeighbourSearch> search, int k);

    const Dataset& dataset() const noexcept {
        return search_->dataset();
    }
    int k() const noexcept {
        return k_;
    }
    const std::vector<double>& k_distance() const noexcept {
        return kdist_;
    }
    const std::vector<double>& lrd() const noexcept {
        return lrd_;
    }

    const std::vector<double>& scores() const noexcept {
        return lof_;
    }

    std::vector<double> score(std::span<const double> num,
                              std::span<const int> cat,
                              std::size_t m) const;

private:
    std::shared_ptr<const NeighbourSearch> search_;
    void fit();

    int k_;
    std::vector<double> kdist_;
    std::vector<double> lrd_;
    std::vector<double> lof_;
};

}  // namespace mixdex
