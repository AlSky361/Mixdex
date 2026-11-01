#pragma once
#include <memory>
#include <vector>

#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {

/// Borrow Dataset storage: keep it alive and rebuild after changing data or weights.
class Index {
public:
    explicit Index(const Dataset& data,
                   SearchMethod method = SearchMethod::BallTree,
                   int leaf_size = constants::kDefaultLeafSize);

    const NeighbourSearch& search() const {
        check_fresh();
        return *search_;
    }
    std::shared_ptr<const NeighbourSearch> shared() const {
        check_fresh();
        return search_;
    }

    const Dataset& dataset() const noexcept {
        return *data_;
    }
    SearchMethod method() const noexcept {
        return method_;
    }
    int leaf_size() const noexcept {
        return leaf_size_;
    }
    int n_rows() const noexcept {
        return search_->n_rows();
    }

    void check_fresh() const;

private:
    std::vector<double> snapshot_weights() const;

    const Dataset* data_;
    SearchMethod method_;
    int leaf_size_;
    std::vector<double> weights_;
    std::map<int, std::vector<std::vector<Code>>> category_sets_;
    std::shared_ptr<const NeighbourSearch> search_;
};

}  // namespace mixdex
