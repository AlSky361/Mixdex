#include "mixdex/index/index.hpp"

#include <stdexcept>

namespace mixdex {

Index::Index(const Dataset& data, SearchMethod method, int leaf_size) :
    data_(&data),
    method_(method),
    leaf_size_(leaf_size),
    weights_(snapshot_weights()),
    category_sets_(data.category_sets()) {
    search_ = make_search(data, method, leaf_size);
}

std::vector<double> Index::snapshot_weights() const {
    std::vector<double> w(data_->n_cols());
    for (std::size_t k = 0; k < w.size(); ++k) w[k] = data_->weight(static_cast<int>(k));
    return w;
}

void Index::check_fresh() const {
    if (snapshot_weights() != weights_ || data_->category_sets() != category_sets_)
        throw std::logic_error(
            "Index: Dataset weights or set dictionaries changed after the index was built; rebuild the index");
}

}  // namespace mixdex
