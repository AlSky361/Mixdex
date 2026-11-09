#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "mixdex/types.hpp"

namespace mixdex {

void NodeCatPresent::build(const Dataset& data, const std::vector<int>& idx, int low, int high) {
    if (low < 0 || high < 0 || static_cast<std::size_t>(high) > idx.size() || low > high) {
        throw std::invalid_argument("NodeCatPresent::build: invalid index range");
    }

    const int n_cat = static_cast<int>(data.n_cat());
    offsets_.assign(static_cast<std::size_t>(n_cat) + 1, 0);
    categories_.clear();

    std::size_t total_k = 0;
    for (int k = 0; k < n_cat; ++k) total_k += data.categories_size(k);
    categories_.reserve(total_k);

    std::vector<unsigned char> seen;
    for (int k = 0; k < n_cat; ++k) {
        auto col = data.col_cat_span(k);
        int K = static_cast<int>(data.categories_size(k));

        seen.assign(static_cast<std::size_t>(K), 0);
        for (int i = low; i < high; ++i) {
            seen[col[idx[i]]] = 1;
        }

        for (int c = 0; c < K; ++c) {
            if (seen[c])
                categories_.push_back(c);
        }
        offsets_[k + 1] = categories_.size();
    }
}

NodeCatPresent NodeCatPresent::merge(const NodeCatPresent& a, const NodeCatPresent& b) {
    NodeCatPresent result;
    const int n_cat = a.n_features();
    result.offsets_.assign(static_cast<std::size_t>(n_cat) + 1, 0);
    result.categories_.reserve(a.categories_.size() + b.categories_.size());

    for (int k = 0; k < n_cat; ++k) {
        CategorySet ca = a[k];
        CategorySet cb = b[k];
        std::set_union(
            ca.begin(), ca.end(), cb.begin(), cb.end(), std::back_inserter(result.categories_));
        result.offsets_[k + 1] = result.categories_.size();
    }
    return result;
}

}  // namespace mixdex
