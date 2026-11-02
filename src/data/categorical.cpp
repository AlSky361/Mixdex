#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

#include "mixdex/types.hpp"

namespace mixdex {

bool CategorySet::intersects(const CategorySet& other) const noexcept {
    auto it_a = categories.begin();
    auto it_b = other.categories.begin();
    while (it_a != categories.end() && it_b != other.categories.end()) {
        if (*it_a == *it_b)
            return true;
        if (*it_a < *it_b)
            ++it_a;
        else
            ++it_b;
    }
    return false;
}
void Dataset::set_category_sets(int k, std::vector<std::vector<Code>> sets) {
    if (k < 0 || k >= n_cat() || sets.size() < categories_size(k))
        throw std::invalid_argument("category set dictionary does not cover the column");
    std::set<std::vector<Code>> unique;
    for (auto& values : sets) {
        if (std::any_of(values.begin(), values.end(), [](Code c) {
                return c < 0;
            }))
            throw std::invalid_argument("set members must be nonnegative codes");
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (!unique.insert(values).second)
            throw std::invalid_argument("dictionary sets must be distinct");
    }
    Dataset candidate = *this;
    candidate.category_sets_[k] = std::move(sets);
    candidate.feat_types_[n_num_ + static_cast<std::size_t>(k)] = FeatureType::CategoricalSet;
    *this = std::move(candidate);
}

namespace {
double jaccard(std::span<const Code> x, std::span<const Code> y) noexcept {
    std::size_t i = 0, j = 0, common = 0;
    while (i < x.size() && j < y.size()) {
        if (x[i] == y[j]) {
            ++common;
            ++i;
            ++j;
        } else if (x[i] < y[j])
            ++i;
        else
            ++j;
    }
    const std::size_t total = x.size() + y.size() - common;
    return total == 0 ? 0.0 : static_cast<double>(total - common) / static_cast<double>(total);
}
}  // namespace

double Dataset::categorical_distance(int k, Code a, Code b) const noexcept {
    if (a == b)
        return 0.0;
    const auto it = category_sets_.find(k);
    if (it == category_sets_.end())
        return 1.0;
    const auto& dict = it->second;
    // Unknown encoded values are isolated; use actual sets for partial overlap.
    if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= dict.size() ||
        static_cast<std::size_t>(b) >= dict.size())
        return 1.0;
    return jaccard(dict[a], dict[b]);
}

double Dataset::categorical_distance(int k,
                                     std::span<const Code> query,
                                     Code stored) const noexcept {
    const auto it = category_sets_.find(k);
    if (it == category_sets_.end() || stored < 0 ||
        static_cast<std::size_t>(stored) >= it->second.size())
        return 1.0;
    return jaccard(query, it->second[stored]);
}

}  // namespace mixdex
