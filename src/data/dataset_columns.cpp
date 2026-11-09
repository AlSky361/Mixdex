#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mixdex/types.hpp"
#include "mixdex/utils/checked_arithmetic.hpp"

namespace mixdex {

namespace {

template <class T>
bool column_is_constant(StridedSpan<T> column) {
    if (column.empty())
        return false;
    const T first = column[0];
    for (std::size_t i = 1; i < column.size(); ++i)
        if (column[i] != first)
            return false;
    return true;
}

template <class T>
std::vector<T> select_items(const std::vector<T>& source, const std::vector<std::size_t>& columns) {
    std::vector<T> result;
    result.reserve(columns.size());
    for (std::size_t k : columns) result.push_back(source[k]);
    return result;
}

template <class T>
std::vector<T> select_columns(const std::vector<T>& source,
                              std::size_t rows,
                              std::size_t width,
                              const std::vector<std::size_t>& columns) {
    std::vector<T> result(detail::checked_size_product(rows, columns.size()));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t k = 0; k < columns.size(); ++k)
            result[i * columns.size() + k] = source[i * width + columns[k]];
    return result;
}

std::map<int, std::vector<std::vector<Code>>> select_dictionaries(
    const std::map<int, std::vector<std::vector<Code>>>& source,
    const std::vector<std::size_t>& columns) {
    std::map<int, std::vector<std::vector<Code>>> result;
    for (std::size_t k = 0; k < columns.size(); ++k) {
        const auto it = source.find(static_cast<int>(columns[k]));
        if (it != source.end())
            result.emplace(static_cast<int>(k), it->second);
    }
    return result;
}

}  // namespace

std::vector<unsigned char> Dataset::find_constant_columns() const {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    std::vector<unsigned char> col_constant(n_num_f + n_cat_f, 0);
    if (n_ == 0)
        return col_constant;

    for (std::size_t k = 0; k < n_num_f; ++k)
        col_constant[k] = column_is_constant(col_num_span(static_cast<int>(k)));
    for (std::size_t k = 0; k < n_cat_f; ++k)
        col_constant[n_num_f + k] = column_is_constant(col_cat_span(static_cast<int>(k)));
    return col_constant;
}

// Project buffers and metadata through the same ordered column selection.
void Dataset::project_columns(std::span<const unsigned char> keep) {
    std::vector<std::size_t> numeric, categorical, all;
    for (std::size_t k = 0; k < keep.size(); ++k) {
        if (!keep[k])
            continue;
        all.push_back(k);
        if (k < n_num_)
            numeric.push_back(k);
        else
            categorical.push_back(k - n_num_);
    }
    auto num = select_columns(num_data_, n_, n_num_, numeric);
    auto cat = select_columns(cat_data_, n_, n_cat_, categorical);
    auto num_types = select_items(num_types_, numeric);
    auto feat_types = select_items(feat_types_, all);
    auto names = select_items(feat_names_, all);
    auto counts = categories_count_.empty() ? std::vector<int>{}
                                            : select_items(categories_count_, categorical);
    auto weights = weights_.empty() ? std::vector<double>{} : select_items(weights_, all);
    auto dictionaries = select_dictionaries(category_sets_, categorical);

    num_data_ = std::move(num);
    cat_data_ = std::move(cat);
    num_types_ = std::move(num_types);
    feat_types_ = std::move(feat_types);
    feat_names_ = std::move(names);
    categories_count_ = std::move(counts);
    weights_ = std::move(weights);
    category_sets_ = std::move(dictionaries);
    n_num_ = numeric.size();
    n_cat_ = categorical.size();
}

void Dataset::check_constant_columns() const {
    const auto constant = find_constant_columns();
    for (std::size_t k = 0; k < constant.size(); ++k)
        if (constant[k])
            throw std::invalid_argument("Dataset: constant feature '" + feat_names_[k] +
                                        "' (index " + std::to_string(k) +
                                        "); Strict policy rejects constant features");
}

void Dataset::drop_constant_columns(DatasetPolicy policy) {
    if (policy == DatasetPolicy::Strict) {
        check_constant_columns();
        return;
    }
    const auto constant = find_constant_columns();
    std::vector<unsigned char> keep(constant.size(), 1);
    bool dropped = false;
    for (std::size_t k = 0; k < constant.size(); ++k) {
        if (!constant[k])
            continue;
        dropped = true;
        keep[k] = 0;
        report_.columns_dropped_constant_idx.push_back(static_cast<int>(k));
        report_.columns_dropped_constant_name.push_back(feat_names_[k]);
    }
    if (dropped)
        project_columns(keep);
}

}  // namespace mixdex
