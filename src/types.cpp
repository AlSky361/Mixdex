#include "mixdex/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/checked_arithmetic.hpp"

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

void NodeCatPresent::build(const Dataset& data, const std::vector<int>& idx, int low, int high) {
    if (low < 0 || high < 0 || static_cast<std::size_t>(high) > idx.size() || low > high) {
        throw std::invalid_argument("NodeCatPresent::build: некорректный диапазон индексов");
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

void default_preprocess_warning_handler(const std::string& msg) {
    std::cerr << "mixdex: " << msg << '\n';
}

Dataset::Dataset(const std::vector<NumFeature>& num_features,
                 const std::vector<CategoricalFeature>& cat_features,
                 const std::vector<std::string>& feat_names,
                 DatasetPolicy policy,
                 const PreprocessWarningHandler& on_warning) {
    n_num_ = num_features.size();
    n_cat_ = cat_features.size();

    if (n_num_ > 0) {
        n_ = num_features[0].size();
    } else if (n_cat_ > 0) {
        n_ = cat_features[0].size();
    } else {
        n_ = 0;
    }

    check_limits();

    if (feat_names.size() != n_num_ + n_cat_) {
        throw std::invalid_argument("Dataset: несоответствующее количество признаков");
    }

    if (n_num_ > 0 && n_cat_ > 0 && num_features[0].size() != cat_features[0].size()) {
        throw std::invalid_argument(
            "Dataset: признаки числового и категориального типов имеют разное количество строк");
    }

    if (n_ == 0 && (n_num_ > 0 || n_cat_ > 0)) {
        throw std::invalid_argument("Dataset: 0 строк, но есть признаки");
    }

    feat_types_.reserve(n_num_ + n_cat_);
    num_types_.reserve(n_num_);
    feat_names_.reserve(n_num_ + n_cat_);

    num_data_.assign(detail::checked_size_product(n_, n_num_), 0.0);
    for (std::size_t k = 0; k < n_num_; ++k) {
        const auto& f = num_features[k];

        if (f.size() != n_) {
            throw std::invalid_argument("Dataset: несоответствующее количество числовых значений");
        }

        feat_types_.push_back(FeatureType::Numeric);
        num_types_.push_back(f.type);
        feat_names_.push_back(feat_names[k]);

        for (std::size_t i = 0; i < n_; ++i) {
            num_data_[i * n_num_ + k] = f.array[i];
        }
    }

    cat_data_.assign(detail::checked_size_product(n_, n_cat_), 0);
    for (std::size_t k = 0; k < n_cat_; ++k) {
        const auto& f = cat_features[k];

        if (f.size() != n_) {
            throw std::invalid_argument(
                "Dataset: несоответствующее количество категориальных значений");
        }

        feat_types_.push_back(FeatureType::Categorical);
        feat_names_.push_back(feat_names[n_num_ + k]);

        for (std::size_t i = 0; i < n_; ++i) {
            cat_data_[i * n_cat_ + k] = f.encoded[i];
        }
    }

    finish_construction(policy, on_warning);
}

Dataset::Dataset(std::vector<double> num_data,
                 std::vector<Code> cat_data,
                 std::size_t n,
                 std::size_t n_num,
                 std::size_t n_cat,
                 std::vector<FeatureType> feat_types,
                 std::vector<NumericType> num_types,
                 std::vector<std::string> feat_names,
                 DatasetPolicy policy,
                 const PreprocessWarningHandler& on_warning) :
    num_data_(std::move(num_data)),
    cat_data_(std::move(cat_data)),
    feat_types_(std::move(feat_types)),
    num_types_(std::move(num_types)),
    feat_names_(std::move(feat_names)),
    n_(n),
    n_num_(n_num),
    n_cat_(n_cat) {
    finish_construction(policy, on_warning);
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

void Dataset::set_weights(std::vector<double> weights) {
    if (weights.size() != n_num_ + n_cat_) {
        throw std::invalid_argument("Dataset::set_weights: несоответствие размера весов n_cols");
    }

    double sum = 0.0;
    for (double w : weights) {
        if (!std::isfinite(w) || !(w >= 0.0)) {
            throw std::invalid_argument(
                "Dataset::set_weights: вес w_k должен быть конечным и >= 0");
        }
        sum += w;
    }

    if (!std::isfinite(sum) || !(sum > 0.0)) {
        throw std::invalid_argument(
            "Dataset::set_weights: W = sum(w_k) должно быть конечным и > 0");
    }
    if (!std::isfinite(1.0 / sum))
        throw std::invalid_argument("weight total has no finite reciprocal; rescale weights");
    const std::size_t old_num = n_num_, old_cat = n_cat_;
    const bool any_zero = std::any_of(weights.begin(), weights.end(), [](double w) {
        return w == 0.0;
    });
    if (any_zero) {
        // Stage changes in a copy to preserve the original Dataset on failure.
        Dataset candidate = *this;
        std::vector<double> new_num;
        std::vector<Code> new_cat;
        std::vector<NumericType> new_num_types;
        std::vector<FeatureType> new_feat_types;
        std::vector<std::string> new_names;
        std::vector<Code> new_counts;
        std::vector<double> new_weights;
        for (std::size_t k = 0; k < old_num + old_cat; ++k) {
            if (weights[k] == 0.0) {
                candidate.report_.columns_dropped_zero_weight_idx.push_back(static_cast<int>(k));
                candidate.report_.columns_dropped_zero_weight_name.push_back(feat_names_[k]);
                continue;
            }
            new_feat_types.push_back(feat_types_[k]);
            new_names.push_back(feat_names_[k]);
            new_weights.push_back(weights[k]);
            if (k < old_num) {
                new_num_types.push_back(num_types_[k]);
                for (std::size_t i = 0; i < n_; ++i) new_num.push_back(num_data_[i * old_num + k]);
            } else {
                new_counts.push_back(categories_count_[k - old_num]);
                for (std::size_t i = 0; i < n_; ++i)
                    new_cat.push_back(cat_data_[i * old_cat + (k - old_num)]);
            }
        }
        // Convert collected columns back to separate row-major buffers.
        const std::size_t kept_num = new_num_types.size(), kept_cat = new_counts.size();
        std::vector<double> num_rows(n_ * kept_num);
        std::vector<Code> cat_rows(n_ * kept_cat);
        for (std::size_t k = 0; k < kept_num; ++k)
            for (std::size_t i = 0; i < n_; ++i) num_rows[i * kept_num + k] = new_num[k * n_ + i];
        for (std::size_t k = 0; k < kept_cat; ++k)
            for (std::size_t i = 0; i < n_; ++i) cat_rows[i * kept_cat + k] = new_cat[k * n_ + i];
        candidate.num_data_ = std::move(num_rows);
        candidate.cat_data_ = std::move(cat_rows);
        candidate.num_types_ = std::move(new_num_types);
        candidate.feat_types_ = std::move(new_feat_types);
        candidate.feat_names_ = std::move(new_names);
        candidate.categories_count_ = std::move(new_counts);
        candidate.category_sets_.clear();
        int next_cat = 0;
        for (int k = 0; k < static_cast<int>(old_cat); ++k) {
            if (weights[old_num + k] == 0.0)
                continue;
            const auto it = category_sets_.find(k);
            if (it != category_sets_.end())
                candidate.category_sets_[next_cat] = it->second;
            ++next_cat;
        }
        candidate.n_num_ = kept_num;
        candidate.n_cat_ = kept_cat;
        candidate.weights_ = std::move(new_weights);
        // Projection may create new duplicate rows.
        std::vector<unsigned char> duplicates;
        candidate.check_duplicates(DatasetPolicy::Clean, {}, &duplicates);
        candidate.drop_rows({}, duplicates);
        *this = std::move(candidate);
    } else {
        weights_ = std::move(weights);
    }
    inv_W_ = 1.0 / sum;
}

FeatureType Dataset::feat_type(int k) const noexcept {
    return feat_types_[k];
}

NumericType Dataset::num_type(int k) const noexcept {
    return num_types_[k];
}

const std::string& Dataset::feat_name(int k) const noexcept {
    return feat_names_[k];
}

void Dataset::check_limits() const {
    constexpr std::size_t kMax = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (n_ > kMax)
        throw std::length_error("Dataset: число строк превышает INT_MAX");
    if (n_num_ > kMax || n_cat_ > kMax || n_num_ + n_cat_ > kMax)
        throw std::length_error("Dataset: число признаков превышает INT_MAX");
}

void Dataset::check_structure() const {
    if (feat_types_.size() != n_num_ + n_cat_ || feat_names_.size() != feat_types_.size()) {
        throw std::invalid_argument("Dataset: несоответствие размера feat_types");
    }

    if (num_data_.size() != n_ * n_num_) {
        throw std::invalid_argument("Dataset: несоответствие размера num_data");
    }

    if (cat_data_.size() != n_ * n_cat_) {
        throw std::invalid_argument("Dataset: несоответствие размера cat_data");
    }

    if (num_types_.size() != n_num_) {
        throw std::invalid_argument("Dataset: несоответствие размера num_types");
    }

    for (std::size_t k = 0; k < feat_types_.size(); ++k) {
        const auto type = feat_types_[k];
        if (k < n_num_) {
            if (type != FeatureType::Numeric)
                throw std::invalid_argument("numeric features must precede categorical features");
        } else if (type != FeatureType::Categorical && type != FeatureType::CategoricalSet) {
            throw std::invalid_argument("invalid categorical feature type");
        } else if (type == FeatureType::CategoricalSet &&
                   !category_sets_.contains(static_cast<int>(k - n_num_))) {
            throw std::invalid_argument("attach set dictionaries with set_category_sets");
        }
    }
    for (const auto type : num_types_)
        if (type != NumericType::Continuous && type != NumericType::Discrete)
            throw std::invalid_argument("invalid numeric feature type");
}

void Dataset::scan_values(std::vector<Code>* cat_max,
                          DatasetPolicy policy,
                          std::vector<unsigned char>* rows_missing) const {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    if (cat_max)
        cat_max->assign(n_cat_f, -1);
    if (rows_missing)
        rows_missing->assign(n_, 0);

    std::size_t bad_num = 0, bad_cat = 0;
    std::size_t first_num_row = 0, first_num_col = 0, first_cat_row = 0, first_cat_col = 0;

    for (std::size_t i = 0; i < n_; ++i) {
        bool row_bad = false;
        const double* x = num_data_.data() + i * n_num_f;
        for (std::size_t k = 0; k < n_num_f; ++k) {
            if (!std::isfinite(x[k])) {
                if (bad_num++ == 0) {
                    first_num_row = i;
                    first_num_col = k;
                }
                row_bad = true;
            }
        }
        const Code* c = cat_data_.data() + i * n_cat_f;
        for (std::size_t k = 0; k < n_cat_f; ++k) {
            if (c[k] == std::numeric_limits<Code>::max())
                throw std::invalid_argument(
                    "Dataset: category code leaves no room for max_code + 1");
            if (c[k] < 0) {
                if (bad_cat++ == 0) {
                    first_cat_row = i;
                    first_cat_col = k;
                }
                row_bad = true;
            } else if (cat_max && c[k] > (*cat_max)[k]) {
                // Clean recomputes final maxima after removing rows and columns.
                (*cat_max)[k] = c[k];
            }
        }
        if (row_bad && rows_missing)
            (*rows_missing)[i] = 1;
    }

    if (bad_num == 0 && bad_cat == 0)
        return;

    if (policy == DatasetPolicy::Strict) {
        std::string msg = "Dataset:";
        if (bad_num > 0)
            msg += " NaN или inf в числовых значениях: " + std::to_string(bad_num) +
                   " шт., первое — строка " + std::to_string(first_num_row) +
                   ", числовой признак " + std::to_string(first_num_col) + ";";
        if (bad_cat > 0)
            msg += " отрицательные коды категорий: " + std::to_string(bad_cat) +
                   " шт., первый — строка " + std::to_string(first_cat_row) +
                   ", категориальный признак " + std::to_string(first_cat_col) + ";";
        throw std::invalid_argument(msg);
    }
}

namespace {

// Compare finite values exactly; -0.0 and +0.0 are equal.
bool row_equal(const double* a_num,
               const double* b_num,
               std::size_t n_num,
               const Code* a_cat,
               const Code* b_cat,
               std::size_t n_cat) {
    for (std::size_t k = 0; k < n_num; ++k) {
        if (a_num[k] != b_num[k])
            return false;
    }
    for (std::size_t k = 0; k < n_cat; ++k) {
        if (a_cat[k] != b_cat[k])
            return false;
    }
    return true;
}

std::uint64_t mix64(std::uint64_t z) {  // splitmix64 finalizer.
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

std::uint64_t row_hash(const double* num, std::size_t n_num, const Code* cat, std::size_t n_cat) {
    std::uint64_t h = 0;
    for (std::size_t k = 0; k < n_num; ++k) {
        double x = num[k];
        if (x == 0.0)
            x = 0.0;  // Equal signed zeros must hash identically.
        std::uint64_t bits;
        std::memcpy(&bits, &x, sizeof bits);
        h = mix64(h ^ bits);
    }
    for (std::size_t k = 0; k < n_cat; ++k)
        h = mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(cat[k])));
    return h;
}

}  // namespace

void Dataset::check_duplicates(DatasetPolicy policy,
                               const std::vector<unsigned char>& rows_dropped_already,
                               std::vector<unsigned char>* rows_duplicate) const {
    if (rows_duplicate)
        rows_duplicate->assign(n_, 0);
    if (n_ < 2)
        return;

    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    const bool skip_dropped = !rows_dropped_already.empty();

    // Hashes group candidates; coordinate comparisons confirm equality.
    // Skip missing rows already marked for removal.
    std::vector<std::pair<std::uint64_t, std::size_t>> keys;
    keys.reserve(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        if (skip_dropped && rows_dropped_already[i])
            continue;
        keys.push_back(
            {row_hash(
                 num_data_.data() + i * n_num_f, n_num_f, cat_data_.data() + i * n_cat_f, n_cat_f),
             i});
    }
    std::sort(keys.begin(), keys.end());

    for (std::size_t a = 0; a < keys.size();) {
        std::size_t b = a + 1;
        while (b < keys.size() && keys[b].first == keys[a].first) ++b;
        for (std::size_t p = a; p < b; ++p) {
            const std::size_t i = keys[p].second;
            if (rows_duplicate && (*rows_duplicate)[i])
                continue;
            for (std::size_t q = p + 1; q < b; ++q) {
                const std::size_t j = keys[q].second;
                if (row_equal(num_data_.data() + i * n_num_f,
                              num_data_.data() + j * n_num_f,
                              n_num_f,
                              cat_data_.data() + i * n_cat_f,
                              cat_data_.data() + j * n_cat_f,
                              n_cat_f)) {
                    const std::size_t lo = std::min(i, j), hi = std::max(i, j);
                    if (policy == DatasetPolicy::Strict) {
                        throw std::invalid_argument(
                            "Dataset: объекты-дубликаты (все координаты равны): строки " +
                            std::to_string(lo) + " и " + std::to_string(hi) +
                            "; один из них должен быть исключён до кластеризации (см. §3.1)");
                    }
                    // Keep the first original row and mark later duplicates.
                    if (rows_duplicate)
                        (*rows_duplicate)[hi] = 1;
                }
            }
        }
        a = b;
    }
}

void Dataset::drop_rows(const std::vector<unsigned char>& rows_missing,
                        const std::vector<unsigned char>& rows_duplicate) {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();

    const bool has_missing = !rows_missing.empty();
    const bool has_duplicate = !rows_duplicate.empty();
    if (!has_missing && !has_duplicate)
        return;

    std::size_t kept = 0;
    for (std::size_t i = 0; i < n_; ++i) {
        const bool drop = (has_missing && rows_missing[i]) || (has_duplicate && rows_duplicate[i]);
        if (drop) {
            if (has_missing && rows_missing[i])
                ++report_.rows_dropped_missing;
            else
                ++report_.rows_dropped_duplicate;
            continue;
        }
        if (kept != i) {
            std::copy(num_data_.begin() + static_cast<std::ptrdiff_t>(i * n_num_f),
                      num_data_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_num_f),
                      num_data_.begin() + static_cast<std::ptrdiff_t>(kept * n_num_f));
            std::copy(cat_data_.begin() + static_cast<std::ptrdiff_t>(i * n_cat_f),
                      cat_data_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_cat_f),
                      cat_data_.begin() + static_cast<std::ptrdiff_t>(kept * n_cat_f));
        }
        ++kept;
    }
    num_data_.resize(kept * n_num_f);
    cat_data_.resize(kept * n_cat_f);
    n_ = kept;
}

std::vector<unsigned char> Dataset::find_constant_columns() const {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    const std::size_t n_cols_f = n_num_f + n_cat_f;

    std::vector<unsigned char> col_constant(n_cols_f, 0);
    if (n_ == 0)
        return col_constant;  // No constant columns without rows.

    for (std::size_t k = 0; k < n_num_f; ++k) {
        const double v0 = num_data_[k];
        bool constant = true;
        for (std::size_t i = 1; i < n_ && constant; ++i) {
            if (num_data_[i * n_num_f + k] != v0)
                constant = false;
        }
        col_constant[k] = constant ? 1 : 0;
    }
    for (std::size_t k = 0; k < n_cat_f; ++k) {
        const Code v0 = cat_data_[k];
        bool constant = true;
        for (std::size_t i = 1; i < n_ && constant; ++i) {
            if (cat_data_[i * n_cat_f + k] != v0)
                constant = false;
        }
        col_constant[n_num_f + k] = constant ? 1 : 0;
    }
    return col_constant;
}

void Dataset::drop_constant_columns(DatasetPolicy policy) {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    const std::size_t n_cols_f = n_num_f + n_cat_f;

    const std::vector<unsigned char> col_constant = find_constant_columns();
    const bool any_constant =
        std::any_of(col_constant.begin(), col_constant.end(), [](unsigned char c) {
            return c != 0;
        });
    if (!any_constant)
        return;

    if (policy == DatasetPolicy::Strict) {
        for (std::size_t k = 0; k < n_cols_f; ++k) {
            if (col_constant[k]) {
                throw std::invalid_argument("Dataset: константный признак '" + feat_names_[k] +
                                            "' (индекс " + std::to_string(k) +
                                            "); такой признак должен быть исключён до "
                                            "построения индексной структуры (см. §3.1)");
            }
        }
        return;
    }

    // Compact data blocks separately and metadata in global feature order.
    std::vector<double> new_num;
    std::vector<NumericType> new_num_types;
    new_num.reserve(n_ * n_num_f);
    for (std::size_t k = 0; k < n_num_f; ++k) {
        if (col_constant[k]) {
            report_.columns_dropped_constant_idx.push_back(static_cast<int>(k));
            report_.columns_dropped_constant_name.push_back(feat_names_[k]);
            continue;
        }
        new_num_types.push_back(num_types_[k]);
    }
    const std::size_t new_n_num = new_num_types.size();
    new_num.resize(n_ * new_n_num);
    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t kcol = 0;
        for (std::size_t k = 0; k < n_num_f; ++k) {
            if (col_constant[k])
                continue;
            new_num[i * new_n_num + kcol] = num_data_[i * n_num_f + k];
            ++kcol;
        }
    }

    std::size_t n_cat_dropped = 0;
    for (std::size_t k = 0; k < n_cat_f; ++k) {
        if (col_constant[n_num_f + k]) {
            report_.columns_dropped_constant_idx.push_back(static_cast<int>(n_num_f + k));
            report_.columns_dropped_constant_name.push_back(feat_names_[n_num_f + k]);
            ++n_cat_dropped;
        }
    }
    const std::size_t new_n_cat = n_cat_f - n_cat_dropped;
    std::vector<Code> new_cat(n_ * new_n_cat);
    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t kcol = 0;
        for (std::size_t k = 0; k < n_cat_f; ++k) {
            if (col_constant[n_num_f + k])
                continue;
            new_cat[i * new_n_cat + kcol] = cat_data_[i * n_cat_f + k];
            ++kcol;
        }
    }

    std::vector<FeatureType> new_feat_types;
    std::vector<std::string> new_feat_names;
    new_feat_types.reserve(new_n_num + new_n_cat);
    new_feat_names.reserve(new_n_num + new_n_cat);
    for (std::size_t k = 0; k < n_cols_f; ++k) {
        if (col_constant[k])
            continue;
        new_feat_types.push_back(feat_types_[k]);
        new_feat_names.push_back(feat_names_[k]);
    }

    num_data_ = std::move(new_num);
    cat_data_ = std::move(new_cat);
    num_types_ = std::move(new_num_types);
    feat_types_ = std::move(new_feat_types);
    feat_names_ = std::move(new_feat_names);
    n_num_ = new_n_num;
    n_cat_ = new_n_cat;
}

void Dataset::validate_all(std::vector<Code>* cat_max,
                           DatasetPolicy policy,
                           const PreprocessWarningHandler& on_warning) {
    check_limits();
    check_structure();

    if (policy == DatasetPolicy::Strict) {
        scan_values(cat_max, policy, nullptr);
        check_duplicates(policy, {}, nullptr);
        drop_constant_columns(policy);  // Strict rejects any constant columns.
        return;
    }

    // Remove rows before checking remaining columns for constancy.
    std::vector<unsigned char> rows_missing, rows_duplicate;
    scan_values(nullptr, policy, &rows_missing);
    check_duplicates(policy, rows_missing, &rows_duplicate);
    drop_rows(rows_missing, rows_duplicate);
    drop_constant_columns(policy);

    if (!report_.empty() && on_warning) {
        if (report_.rows_dropped_missing > 0)
            on_warning("исключено строк с пропуском: " +
                       std::to_string(report_.rows_dropped_missing));
        if (report_.rows_dropped_duplicate > 0)
            on_warning("исключено строк-дубликатов: " +
                       std::to_string(report_.rows_dropped_duplicate));
        for (std::size_t j = 0; j < report_.columns_dropped_constant_idx.size(); ++j) {
            on_warning("исключён константный признак '" + report_.columns_dropped_constant_name[j] +
                       "' (исходный индекс " +
                       std::to_string(report_.columns_dropped_constant_idx[j]) + ")");
        }
    }

    // Removed rows/columns invalidate previous category maxima.
    if (cat_max) {
        scan_values(cat_max, DatasetPolicy::Clean, nullptr);
    }
}

void Dataset::finish_construction(DatasetPolicy policy,
                                  const PreprocessWarningHandler& on_warning) {
    std::vector<Code> mx;
    validate_all(&mx, policy, on_warning);
    categories_count_.resize(mx.size());
    for (std::size_t k = 0; k < mx.size(); ++k) categories_count_[k] = mx[k] + 1;
}

void Dataset::validate() const {
    // Repeat Strict checks without modifying data.
    check_limits();
    check_structure();
    scan_values(nullptr, DatasetPolicy::Strict, nullptr);
    check_duplicates(DatasetPolicy::Strict, {}, nullptr);

    const std::vector<unsigned char> col_constant = find_constant_columns();
    for (std::size_t k = 0; k < col_constant.size(); ++k) {
        if (col_constant[k]) {
            throw std::invalid_argument("Dataset: константный признак '" + feat_names_[k] +
                                        "' (индекс " + std::to_string(k) +
                                        "); такой признак должен быть исключён до "
                                        "построения индексной структуры (см. §3.1)");
        }
    }

    if (categories_count_.size() != n_cat_) {
        throw std::invalid_argument("Dataset: несоответствие n_cat_");
    }
}

}  // namespace mixdex
