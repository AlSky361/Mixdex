#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "mixdex/types.hpp"

namespace mixdex {

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
void Dataset::validate() const {
    // Repeat Strict checks without modifying data.
    check_limits();
    check_structure();
    scan_values(nullptr, DatasetPolicy::Strict, nullptr);
    check_duplicates(DatasetPolicy::Strict, {}, nullptr);

    check_constant_columns();

    if (categories_count_.size() != n_cat_) {
        throw std::invalid_argument("Dataset: несоответствие n_cat_");
    }
}

}  // namespace mixdex
