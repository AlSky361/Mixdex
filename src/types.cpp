#include "mixdex/types.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

#include "mixdex/utils/checked_arithmetic.hpp"

namespace mixdex {

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
FeatureType Dataset::feat_type(int k) const noexcept {
    return feat_types_[k];
}

NumericType Dataset::num_type(int k) const noexcept {
    return num_types_[k];
}

const std::string& Dataset::feat_name(int k) const noexcept {
    return feat_names_[k];
}
void Dataset::finish_construction(DatasetPolicy policy,
                                  const PreprocessWarningHandler& on_warning) {
    std::vector<Code> mx;
    validate_all(&mx, policy, on_warning);
    categories_count_.resize(mx.size());
    for (std::size_t k = 0; k < mx.size(); ++k) categories_count_[k] = mx[k] + 1;
}

}  // namespace mixdex
