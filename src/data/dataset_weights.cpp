#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "mixdex/types.hpp"

namespace mixdex {
namespace {

double validated_weight_sum(const std::vector<double>& weights) {
    double sum = 0.0;
    for (double w : weights) {
        if (!std::isfinite(w) || !(w >= 0.0))
            throw std::invalid_argument(
                "Dataset::set_weights: each weight must be finite and >= 0");
        sum += w;
    }
    if (!std::isfinite(sum) || !(sum > 0.0))
        throw std::invalid_argument("Dataset::set_weights: W = sum(w_k) must be finite and > 0");
    if (!std::isfinite(1.0 / sum))
        throw std::invalid_argument("weight total has no finite reciprocal; rescale weights");
    return sum;
}

}  // namespace

void Dataset::set_weights(std::vector<double> weights) {
    if (weights.size() != n_num_ + n_cat_)
        throw std::invalid_argument("Dataset::set_weights: weights size must equal n_cols");
    const double sum = validated_weight_sum(weights);
    const bool any_zero = std::any_of(weights.begin(), weights.end(), [](double w) {
        return w == 0.0;
    });
    if (any_zero) {
        // Commit projection only after cleanup succeeds.
        Dataset candidate = *this;
        std::vector<unsigned char> keep(weights.size(), 1);
        for (std::size_t k = 0; k < weights.size(); ++k) {
            if (weights[k] != 0.0)
                continue;
            keep[k] = 0;
            candidate.report_.columns_dropped_zero_weight_idx.push_back(static_cast<int>(k));
            candidate.report_.columns_dropped_zero_weight_name.push_back(feat_names_[k]);
        }
        candidate.weights_ = std::move(weights);
        candidate.project_columns(keep);
        // Removing coordinates can make distinct rows identical.
        std::vector<unsigned char> duplicates;
        candidate.check_duplicates(DatasetPolicy::Clean, {}, &duplicates);
        candidate.drop_rows({}, duplicates);
        *this = std::move(candidate);
    } else {
        weights_ = std::move(weights);
    }
    inv_W_ = 1.0 / sum;
}

}  // namespace mixdex
