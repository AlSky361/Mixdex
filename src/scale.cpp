#include "mixdex/scale.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mixdex/quantile.hpp"
#include "mixdex/utils/checked_arithmetic.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/numeric_validation.hpp"
#include "mixdex/utils/parallel.hpp"

using namespace mixdex::constants;

namespace mixdex {

ScaleParams fit_scale_params(std::span<const double> values, NumericType type) {
    if (values.empty()) {
        throw std::invalid_argument("scale_feature: пустой признак");
    }

    const QuantileMethod qm = method_for(type);
    std::vector<double> work(values.begin(), values.end());
    // Quantile order: min, q10, q25, median, q75, q90, max.
    const auto qs = quantiles_select(work, kScaleProbs, qm);
    const double median = qs[3];

    // All candidate widths use one relative threshold, not machine epsilon.
    // Rescaling preserves the test in exact arithmetic; shifting the data may change it.
    const double tol = kScaleRelEps * std::max(std::fabs(qs[0]), std::fabs(qs[6]));
    const struct {
        double width;
        ScaleLevel level;
    } ladder[] = {{qs[4] - qs[2], ScaleLevel::Iqr},
                  {qs[5] - qs[1], ScaleLevel::Idr},
                  {qs[6] - qs[0], ScaleLevel::Range}};
    for (const auto& step : ladder) {
        if (step.width > tol)
            return {median, step.width, step.level};
    }
    // Dataset rejects exact constants; nonconstant widths can still be numerically unusable.
    if (qs[0] == qs[6]) {
        throw std::invalid_argument(
            "fit_scale_params: признак константен (min == max); "
            "исключите его перед масштабированием");
    }
    throw std::invalid_argument(
        "fit_scale_params: численно вырожденный масштаб: ширины IQR, IDR и max-min "
        "не превышают относительный допуск kScaleRelEps * max(abs(min), abs(max)); "
        "это не означает точную константность признака");
}

std::vector<double> scale_feature(std::span<const double> values, NumericType type) {
    const ScaleParams p = fit_scale_params(values, type);

    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = (values[i] - p.median) / p.scale;
    }
    return result;
}

Scaler Scaler::fit(const Dataset& data) {
    const int n = data.n_rows();
    const int n_num = data.n_num();
    if (n == 0)
        throw std::invalid_argument("Scaler::fit: набор без строк");

    Scaler s;
    s.median_.resize(n_num);
    s.scale_.resize(n_num);
    s.level_.resize(n_num);

    parallel_for<Schedule::Static>(n_num, [&](int k) {
        auto span_col = data.col_num_span(k);
        std::vector<double> col(n);
        span_col.copy_to(col.data());
        const ScaleParams p =
            fit_scale_params(std::span<const double>(col.data(), col.size()), data.num_type(k));
        s.median_[k] = p.median;
        s.scale_[k] = p.scale;
        s.level_[k] = p.level;
    });

    return s;
}

std::vector<double> Scaler::transform_rows(std::span<const double> num, std::size_t m) const {
    const std::size_t d = static_cast<std::size_t>(n_num());
    if (num.size() != detail::checked_size_product(m, d))
        throw std::invalid_argument("Scaler::transform_rows: размер не равен m * n_num");
    detail::require_finite(num, "Scaler::transform_rows: NaN или inf во входных значениях");

    std::vector<double> out(num.size());
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < d; ++k)
            out[i * d + k] = (num[i * d + k] - median_[k]) / scale_[k];
    return out;
}

Dataset Scaler::transform(const Dataset& data) const {
    int n = data.n_rows();
    int n_num = data.n_num();
    int n_cat = data.n_cat();

    if (n_num != this->n_num())
        throw std::invalid_argument("Scaler::transform: число числовых признаков не совпадает");

    if (n == 0)
        return Dataset();

    std::vector<double> num_data(static_cast<std::size_t>(n) * n_num);
    std::vector<NumericType> num_types(n_num);

    parallel_for<Schedule::Static>(n_num, [&](int k) {
        auto span_col = data.col_num_span(k);
        num_types[k] = data.num_type(k);
        for (int i = 0; i < n; ++i) {
            num_data[static_cast<std::size_t>(i) * n_num + k] =
                (span_col[i] - median_[k]) / scale_[k];
        }
    });

    std::vector<int> cat_data(static_cast<std::size_t>(n) * n_cat);

    parallel_for<Schedule::Static>(n_cat, [&](int k) {
        auto span_col = data.col_cat_span(k);
        span_col.copy_to(&cat_data[k], n_cat);
    });

    std::vector<FeatureType> feat_types(static_cast<std::size_t>(data.n_cols()));
    std::vector<std::string> feat_names(static_cast<std::size_t>(data.n_cols()));

    parallel_for<Schedule::Static>(data.n_cols(), [&](int k) {
        // Dictionaries are attached after constructing the scaled data.
        feat_types[static_cast<std::size_t>(k)] = data.feat_type(k) == FeatureType::CategoricalSet
                                                      ? FeatureType::Categorical
                                                      : data.feat_type(k);
        feat_names[static_cast<std::size_t>(k)] = data.feat_name(k);
    });

    Dataset result(std::move(num_data),
                   std::move(cat_data),
                   n,
                   n_num,
                   n_cat,
                   std::move(feat_types),
                   std::move(num_types),
                   std::move(feat_names));

    if (data.has_weights()) {
        std::vector<double> w(data.n_cols());
        for (std::size_t k = 0; k < w.size(); ++k) w[k] = data.weight(static_cast<int>(k));
        result.set_weights(std::move(w));
    }
    for (const auto& [k, sets] : data.category_sets()) result.set_category_sets(k, sets);
    return result;
}

Dataset scale(const Dataset& data) {
    if (data.n_rows() == 0)
        return Dataset();
    return Scaler::fit(data).transform(data);
}

}  // namespace mixdex
