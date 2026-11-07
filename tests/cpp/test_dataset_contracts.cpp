#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mixdex/types.hpp"

namespace {
using namespace mixdex;

void check(bool value) {
    if (!value)
        throw std::runtime_error("Dataset contract regression");
}

template <class F>
void rejects(F operation, const std::string& message = {}) {
    try {
        operation();
    } catch (const std::invalid_argument& error) {
        check(message.empty() || error.what() == message);
        return;
    }
    throw std::runtime_error("expected invalid_argument");
}

void check_empty_schema(const Dataset& data, int numeric, int categorical) {
    check(data.empty() && data.n_rows() == 0);
    check(data.n_num() == numeric && data.n_cat() == categorical);
    check(data.n_cols() == numeric + categorical);
    check(data.num_data().empty() && data.cat_data().empty());
    check(data.preprocess_report().empty());
    for (int k = 0; k < numeric; ++k) {
        check(data.feat_type(k) == FeatureType::Numeric);
        check(data.num_type(k) == NumericType::Discrete);
        check(data.feat_name(k) == "n" + std::to_string(k));
        auto col = data.col_num_span(k);
        check(col.empty() && col.begin() == col.end());
        check(col.to_vector().empty());
        col.copy_to(std::span<double>{});
    }
    for (int k = 0; k < categorical; ++k) {
        check(data.feat_type(numeric + k) == FeatureType::Categorical);
        check(data.feat_name(numeric + k) == "c" + std::to_string(k));
        check(data.categories_size(k) == 0);
        auto col = data.col_cat_span(k);
        check(col.empty() && col.begin() == col.end());
        check(col.to_vector().empty());
        col.copy_to(std::span<int>{});
    }
    data.validate();
}

void check_empty_constructors() {
    for (auto policy : {DatasetPolicy::Strict, DatasetPolicy::Clean}) {
        for (int numeric : {0, 2}) {
            for (int categorical : {0, 2}) {
                std::vector<NumFeature> nums(numeric);
                for (auto& feature : nums) feature.type = NumericType::Discrete;
                std::vector<CategoricalFeature> cats(categorical);
                std::vector<FeatureType> types(numeric, FeatureType::Numeric);
                types.insert(types.end(), categorical, FeatureType::Categorical);
                std::vector<std::string> names;
                for (int k = 0; k < numeric; ++k) names.push_back("n" + std::to_string(k));
                for (int k = 0; k < categorical; ++k) names.push_back("c" + std::to_string(k));
                int warnings = 0;
                auto on_warning = [&](const std::string&) {
                    ++warnings;
                };
                Dataset columns(nums, cats, names, policy, on_warning);
                Dataset flat({},
                             {},
                             0,
                             numeric,
                             categorical,
                             types,
                             std::vector<NumericType>(numeric, NumericType::Discrete),
                             names,
                             policy,
                             on_warning);
                check_empty_schema(columns, numeric, categorical);
                check_empty_schema(flat, numeric, categorical);
                check(warnings == 0);
            }
        }
    }
}

void check_invalid_policies() {
    for (int value : {-1, 2, 123}) {
        auto policy = static_cast<DatasetPolicy>(value);
        int warnings = 0;
        auto on_warning = [&](const std::string&) {
            ++warnings;
        };
        // Invalid policies must not silently remove duplicates.
        rejects(
            [&] {
                Dataset({1, 1, 2},
                        {},
                        3,
                        1,
                        0,
                        {FeatureType::Numeric},
                        {NumericType::Continuous},
                        {"x"},
                        policy,
                        on_warning);
            },
            "invalid DatasetPolicy");
        rejects(
            [&] {
                Dataset(std::vector<NumFeature>{{{1, 1, 2}, NumericType::Continuous}},
                        {},
                        {"x"},
                        policy,
                        on_warning);
            },
            "invalid DatasetPolicy");
        rejects(
            [&] {
                Dataset({}, {}, {}, policy, on_warning);
            },
            "invalid DatasetPolicy");
        rejects(
            [&] {
                Dataset({}, {}, 0, 0, 0, {}, {}, {}, policy, on_warning);
            },
            "invalid DatasetPolicy");
        check(warnings == 0);
    }
}

void check_malformed_empty_inputs() {
    rejects([] {
        Dataset({}, {}, 0, 1, 0, {FeatureType::Numeric}, {NumericType::Continuous}, {});
    });
    rejects([] {
        Dataset({}, {}, 0, 1, 0, {FeatureType::Categorical}, {NumericType::Continuous}, {"x"});
    });
    rejects([] {
        Dataset({1}, {}, 0, 1, 0, {FeatureType::Numeric}, {NumericType::Continuous}, {"x"});
    });
    rejects([] {
        Dataset({}, {}, 0, 1, 0, {FeatureType::Numeric}, {static_cast<NumericType>(123)}, {"x"});
    });
    rejects([] {
        Dataset(
            std::vector<NumFeature>{{{}, NumericType::Continuous}, {{1}, NumericType::Continuous}},
            {},
            {"x", "y"});
    });
    rejects([] {
        Dataset({}, std::vector<CategoricalFeature>{{{}}}, {});
    });
}

void check_nonempty_policies() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<NumFeature> nums{{{1, 1, nan, 3, 4}, NumericType::Continuous},
                                 {{5, 5, 5, 5, 5}, NumericType::Discrete}};
    std::vector<CategoricalFeature> cats{{{10, 10, 1000000, 20, 10}}};
    Dataset columns(nums, cats, {"x", "constant", "kind"}, DatasetPolicy::Clean, {});
    Dataset flat({1, 5, 1, 5, nan, 5, 3, 5, 4, 5},
                 {10, 10, 1000000, 20, 10},
                 5,
                 2,
                 1,
                 {FeatureType::Numeric, FeatureType::Numeric, FeatureType::Categorical},
                 {NumericType::Continuous, NumericType::Discrete},
                 {"x", "constant", "kind"},
                 DatasetPolicy::Clean,
                 {});
    for (const Dataset* data : {&columns, &flat}) {
        data->validate();
        check(data->n_rows() == 3 && data->n_num() == 1 && data->n_cat() == 1);
        check(data->categories_size(0) == 21);
        const auto& report = data->preprocess_report();
        check(report.rows_dropped_missing == 1 && report.rows_dropped_duplicate == 1);
        check(report.columns_dropped_constant_idx == std::vector<int>{1});
    }
    check(columns.num_data() == flat.num_data() && columns.cat_data() == flat.cat_data());
    rejects([&] {
        Dataset(nums, cats, {"x", "constant", "kind"});
    });

    Dataset all_missing(std::vector<NumFeature>{{{nan, nan}, NumericType::Continuous}},
                        std::vector<CategoricalFeature>{{{0, 1}}, {{1, 0}}},
                        {"x", "a", "b"},
                        DatasetPolicy::Clean,
                        {});
    check(all_missing.empty() && all_missing.n_cols() == 3);
    check(all_missing.preprocess_report().rows_dropped_missing == 2);
    check(all_missing.col_cat_span(1).empty());
    check(all_missing.categories_size(1) == 0);
    all_missing.validate();
}
}  // namespace

int main() {
    check_empty_constructors();
    check_invalid_policies();
    check_malformed_empty_inputs();
    check_nonempty_policies();
}
