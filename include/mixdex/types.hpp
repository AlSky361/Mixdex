#pragma once
#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "mixdex/utils/aliases.hpp"
#include "mixdex/utils/strided_span.hpp"

namespace mixdex {

enum class NumericType { Continuous, Discrete };
enum class FeatureType { Numeric, Categorical, CategoricalSet };

// Strict rejects missing values, duplicate rows and constants; Clean removes them.
// Both preserve a valid schema with zero rows.
enum class DatasetPolicy { Strict, Clean };

class Dataset;

/// Non-owning category view; input codes must be sorted.
class CategorySet {
public:
    constexpr CategorySet() noexcept = default;
    constexpr explicit CategorySet(std::span<const int> cats) noexcept : categories(cats) {
    }

    constexpr bool contains(int c) const noexcept {
        return std::binary_search(categories.begin(), categories.end(), c);
    }

    bool intersects(const CategorySet& other) const noexcept;

    auto begin() const noexcept {
        return categories.begin();
    }
    auto end() const noexcept {
        return categories.end();
    }

    std::size_t size() const noexcept {
        return categories.size();
    }

private:
    std::span<const int> categories;
};

// Node categories share one buffer with per-feature offsets.
class NodeCatPresent {
public:
    void build(const Dataset& data, const std::vector<int>& idx, int low, int high);
    static NodeCatPresent merge(const NodeCatPresent& a, const NodeCatPresent& b);

    CategorySet operator[](int k) const noexcept {
        return CategorySet(
            std::span<const int>(categories_.data() + offsets_[k], offsets_[k + 1] - offsets_[k]));
    }

    int n_features() const noexcept {
        return offsets_.empty() ? 0 : static_cast<int>(offsets_.size() - 1);
    }

    bool empty() const noexcept {
        return n_features() == 0;
    }

private:
    std::vector<int> categories_;
    std::vector<std::size_t> offsets_;
};

struct NumFeature {
    std::vector<double> array;
    NumericType type{NumericType::Continuous};

    constexpr std::size_t size() const noexcept {
        return array.size();
    }
};

/// Non-owning row view with separate numeric and categorical blocks.
struct RowView {
    std::span<const double> num;
    std::span<const Code> cat;
    std::span<const std::vector<Code>> cat_sets{};  // Optional query sets, by categorical column.
};

struct CategoricalFeature {
    std::vector<Code> encoded;

    constexpr std::size_t size() const noexcept {
        return encoded.size();
    }
};

/// Records construction cleanup and projection after weight changes.
struct PreprocessReport {
    std::size_t rows_dropped_missing{0};            // NaN, infinity or a negative category code.
    std::size_t rows_dropped_duplicate{0};          // Exact duplicates; signed zeros compare equal.
    std::vector<int> columns_dropped_constant_idx;  // Removed constant features.
    std::vector<std::string> columns_dropped_constant_name;  // Same order as the indices.
    std::vector<int> columns_dropped_zero_weight_idx;        // Indices at the time of set_weights.
    std::vector<std::string> columns_dropped_zero_weight_name;

    constexpr bool empty() const noexcept {
        return rows_dropped_missing == 0 && rows_dropped_duplicate == 0 &&
               columns_dropped_constant_idx.empty() && columns_dropped_zero_weight_idx.empty();
    }
};

using PreprocessWarningHandler = std::function<void(const std::string&)>;
/// Send Clean-policy warnings to stderr.
void default_preprocess_warning_handler(const std::string& msg);

/// Own separate row-major blocks; feature metadata lists numeric columns first.
class Dataset {
public:
    Dataset() = default;

    explicit Dataset(
        const std::vector<NumFeature>& num_features,
        const std::vector<CategoricalFeature>& cat_features,
        const std::vector<std::string>& feat_names,
        DatasetPolicy policy = DatasetPolicy::Strict,
        const PreprocessWarningHandler& on_warning = default_preprocess_warning_handler);

    explicit Dataset(
        std::vector<double> num_data,
        std::vector<Code> cat_data,
        std::size_t n,
        std::size_t n_num,
        std::size_t n_cat,
        std::vector<FeatureType> feat_types,
        std::vector<NumericType> num_types,
        std::vector<std::string> feat_names,
        DatasetPolicy policy = DatasetPolicy::Strict,
        const PreprocessWarningHandler& on_warning = default_preprocess_warning_handler);

    std::span<const double> row_num_span(int i) const noexcept {
        if (n_num_ == 0)
            return {};
        return {num_data_.data() + static_cast<std::size_t>(i) * n_num_, n_num_};
    }
    std::span<const int> row_cat_span(int i) const noexcept {
        if (n_cat_ == 0)
            return {};
        return {cat_data_.data() + static_cast<std::size_t>(i) * n_cat_, n_cat_};
    }

    RowView row(int i) const noexcept {
        return {row_num_span(i), row_cat_span(i)};
    }

    /// Borrow a numeric column; the view expires when storage changes.
    StridedSpan<double> col_num_span(int k) const noexcept {
        return StridedSpan<double>(
            n_ == 0 ? num_data_.data() : num_data_.data() + static_cast<std::size_t>(k),
            static_cast<std::size_t>(n_num()),
            static_cast<std::size_t>(n_rows()));
    }
    /// Borrow a categorical column; the view expires when storage changes.
    StridedSpan<int> col_cat_span(int k) const noexcept {
        return StridedSpan<int>(
            n_ == 0 ? cat_data_.data() : cat_data_.data() + static_cast<std::size_t>(k),
            static_cast<std::size_t>(n_cat()),
            static_cast<std::size_t>(n_rows()));
    }

    std::size_t categories_size(int k) const noexcept {
        return static_cast<std::size_t>(categories_count_[k]);
    }

    constexpr int n_rows() const noexcept {
        return static_cast<int>(n_);
    }
    constexpr int n_num() const noexcept {
        return static_cast<int>(n_num_);
    }
    constexpr int n_cat() const noexcept {
        return static_cast<int>(n_cat_);
    }
    constexpr int n_cols() const noexcept {
        return static_cast<int>(feat_types_.size());
    }

    constexpr bool empty() const noexcept {
        return n_ == 0;
    }

    FeatureType feat_type(int k) const noexcept;
    NumericType num_type(int k) const noexcept;
    const std::string& feat_name(int k) const noexcept;

    /// Validate current data without modifying it.
    void validate() const;

    // Cleanup report, including zero-weight projection.
    const PreprocessReport& preprocess_report() const noexcept {
        return report_;
    }

    const std::vector<double>& num_data() const noexcept {
        return num_data_;
    }

    const std::vector<Code>& cat_data() const noexcept {
        return cat_data_;
    }

    /// Require finite nonnegative weights with a positive finite sum.
    /// Zero weights remove features and resulting duplicate rows; keep the first row.
    /// Callers must rebuild existing indices; removed features cannot be restored here.
    void set_weights(std::vector<double> weights);

    /// Attach a dictionary of distinct finite sets to categorical column k.
    /// Codes index the dictionary; rebuild indices after changing it.
    void set_category_sets(int k, std::vector<std::vector<Code>> sets);
    const auto& category_sets() const noexcept {
        return category_sets_;
    }
    double categorical_distance(int k, Code a, Code b) const noexcept;
    double categorical_distance(int k, std::span<const Code> query, Code stored) const noexcept;

    double weight(int k) const noexcept {
        return weights_.empty() ? 1.0 : weights_[static_cast<std::size_t>(k)];
    }

    double inv_weights_sum() const noexcept {
        return inv_W_;
    }

    bool has_weights() const noexcept {
        return !weights_.empty();
    }

private:
    // Validate structure, then missing values, duplicates and constants.
    void validate_all(std::vector<Code>* cat_max,
                      DatasetPolicy policy,
                      const PreprocessWarningHandler& on_warning);
    void check_limits() const;
    void check_structure() const;
    void check_constant_columns() const;
    void project_columns(std::span<const unsigned char> keep);
    // Strict rejects NaN/inf or negative codes; Clean marks affected rows.
    void scan_values(std::vector<Code>* cat_max,
                     DatasetPolicy policy,
                     std::vector<unsigned char>* rows_missing) const;
    // Compare finite coordinates exactly, including -0.0 == +0.0.
    // Clean keeps the first row and skips rows already marked for removal.
    void check_duplicates(DatasetPolicy policy,
                          const std::vector<unsigned char>& rows_dropped_already,
                          std::vector<unsigned char>* rows_duplicate) const;
    void drop_rows(const std::vector<unsigned char>& rows_missing,
                   const std::vector<unsigned char>& rows_duplicate);
    // Empty datasets have no constant columns.
    std::vector<unsigned char> find_constant_columns() const;
    // Check constants after row cleanup; Strict rejects, Clean removes.
    void drop_constant_columns(DatasetPolicy policy);
    void finish_construction(DatasetPolicy policy, const PreprocessWarningHandler& on_warning);

    // Separate row-major blocks; feature metadata lists numeric columns first.
    std::vector<double> num_data_;

    std::vector<Code> cat_data_;
    std::vector<int> categories_count_;

    std::vector<FeatureType> feat_types_;
    std::vector<NumericType> num_types_;
    std::vector<std::string> feat_names_;

    std::size_t n_{0};
    std::size_t n_num_{0};
    std::size_t n_cat_{0};

    std::map<int, std::vector<std::vector<Code>>> category_sets_;
    std::vector<double> weights_;
    double inv_W_{0.0};

    PreprocessReport report_;
};

}  // namespace mixdex
