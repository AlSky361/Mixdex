#pragma once
#include <cstdint>
#include <vector>

#include "mixdex/types.hpp"

namespace mixdex {

struct SyntheticParams {
    std::size_t n = 300;
    std::size_t d_num = 2;  // Numeric count; total feature count when p_cat=1.
    int n_clusters = 3;
    double separation = 2.0;
    double cat_separation = -1.0;  // Negative values reuse separation.
    double p_cat = 0.0;            // Target categorical fraction; column counts are rounded.
    int cardinality = 4;
    double cluster_density = 1.0;
    double elongation = 1.0;
    double cluster_std = 0.3;
    std::uint64_t seed = 0;
};

struct SyntheticResult {
    Dataset dataset;
    std::vector<int> true_labels;
};

SyntheticResult make_synthetic(const SyntheticParams& params);

}  // namespace mixdex
