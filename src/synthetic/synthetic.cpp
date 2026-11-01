#include "mixdex/synthetic/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mixdex {

namespace {

std::vector<std::size_t> cluster_sizes(std::size_t n, int n_clusters, double cluster_density) {
    if (n_clusters <= 0)
        throw std::invalid_argument("n_clusters must be positive");
    std::vector<std::size_t> sizes(static_cast<std::size_t>(n_clusters));

    if (cluster_density <= 1.0 || n_clusters == 1) {
        std::size_t base = n / static_cast<std::size_t>(n_clusters);
        std::fill(sizes.begin(), sizes.end(), base);
        sizes.back() += n - base * static_cast<std::size_t>(n_clusters);
        return sizes;
    }

    std::size_t small = static_cast<std::size_t>(static_cast<double>(n) / cluster_density);
    small = std::max<std::size_t>(small, 1);
    std::size_t remaining = n - small;
    std::size_t rest_count = static_cast<std::size_t>(n_clusters - 1);
    std::size_t base = remaining / rest_count;

    sizes[0] = small;
    for (std::size_t i = 1; i < sizes.size(); ++i) sizes[i] = base;
    sizes.back() += remaining - base * rest_count;
    return sizes;
}

std::vector<std::vector<double>> cluster_centers(int n_clusters,
                                                 std::size_t d_num,
                                                 double separation,
                                                 double cluster_std) {
    std::vector<std::vector<double>> centers(static_cast<std::size_t>(n_clusters),
                                             std::vector<double>(d_num, 0.0));
    for (int k = 0; k < n_clusters; ++k) {
        std::size_t axis = static_cast<std::size_t>(k) % std::max<std::size_t>(d_num, 1);
        if (d_num > 0) {
            centers[static_cast<std::size_t>(k)][axis] =
                static_cast<double>(k) * separation * cluster_std;
        }
    }
    return centers;
}

}  // namespace

SyntheticResult make_synthetic(const SyntheticParams& p) {
    if (p.n == 0)
        throw std::invalid_argument("n must be positive");
    if (p.d_num == 0 && p.p_cat <= 0.0) {
        throw std::invalid_argument(
            "dataset must have at least one feature (numeric or categorical)");
    }
    if (p.p_cat < 0.0 || p.p_cat > 1.0) {
        throw std::invalid_argument("p_cat must be in [0,1]");
    }
    if (p.cardinality < 2)
        throw std::invalid_argument("cardinality must be >= 2");
    if (p.cluster_density < 1.0)
        throw std::invalid_argument("cluster_density must be >= 1.0");
    if (p.elongation < 1.0)
        throw std::invalid_argument("elongation must be >= 1.0");

    double cat_separation = (p.cat_separation < 0.0) ? p.separation : p.cat_separation;

    std::size_t d_cat = 0;
    if (p.p_cat > 0.0) {
        double d_total = (p.p_cat >= 1.0) ? static_cast<double>(std::max<std::size_t>(p.d_num, 1))
                                          : static_cast<double>(p.d_num) / (1.0 - p.p_cat);
        d_cat = static_cast<std::size_t>(std::llround(d_total * p.p_cat));
        d_cat = std::max<std::size_t>(d_cat, p.p_cat > 0.0 ? 1 : 0);
    }
    std::size_t d_num_eff = (p.p_cat >= 1.0) ? 0 : p.d_num;

    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    auto sizes = cluster_sizes(p.n, p.n_clusters, p.cluster_density);
    std::size_t d_num_for_centers = std::max<std::size_t>(p.d_num, 1);
    auto centers = cluster_centers(p.n_clusters, d_num_for_centers, p.separation, p.cluster_std);

    std::vector<double> num_data;
    std::vector<double> latent_for_cat;
    std::vector<int> labels;
    num_data.reserve(p.n * d_num_eff);
    latent_for_cat.reserve(p.n * d_cat);
    labels.reserve(p.n);

    for (int k = 0; k < p.n_clusters; ++k) {
        std::size_t k_idx = static_cast<std::size_t>(k);
        for (std::size_t j = 0; j < sizes[k_idx]; ++j) {
            for (std::size_t ax = 0; ax < d_num_for_centers; ++ax) {
                double std_ax = (ax == 0) ? p.cluster_std * p.elongation : p.cluster_std;
                double val = centers[k_idx][ax] + gauss(rng) * std_ax;
                if (ax < d_num_eff)
                    num_data.push_back(val);
            }
            for (std::size_t c = 0; c < d_cat; ++c) {
                double val = static_cast<double>(k) * cat_separation * p.cluster_std +
                             gauss(rng) * p.cluster_std;
                latent_for_cat.push_back(val);
            }
            labels.push_back(k);
        }
    }

    std::vector<int> cat_data;
    cat_data.resize(p.n * d_cat);
    for (std::size_t c = 0; c < d_cat; ++c) {
        std::vector<double> col(p.n);
        for (std::size_t i = 0; i < p.n; ++i) col[i] = latent_for_cat[i * d_cat + c];
        std::vector<double> sorted_col = col;
        std::sort(sorted_col.begin(), sorted_col.end());

        std::vector<double> cuts(static_cast<std::size_t>(p.cardinality - 1));
        for (int q = 1; q < p.cardinality; ++q) {
            double frac = static_cast<double>(q) / static_cast<double>(p.cardinality);
            std::size_t idx = static_cast<std::size_t>(frac * static_cast<double>(p.n));
            idx = std::min(idx, p.n - 1);
            cuts[static_cast<std::size_t>(q - 1)] = sorted_col[idx];
        }
        for (std::size_t i = 0; i < p.n; ++i) {
            int bin =
                static_cast<int>(std::upper_bound(cuts.begin(), cuts.end(), col[i]) - cuts.begin());
            cat_data[i * d_cat + c] = bin;
        }
    }

    std::vector<FeatureType> feat_types(d_num_eff, FeatureType::Numeric);
    feat_types.insert(feat_types.end(), d_cat, FeatureType::Categorical);
    std::vector<NumericType> num_types(d_num_eff, NumericType::Continuous);
    std::vector<std::string> feat_names;
    feat_names.reserve(d_num_eff + d_cat);
    for (std::size_t i = 0; i < d_num_eff; ++i) feat_names.push_back("num_" + std::to_string(i));
    for (std::size_t i = 0; i < d_cat; ++i) feat_names.push_back("cat_" + std::to_string(i));

    Dataset ds(std::move(num_data),
               std::move(cat_data),
               p.n,
               d_num_eff,
               d_cat,
               std::move(feat_types),
               std::move(num_types),
               std::move(feat_names));

    return SyntheticResult{std::move(ds), std::move(labels)};
}

}  // namespace mixdex
