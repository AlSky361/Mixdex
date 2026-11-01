#include "mixdex/algorithms/outlier/lof.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/parallel.hpp"

namespace mixdex {

LOF::LOF(const Dataset& data, int k, SearchMethod method, int leaf_size) : k_(k) {
    const int n = static_cast<int>(data.n_rows());
    if (k < 1 || k > n - 1)
        throw std::invalid_argument("LOF: k must be in [1, n-1]");

    search_ = make_search(data, method, leaf_size);
    fit();
}

LOF::LOF(std::shared_ptr<const NeighbourSearch> search, int k) : search_(std::move(search)), k_(k) {
    if (!search_)
        throw std::invalid_argument("LOF: search must not be null");
    const int n = search_->n_rows();
    if (k < 1 || k > n - 1)
        throw std::invalid_argument("LOF: k must be in [1, n-1]");
    fit();
}

void LOF::fit() {
    const int n = search_->n_rows();
    const int k = k_;

    const std::size_t K = static_cast<std::size_t>(k);
    std::vector<int> nb_idx(static_cast<std::size_t>(n) * K);
    std::vector<double> nb_dist(static_cast<std::size_t>(n) * K);

    parallel_for(n, [&](int i) {
        const auto nb = search_->knn(i, k);
        for (std::size_t t = 0; t < K; ++t) {
            nb_idx[i * K + t] = nb[t].first;
            nb_dist[i * K + t] = nb[t].second;
        }
    });

    kdist_.resize(n);
    for (int i = 0; i < n; ++i) kdist_[i] = nb_dist[i * K + K - 1];

    lrd_.resize(n);
    parallel_for(n, [&](int i) {
        double sum = 0.0;
        for (std::size_t t = 0; t < K; ++t)
            sum += std::max(kdist_[nb_idx[i * K + t]], nb_dist[i * K + t]);
        lrd_[i] = 1.0 / (sum / k + constants::kLofLrdRegularizer);
    });

    lof_.resize(n);
    parallel_for(n, [&](int i) {
        double sum = 0.0;
        for (std::size_t t = 0; t < K; ++t) sum += lrd_[nb_idx[i * K + t]];
        lof_[i] = (sum / k) / lrd_[i];
    });
}

std::vector<double> LOF::score(std::span<const double> num,
                               std::span<const int> cat,
                               std::size_t m) const {
    search_->validate_batch(num, cat, m);

    const std::size_t nn = search_->dataset().n_num();
    const std::size_t no = search_->dataset().n_cat();

    std::vector<double> out(m);
    parallel_for(m, [&](long long i) {
        const RowView row{num.subspan(i * nn, nn), cat.subspan(i * no, no)};
        const auto nb = search_->knn(row, k_);
        double reach = 0.0, lrd_sum = 0.0;
        for (const auto& [j, d] : nb) {
            reach += std::max(kdist_[j], d);
            lrd_sum += lrd_[j];
        }
        const double lrd_q = 1.0 / (reach / k_ + constants::kLofLrdRegularizer);
        out[i] = (lrd_sum / k_) / lrd_q;
    });
    return out;
}

}  // namespace mixdex
