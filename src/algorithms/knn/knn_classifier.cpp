#include "mixdex/algorithms/knn/knn_classifier.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/parallel.hpp"

namespace mixdex {

namespace {

int vote(const std::vector<std::pair<int, double>>& nb, bool weighted) {
    bool has_zero = false;
    if (weighted) {
        for (const auto& [lab, d] : nb)
            if (constants::f_zero(d)) {
                has_zero = true;
                break;
            }
    }

    std::vector<std::pair<int, double>> acc;
    acc.reserve(nb.size());
    for (const auto& [lab, d] : nb) {
        double w = 1.0;
        if (weighted) {
            if (has_zero)
                w = constants::f_zero(d) ? 1.0 : 0.0;
            else
                w = 1.0 / d;
        }
        acc.emplace_back(lab, w);
    }
    std::sort(acc.begin(), acc.end());

    int best_label = acc.front().first;
    double best_w = -1.0;
    for (std::size_t i = 0; i < acc.size();) {
        std::size_t j = i;
        double sum = 0.0;
        while (j < acc.size() && acc[j].first == acc[i].first) sum += acc[j++].second;
        if (sum > best_w) {
            best_w = sum;
            best_label = acc[i].first;
        }
        i = j;
    }
    return best_label;
}

}  // namespace

namespace {

int count_labeled(const std::vector<int>& labels, int n) {
    if (static_cast<int>(labels.size()) != n)
        throw std::invalid_argument("knn_classify: labels size must equal n_rows");
    int n_labeled = 0;
    for (int l : labels) {
        if (l < kUnlabeled)
            throw std::invalid_argument("knn_classify: labels must be >= 0 or kUnlabeled (-1)");
        if (l != kUnlabeled)
            ++n_labeled;
    }
    if (n_labeled == 0)
        throw std::invalid_argument("knn_classify: no labeled points");
    return n_labeled;
}

template <typename KnnFn>
int predict_point(
    const std::vector<int>& labels, int need, int max_q, bool weighted, KnnFn&& knn_fn) {
    std::vector<std::pair<int, double>> picked;
    int kk = std::max(need, 1);
    while (true) {
        const int q = std::min(kk, max_q);
        picked.clear();
        for (const auto& [j, d] : knn_fn(q)) {
            if (labels[j] != kUnlabeled) {
                picked.emplace_back(labels[j], d);
                if (static_cast<int>(picked.size()) == need)
                    break;
            }
        }
        if (static_cast<int>(picked.size()) == need || q >= max_q)
            break;
        kk *= 2;
    }
    return vote(picked, weighted);
}

}  // namespace

std::vector<int> knn_classify(const NeighbourSearch& search,
                              const std::vector<int>& labels,
                              int k,
                              bool distance_weighted) {
    const int n = search.n_rows();
    if (k < 1)
        throw std::invalid_argument("knn_classify: k must be >= 1");
    const int need = std::min(k, count_labeled(labels, n));
    std::vector<int> out(labels);

    parallel_for(n, [&](int i) {
        if (labels[i] != kUnlabeled)
            return;
        out[i] = predict_point(labels, need, n - 1, distance_weighted, [&](int q) {
            return search.knn(i, q);
        });
    });
    return out;
}

std::vector<int> knn_predict(const NeighbourSearch& search,
                             const std::vector<int>& labels,
                             std::span<const double> num,
                             std::span<const int> cat,
                             std::size_t m,
                             int k,
                             bool distance_weighted) {
    const int n = search.n_rows();
    if (k < 1)
        throw std::invalid_argument("knn_predict: k must be >= 1");
    const int need = std::min(k, count_labeled(labels, n));

    search.validate_batch(num, cat, m);

    const std::size_t nn = search.dataset().n_num();
    const std::size_t no = search.dataset().n_cat();

    std::vector<int> out(m);
    parallel_for(m, [&](long long i) {
        const RowView row{num.subspan(i * nn, nn), cat.subspan(i * no, no)};
        out[i] = predict_point(labels, need, n, distance_weighted, [&](int q) {
            return search.knn(row, q);
        });
    });
    return out;
}

std::vector<int> knn_predict(const Dataset& data,
                             const std::vector<int>& labels,
                             std::span<const double> num,
                             std::span<const int> cat,
                             std::size_t m,
                             int k,
                             bool distance_weighted,
                             SearchMethod method,
                             int leaf_size) {
    auto search = make_search(data, method, leaf_size);
    return knn_predict(*search, labels, num, cat, m, k, distance_weighted);
}

std::vector<int> knn_classify(const Dataset& data,
                              const std::vector<int>& labels,
                              int k,
                              bool distance_weighted,
                              SearchMethod method,
                              int leaf_size) {
    auto search = make_search(data, method, leaf_size);
    return knn_classify(*search, labels, k, distance_weighted);
}

}  // namespace mixdex
