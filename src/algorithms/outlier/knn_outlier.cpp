#include "mixdex/algorithms/outlier/knn_outlier.hpp"

#include <stdexcept>

namespace mixdex {

std::vector<double> knn_distance_scores(const NeighbourSearch& search, int k) {
    const int n = search.n_rows();
    if (k < 1 || k > n - 1)
        throw std::invalid_argument("knn_distance_scores: k must be in [1, n-1]");
    return search.kth_distance_all(k);
}

std::vector<double> knn_distance_scores(const NeighbourSearch& search,
                                        std::span<const double> num,
                                        std::span<const int> cat,
                                        std::size_t m,
                                        int k) {
    if (k < 1 || k > search.n_rows())
        throw std::invalid_argument("knn_distance_scores: k must be in [1, n]");
    return search.kth_distance_batch(num, cat, m, k);
}

}  // namespace mixdex
