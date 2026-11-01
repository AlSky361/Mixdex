// Compare indexed algorithms at one and four threads, including TSan coverage.
// Exercise DBSCAN, kNN, LOF, scaling, HDBSCAN and Approximate graph construction.
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mixdex/algorithms/dbscan/dbscan.hpp"
#include "mixdex/algorithms/hdbscan/hdbscan.hpp"
#include "mixdex/algorithms/knn/knn_classifier.hpp"
#include "mixdex/algorithms/outlier/lof.hpp"
#include "mixdex/scale.hpp"
#include "mixdex/synthetic/synthetic.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/constants.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

void set_threads(int t) {
#ifdef _OPENMP
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}

Dataset make_data(std::uint64_t seed, double p_cat) {
    SyntheticParams p;
    p.n = 400;
    p.d_num = 4;
    p.n_clusters = 3;
    p.separation = 3.0;
    p.p_cat = p_cat;
    p.cardinality = 4;
    p.seed = seed;
    return make_synthetic(p).dataset;
}

struct Outputs {
    std::vector<int> dbscan_labels, classify, predict;
    std::vector<double> lof, lof_new;
    HDBSCANResult hd_exact, hd_approx, hd_brute;
    Dataset scaled;
};

Outputs run_all(const Dataset& data) {
    Outputs o;
    o.dbscan_labels = dbscan(data, 0.25, 5);

    std::vector<int> labels(static_cast<std::size_t>(data.n_rows()), kUnlabeled);
    for (int i = 0; i < data.n_rows(); i += 3) labels[static_cast<std::size_t>(i)] = i % 3;
    o.classify = knn_classify(data, labels, 5, true);

    // Reuse input rows as external queries to exercise knn_predict.
    const int m = 40;
    std::vector<double> num;
    std::vector<int> cat;
    for (int i = 0; i < m; ++i) {
        for (double v : data.row_num_span(i)) num.push_back(v + 0.01);
        for (int v : data.row_cat_span(i)) cat.push_back(v);
    }
    o.predict = knn_predict(data, labels, num, cat, static_cast<std::size_t>(m), 5, true);

    LOF lof(data, 10);
    o.lof = lof.scores();
    o.lof_new = lof.score(num, cat, static_cast<std::size_t>(m));

    o.hd_exact = hdbscan(data, 5, 10, HDBSCANMode::Exact, SearchMethod::BallTree);
    o.hd_approx = hdbscan(data,
                          5,
                          10,
                          HDBSCANMode::Approximate,
                          SearchMethod::BallTree,
                          constants::kDefaultLeafSize,
                          /*k_opt=*/15);
    o.hd_brute = hdbscan(data,
                         5,
                         10,
                         HDBSCANMode::Approximate,
                         SearchMethod::BruteForce,
                         constants::kDefaultLeafSize,
                         /*k_opt=*/15);

    o.scaled = scale(data);
    return o;
}

bool same(const HDBSCANResult& a, const HDBSCANResult& b) {
    return a.labels == b.labels && a.outlier_scores == b.outlier_scores &&
           a.cluster_stabilities.size() == b.cluster_stabilities.size();
}

}  // namespace

int main() {
    for (std::uint64_t seed : {1ull, 2ull}) {
        for (double p_cat : {0.0, 0.5}) {
            const Dataset data = make_data(seed, p_cat);
            set_threads(1);
            const Outputs a = run_all(data);
            set_threads(4);
            const Outputs b = run_all(data);

            CHECK(a.dbscan_labels == b.dbscan_labels);
            CHECK(a.classify == b.classify);
            CHECK(a.predict == b.predict);
            CHECK(a.lof == b.lof);
            CHECK(a.lof_new == b.lof_new);
            CHECK(same(a.hd_exact, b.hd_exact));
            CHECK(same(a.hd_approx, b.hd_approx));
            CHECK(same(a.hd_brute, b.hd_brute));
            CHECK(a.scaled.n_rows() == b.scaled.n_rows());
            for (int i = 0; i < a.scaled.n_rows(); ++i) {
                const auto ra = a.scaled.row_num_span(i), rb = b.scaled.row_num_span(i);
                for (std::size_t k = 0; k < ra.size(); ++k) CHECK(ra[k] == rb[k]);
            }
        }
    }
    std::puts("OK");
    return 0;
}
