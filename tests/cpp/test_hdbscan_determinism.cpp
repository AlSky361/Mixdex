// Check full HDBSCAN determinism across thread counts on tie-heavy data.
//
// Cover the pipeline after MST construction, including extraction and GLOSH.
// Continuous fixtures retain the regression for repeated point departures.
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mixdex/algorithms/hdbscan/hdbscan.hpp"
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

// Unique lattice rows with low-cardinality categories create many distance ties.
Dataset lattice_mixed(int g) {
    std::vector<double> num;
    std::vector<int> cat;
    std::size_t n = 0;
    for (int x = 0; x < g; ++x) {
        for (int y = 0; y < g; ++y) {
            num.push_back(static_cast<double>(x));
            num.push_back(static_cast<double>(y));
            cat.push_back((x + y) % 3);
            ++n;
        }
    }
    std::vector<FeatureType> ft = {
        FeatureType::Numeric, FeatureType::Numeric, FeatureType::Categorical};
    std::vector<NumericType> nt = {NumericType::Continuous, NumericType::Continuous};
    return Dataset(
        std::move(num), std::move(cat), n, 2, 1, std::move(ft), std::move(nt), {"x", "y", "c"});
}

// Unique random lattices expose different tied MSTs when determinism is disabled.
// Use 1D/3D inputs to exercise canonical cluster labels.
Dataset random_lattice(std::uint64_t seed, std::size_t n, std::size_t dn, std::size_t dc) {
    std::mt19937_64 rng(seed);
    const std::size_t d = dn + dc;
    const int card = static_cast<int>(
        std::ceil(std::pow(1.5 * static_cast<double>(n), 1.0 / static_cast<double>(d))));
    std::set<std::vector<int>> seen;
    std::vector<double> num;
    std::vector<int> cat;
    while (seen.size() < n) {
        std::vector<int> row(d);
        for (int& v : row) v = static_cast<int>(rng() % static_cast<std::uint64_t>(card));
        if (seen.insert(row).second) {
            for (std::size_t k = 0; k < dn; ++k) num.push_back(row[k]);
            for (std::size_t k = 0; k < dc; ++k) cat.push_back(row[dn + k]);
        }
    }
    std::vector<FeatureType> ft;
    std::vector<NumericType> nt;
    std::vector<std::string> names;
    for (std::size_t k = 0; k < dn; ++k) {
        ft.push_back(FeatureType::Numeric);
        nt.push_back(NumericType::Continuous);
        names.push_back("n" + std::to_string(k));
    }
    for (std::size_t k = 0; k < dc; ++k) {
        ft.push_back(FeatureType::Categorical);
        names.push_back("c" + std::to_string(k));
    }
    return Dataset(
        std::move(num), std::move(cat), n, dn, dc, std::move(ft), std::move(nt), std::move(names));
}

Dataset synthetic_continuous(std::uint64_t seed) {
    SyntheticParams p;
    p.n = 300;
    p.d_num = 3;
    p.n_clusters = 3;
    p.separation = 3.0;
    p.cardinality = 4;
    p.seed = seed;
    return make_synthetic(p).dataset;
}

// Labels and GLOSH match exactly; stability sums allow floating-point rounding differences.
bool same_up_to_rounding(const HDBSCANResult& a, const HDBSCANResult& b) {
    if (a.labels != b.labels || a.outlier_scores != b.outlier_scores)
        return false;
    if (a.cluster_stabilities.size() != b.cluster_stabilities.size())
        return false;
    for (std::size_t i = 0; i < a.cluster_stabilities.size(); ++i) {
        const double x = a.cluster_stabilities[i], y = b.cluster_stabilities[i];
        if (std::abs(x - y) > 1e-12 * std::max(std::abs(x), std::abs(y)))
            return false;
    }
    return true;
}

std::string canonical(const HDBSCANResult& r) {
    std::ostringstream os;
    os.precision(17);
    for (int lbl : r.labels) os << lbl << ' ';
    os << '|';
    for (double s : r.cluster_stabilities) os << s << ' ';
    os << '|';
    for (double s : r.outlier_scores) os << s << ' ';
    return os.str();
}

}  // namespace

int main() {
    struct NamedDataset {
        const char* name;
        Dataset ds;
        int min_cluster_size;
    };
    std::vector<NamedDataset> datasets;
    // Tie-heavy lattice with no repeated point departures.
    datasets.push_back({"lattice", lattice_mixed(14), 5});
    // Continuous fixtures retain the repeated-departure GLOSH regression.
    // Also exercise 1D lines, a 3D lattice and a categorical lattice.
    datasets.push_back({"line_1d", random_lattice(21, 90, 1, 0), 5});
    datasets.push_back({"line_1d", random_lattice(22, 260, 1, 0), 5});
    datasets.push_back({"lattice_3d", random_lattice(23, 390, 3, 0), 5});
    datasets.push_back({"lattice_2n1c", random_lattice(24, 250, 2, 1), 5});
    for (std::uint64_t seed : {1, 2, 3}) {
        datasets.push_back({"synthetic", synthetic_continuous(seed), 5});
        datasets.push_back({"synthetic", synthetic_continuous(seed), 10});
    }

    struct Mode {
        const char* name;
        HDBSCANMode mode;
        int k_opt;
    };
    // Approximate requires connected kNN graphs with k_opt in [m_pts, sqrt(n)].
    const Mode modes[] = {{"Exact", HDBSCANMode::Exact, constants::kKUnset},
                          {"Approximate", HDBSCANMode::Approximate, 12},
                          // Automatic mutual-reachability k selection must also be deterministic.
                          {"ApproximateAuto", HDBSCANMode::Approximate, constants::kKUnset}};
    const int thread_counts[] = {2, 4, 8, 16};

    auto run_result = [&](const NamedDataset& d, const Mode& m, int threads, bool det_mst) {
#ifdef _OPENMP
        // Set threads explicitly: omp_get_max_threads would remain 1 after the serial run.
        omp_set_num_threads(threads);
        CHECK(omp_get_max_threads() == threads);
#else
        (void)threads;
#endif
        // Approximate k_opt must not exceed sqrt(n).
        const int k_opt =
            (m.k_opt == constants::kKUnset)
                ? m.k_opt
                : std::min(m.k_opt,
                           static_cast<int>(std::sqrt(static_cast<double>(d.ds.n_rows()))));
        return hdbscan(d.ds,
                       /*m_pts=*/5,
                       d.min_cluster_size,
                       m.mode,
                       SearchMethod::BallTree,
                       constants::kDefaultLeafSize,
                       k_opt,
                       det_mst);
    };
    auto run = [&](const NamedDataset& d, const Mode& m, int threads) {
        return canonical(run_result(d, m, threads, /*det_mst=*/true));
    };

    for (const NamedDataset& d : datasets) {
        for (const Mode& m : modes) {
            const std::string ref_text = run(d, m, 1);
            CHECK(!ref_text.empty());
            for (int T : thread_counts) {
                for (int rep = 0; rep < 6; ++rep) {
                    const std::string text = run(d, m, T);
                    if (text != ref_text) {
                        std::fprintf(stderr,
                                     "FAILED: hdbscan(%s, %s, mcs=%d) at T=%d, rep=%d differs "
                                     "from T=1\nreference labels|stabilities|scores:\n%s\n"
                                     "got:\n%s\n",
                                     d.name,
                                     m.name,
                                     d.min_cluster_size,
                                     T,
                                     rep,
                                     ref_text.c_str(),
                                     text.c_str());
                        std::exit(1);
                    }
                }
            }
            // Nondeterministic Exact MSTs may differ on ties. Canonical labels and GLOSH
            // match exactly; stability sums may differ by rounding. Approximate ignores the flag.
            const HDBSCANResult ref_result = run_result(d, m, 1, /*det_mst=*/true);
            for (int T : {1, 4, 16}) {
                for (int rep = 0; rep < 6; ++rep) {
                    const HDBSCANResult got = run_result(d, m, T, /*det_mst=*/false);
                    if (!same_up_to_rounding(ref_result, got)) {
                        std::fprintf(stderr,
                                     "FAILED: hdbscan(%s, %s, mcs=%d, deterministic_mst=false) at "
                                     "T=%d, rep=%d differs from the deterministic reference\n"
                                     "reference labels|stabilities|scores:\n%s\ngot:\n%s\n",
                                     d.name,
                                     m.name,
                                     d.min_cluster_size,
                                     T,
                                     rep,
                                     canonical(ref_result).c_str(),
                                     canonical(got).c_str());
                        std::exit(1);
                    }
                }
            }
        }
    }

    std::printf("OK\n");
    return 0;
}
