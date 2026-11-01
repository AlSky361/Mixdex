// Tie-heavy dual-tree MSTs must have the same edges across runs and thread counts.
// Equal-bound pruning must not bypass index tie resolution.
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "mixdex/density/core/core.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/types.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

Dataset lattice(int g, int blocks) {
    std::vector<double> num;
    std::size_t n = 0;
    for (int x = 0; x < g; ++x) {
        for (int y = 0; y < g; ++y) {
            int gap = 0;
            for (int b = 1; b < blocks; ++b)
                if (x >= b * g / blocks)
                    gap += 9;
            num.push_back(x + gap);
            num.push_back(y);
            ++n;
        }
    }
    std::vector<FeatureType> ft(2, FeatureType::Numeric);
    std::vector<NumericType> nt(2, NumericType::Continuous);
    return Dataset(std::move(num), {}, n, 2, 0, std::move(ft), std::move(nt), {"x", "y"});
}

using Canon = std::vector<std::tuple<int, int, double>>;

Canon canonical_mst(const BallTree& tree) {
    const std::vector<Edge> edges = dual_tree_boruvka(tree);
    Canon c;
    c.reserve(edges.size());
    for (const Edge& e : edges) c.emplace_back(std::min(e.u, e.v), std::max(e.u, e.v), e.w);
    std::sort(c.begin(), c.end());
    return c;
}

}  // namespace

int main() {
    const Dataset data = lattice(30, 3);  // 900 points with many equal distances.
    BallTree tree(data, 8);
    tree.set_core_dist(core_distances(tree, 4));

    const std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    Canon reference;
    bool have_ref = false;
    for (int t : thread_counts) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#else
        (void)t;
#endif
        for (int rep = 0; rep < 6; ++rep) {
            const Canon c = canonical_mst(tree);
            CHECK(c.size() + 1 == data.n_rows());
            if (!have_ref) {
                reference = c;
                have_ref = true;
            }
            CHECK(c == reference);  // Expect identical edges and weights in every run.
        }
    }
    // Nondeterministic MSTs may change tied edges, but must stay connected and minimal.
    std::vector<double> ref_weights;
    for (const auto& e : reference) ref_weights.push_back(std::get<2>(e));
    std::sort(ref_weights.begin(), ref_weights.end());
    for (int t : {1, 4, 16}) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#else
        (void)t;
#endif
        for (int rep = 0; rep < 4; ++rep) {
            const std::vector<Edge> fast = dual_tree_boruvka(tree, /*deterministic=*/false);
            CHECK(fast.size() + 1 == data.n_rows());
            std::vector<int> parent(data.n_rows());
            for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);
            auto find = [&](int x) {
                while (parent[x] != x) x = parent[x] = parent[parent[x]];
                return x;
            };
            std::vector<double> w;
            for (const Edge& e : fast) {
                const int a = find(e.u), b = find(e.v);
                CHECK(a != b);  // Reject cycles.
                parent[a] = b;
                w.push_back(e.w);
            }
            std::sort(w.begin(), w.end());
            CHECK(w == ref_weights);
        }
    }

    std::puts("OK");
    return 0;
}
