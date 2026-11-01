// Compare dual-tree MST weights with Prim and validate statistics.
// TSan also checks counters and component best-weight updates.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "mixdex/density/core/core.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/synthetic/synthetic.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// O(n^2) Prim reference using the same mutual-reachability distance.
static double prim_weight(const BallTree& tree, int n) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    std::vector<double> best(n, kInf);
    std::vector<char> in(n, 0);
    best[0] = 0.0;
    double total = 0.0;
    for (int it = 0; it < n; ++it) {
        int u = -1;
        for (int v = 0; v < n; ++v)
            if (!in[v] && (u < 0 || best[v] < best[u]))
                u = v;
        in[u] = 1;
        total += best[u];
        for (int v = 0; v < n; ++v)
            if (!in[v]) {
                const double w = tree.call_metric(u, v);
                if (w < best[v])
                    best[v] = w;
            }
    }
    return total;
}

static void run_case(std::size_t n, double p_cat, int m_pts, int leaf_size, std::uint64_t seed) {
    SyntheticParams p;
    p.n = n;
    p.d_num = 3;
    p.n_clusters = 4;
    p.p_cat = p_cat;
    p.seed = seed;
    const SyntheticResult syn = make_synthetic(p);

    BallTree tree(syn.dataset, leaf_size);
    tree.set_core_dist(core_distances(tree, m_pts));
    const int nn = tree.n_rows();

    const double ref = prim_weight(tree, nn);

    const DTBoruvkaResult res = dual_tree_boruvka_with_stats(tree);
    CHECK(static_cast<int>(res.edges.size()) == nn - 1);
    double got = 0.0;
    for (const Edge& e : res.edges) got += e.w;
    CHECK(std::fabs(got - ref) <= 1e-9 * (1.0 + std::fabs(ref)));

    const std::vector<Edge> plain = dual_tree_boruvka(tree);
    double got2 = 0.0;
    for (const Edge& e : plain) got2 += e.w;
    CHECK(std::fabs(got2 - ref) <= 1e-9 * (1.0 + std::fabs(ref)));

    const DTBoruvkaStats& s = res.stats;
    CHECK(s.total_calls > 0);
    CHECK(s.pruned_calls >= 0 && s.pruned_calls <= s.total_calls);
    CHECK(s.zero_lb_calls >= 0 && s.zero_lb_calls <= s.total_calls);
    CHECK(s.leaf_pairs >= 0 && s.leaf_pairs <= s.total_calls);
    CHECK(s.metric_calls >= 0 && s.fast_skipped >= 0);
    CHECK(s.iterations >= 1);
}

int main() {
    run_case(600, 0.0, 5, 8, 1);
    run_case(600, 0.4, 5, 8, 2);
    run_case(900, 0.0, 1, 4, 3);
    run_case(900, 0.7, 8, 16, 4);
    std::puts("test_dt_boruvka: OK");
    return 0;
}
