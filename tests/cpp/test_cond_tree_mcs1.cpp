// Regression: min_cluster_size=1 can create more than n clusters.
// Check invariants; ASan detects reference invalidation from insufficient reserve.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/utils/graph.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

bool order(const Edge& a, const Edge& b) {
    if (a.w != b.w)
        return a.w > b.w;
    bool as = a.u == a.v, bs = b.u == b.v;
    if (as != bs)
        return bs;
    if (a.u != b.u)
        return a.u < b.u;
    return a.v < b.v;
}

}  // namespace

int main() {
    std::mt19937 rng(2024);
    int max_clusters_seen_over_n = 0;
    for (int sc = 0; sc < 600; ++sc) {
        const int n = 2 + static_cast<int>(rng() % 40);
        const int mcs = 1 + static_cast<int>(rng() % 3);  // 1, 2, 3
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[rng() % (i + 1)]);

        std::vector<Edge> edges;
        for (int i = 1; i < n; ++i) {
            int p = perm[rng() % i];
            edges.emplace_back(
                p, perm[i], 1.0 + static_cast<double>(rng() % 4));  // Exercise frequent ties.
        }
        std::vector<double> core(n, 0.0);
        for (int i = 0; i < n; ++i) {
            if (rng() % 3) {
                core[i] = 0.5 + static_cast<double>(rng() % 3);
                edges.emplace_back(i, i, core[i]);
            }
        }
        std::stable_sort(edges.begin(), edges.end(), order);

        std::vector<CondRow> rows;
        std::vector<double> lambda_birth;
        std::vector<int> final_cluster;
        int next_cl = 0;
        build_condensed_tree(edges, n, mcs, core, rows, lambda_birth, next_cl, final_cluster);

        const int n_cl = next_cl - n;  // Include the root.
        CHECK(n_cl >= 1);
        CHECK(n_cl <= 2 * n);
        CHECK(static_cast<int>(lambda_birth.size()) == n_cl);
        max_clusters_seen_over_n = std::max(max_clusters_seen_over_n, n_cl - n);

        std::vector<int> point_rows(n, 0);
        for (const auto& r : rows) {
            CHECK(r.parent_cl >= n && r.parent_cl < next_cl);
            CHECK(r.child_cl_size >= 1);
            if (r.child_cl < n) {
                CHECK(r.child_cl >= 0);
                CHECK(r.child_cl_size == 1);
                ++point_rows[r.child_cl];
            } else {
                CHECK(r.child_cl > n && r.child_cl < next_cl);
            }
        }
        // Each point may leave the condensed tree at most once.
        for (int p = 0; p < n; ++p) CHECK(point_rows[p] <= 1);

        std::vector<int> labels;
        std::vector<double> stab;
        extract_clusters(rows, lambda_birth, n, next_cl, final_cluster, labels, stab);
        CHECK(static_cast<int>(labels.size()) == n);
    }
    // Require coverage of the more-than-n-clusters case.
    CHECK(max_clusters_seen_over_n >= 1);
    std::puts("OK");
    return 0;
}
