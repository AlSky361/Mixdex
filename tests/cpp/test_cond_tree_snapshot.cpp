// Check condensed-tree extraction and full HDBSCAN against fixed snapshots.
// Regenerate with --dump into cond_tree_snapshot.inc (@@COND and @@HDBSCAN).
// Synthetic normal-distribution fixtures are enabled only for libstdc++.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/algorithms/hdbscan/hdbscan.hpp"
#include "mixdex/synthetic/synthetic.hpp"
#include "mixdex/utils/graph.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cond_tree_snapshot.inc"

using namespace mixdex;

namespace {

bool edge_order(const Edge& a, const Edge& b) {
    if (a.w != b.w)
        return a.w > b.w;
    bool a_self = (a.u == a.v), b_self = (b.u == b.v);
    if (a_self != b_self)
        return b_self;
    if (a.u != b.u)
        return a.u < b.u;
    return a.v < b.v;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void append(std::string& out, const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    out += buf;
}

void run_scenario(std::string& out,
                  const char* name,
                  std::vector<Edge> edges,
                  int n,
                  int min_cluster_size,
                  std::vector<double> core) {
    std::sort(edges.begin(), edges.end(), edge_order);
    std::vector<CondRow> rows;
    std::vector<double> lambda_birth;
    std::vector<int> final_cluster;
    int next_cl = 0;
    build_condensed_tree(
        edges, n, min_cluster_size, core, rows, lambda_birth, next_cl, final_cluster);
    std::vector<int> labels;
    std::vector<double> stabilities;
    extract_clusters(rows, lambda_birth, n, next_cl, final_cluster, labels, stabilities);

    append(out, "=== %s ===\n", name);
    append(out, "n_rows=%zu next_cl=%d\n", rows.size(), next_cl);
    for (const auto& r : rows)
        append(out,
               "row parent=%d child=%d lambda=%.10f size=%d\n",
               r.parent_cl,
               r.child_cl,
               r.lambda,
               r.child_cl_size);
    out += "lambda_birth:";
    for (double lb : lambda_birth) append(out, " %.10f", lb);
    out += "\nlabels:";
    for (int l : labels) append(out, " %d", l);
    out += "\nstabilities:";
    for (double s : stabilities) append(out, " %.10f", s);
    out += "\n\n";
}

std::string cond_tree_text() {
    std::string out;
    run_scenario(out,
                 "chain5_fastpath",
                 {{0, 1, 4.0}, {1, 2, 3.0}, {2, 3, 2.0}, {3, 4, 1.0}},
                 5,
                 2,
                 std::vector<double>(5, 0.0));
    run_scenario(
        out,
        "two_triangles_slowpath",
        {{2, 3, 5.0}, {0, 1, 2.0}, {1, 2, 2.0}, {0, 2, 2.0}, {3, 4, 1.0}, {4, 5, 1.0}, {3, 5, 1.0}},
        6,
        2,
        std::vector<double>(6, 0.0));
    run_scenario(out,
                 "self_edges_core",
                 {{0, 1, 3.0}, {1, 2, 2.0}, {2, 3, 1.0}, {0, 0, 0.5}, {3, 3, 0.5}},
                 4,
                 1,
                 {0.5, 0.1, 0.1, 0.5});
    {
        const int n = 40;
        std::mt19937 rng(7);  // mt19937 is portable; distribution sampling is not.
        std::vector<Edge> edges;
        std::vector<int> nodes(n);
        for (int i = 0; i < n; ++i) nodes[i] = i;
        for (int i = n - 1; i > 0; --i)
            std::swap(nodes[i], nodes[rng() % static_cast<unsigned>(i + 1)]);
        for (int i = 1; i < n; ++i) {
            int parent = nodes[rng() % static_cast<unsigned>(i)];
            double w = 0.1 + static_cast<double>(rng() % 100) /
                                 10.0;  // Integer tenths produce exact ties.
            edges.emplace_back(parent, nodes[i], w);
        }
        run_scenario(out, "random_tree_n40", edges, n, 3, std::vector<double>(n, 0.0));
    }
    run_scenario(out, "single_point", {}, 1, 2, {0.0});
    return out;
}

std::string hdbscan_text() {
#if defined(__GLIBCXX__)
    SyntheticParams p;
    p.n = 200;
    p.d_num = 2;
    p.n_clusters = 4;
    p.separation = 8.0;
    p.cluster_std = 0.5;
    p.p_cat = 0.3;
    p.cardinality = 3;
    p.seed = 42;
    // Pin one thread for reproducible snapshot inputs.
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    auto synth = make_synthetic(p);
    auto r = hdbscan(synth.dataset, 5, 5, HDBSCANMode::Exact, SearchMethod::BallTree);
    std::string out = "labels:";
    for (int l : r.labels) append(out, " %d", l);
    out += "\nstabilities:";
    for (double s : r.cluster_stabilities) append(out, " %.6f", s);
    out += "\nscores:";
    for (double s : r.outlier_scores) append(out, " %.6f", s);
    out += "\n";
    return out;
#else
    return std::string(
        kSnapshotHdbscan);  // The scenario is unavailable; compare its placeholder with itself.
#endif
}

void report_diff(const char* what, const std::string& got, const std::string& want) {
    std::size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    std::size_t ls = got.rfind('\n', i == 0 ? 0 : i - 1);
    ls = (ls == std::string::npos) ? 0 : ls + 1;
    std::fprintf(stderr,
                 "SNAPSHOT MISMATCH in %s at byte %zu\n got : %.120s\n want: %.120s\n",
                 what,
                 i,
                 got.c_str() + ls,
                 want.c_str() + std::min(ls, want.size()));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cond = cond_tree_text();
    const std::string hd = hdbscan_text();
    if (argc > 1 && std::strcmp(argv[1], "--dump") == 0) {
        std::printf("@@COND\n%s@@HDBSCAN\n%s@@END\n", cond.c_str(), hd.c_str());
        return 0;
    }
    bool ok = true;
    if (cond != kSnapshotCondTree) {
        report_diff("cond_tree", cond, kSnapshotCondTree);
        ok = false;
    }
    if (hd != kSnapshotHdbscan) {
        report_diff("hdbscan", hd, kSnapshotHdbscan);
        ok = false;
    }
    if (!ok)
        return 1;
    std::puts("OK");
    return 0;
}
