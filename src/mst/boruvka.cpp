#include "mixdex/mst/boruvka.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mixdex/utils/checked_arithmetic.hpp"
#include <limits>
#include <queue>
#include <tuple>

#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/parallel.hpp"
#include "mixdex/utils/uf.hpp"

using namespace mixdex::constants;

namespace mixdex {

namespace {

// Order candidates by exact (weight, min(u,v), max(u,v)) for reproducible ties.
// Tolerant equality is not transitive and can prevent replacing an infinite sentinel.
bool better_candidate(const Edge& incumbent, const Edge& candidate) {
    if (candidate.w != incumbent.w)
        return candidate.w < incumbent.w;
    int lo = std::min(candidate.u, candidate.v), hi = std::max(candidate.u, candidate.v);
    int inc_lo = std::min(incumbent.u, incumbent.v), inc_hi = std::max(incumbent.u, incumbent.v);
    return std::tie(lo, hi) < std::tie(inc_lo, inc_hi);
}

}  // namespace

std::vector<Edge> boruvka(const std::vector<Edge>& edges, int n) {
    if (n < 0)
        throw std::invalid_argument("vertex count must be nonnegative");
    const Idx m = detail::checked_index_size(edges.size());
    for (const auto& e : edges) {
        if (e.u < 0 || e.u >= n || e.v < 0 || e.v >= n)
            throw std::invalid_argument("edge vertex out of range");
        if (std::isnan(e.w))
            throw std::invalid_argument("edge weight must not be NaN");
    }
    if (n == 0)
        return {};

    UF uf(n);
    std::vector<Edge> mst;
    mst.reserve(n - 1);

    const int T = std::max(1, MIXDEX_OMP_MAX_THREADS());
    std::vector<std::vector<Edge>> local_best(T, std::vector<Edge>(n, Edge{-1, -1, kInf}));
    const Edge none{-1, -1, kInf};

    while (static_cast<int>(mst.size()) < n - 1) {
        std::vector<Edge> best(n, none);
        for (auto& lb : local_best) std::fill(lb.begin(), lb.end(), none);

        parallel_for<Schedule::Static>(T, [&](int t) {
            std::vector<Edge>& lb = local_best[t];
            const int lo = chunk_begin(m, T, t);
            const int hi = chunk_begin(m, T, t + 1);
            for (int i = lo; i < hi; ++i) {
                const Edge& e = edges[i];
                int comp_u = uf.find_root(e.u);
                int comp_v = uf.find_root(e.v);
                if (comp_u == comp_v)
                    continue;
                if (better_candidate(lb[comp_u], e))
                    lb[comp_u] = e;
                if (better_candidate(lb[comp_v], e))
                    lb[comp_v] = e;
            }
        });

        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < n; ++c) {
                const Edge& e = local_best[t][c];
                if (e.u == -1)
                    continue;
                if (better_candidate(best[c], e))
                    best[c] = e;
            }
        }

        bool any = false;

        for (int c = 0; c < n; ++c) {
            const Edge& e = best[c];
            if (e.u == -1)
                continue;

            if (uf.unite(e.u, e.v) != -1) {
                mst.push_back(e);
                any = true;
                if (static_cast<int>(mst.size()) == n - 1)
                    break;
            }
        }

        uf.flatten();

        if (!any)
            break;
    }

    return mst;
}

std::vector<Edge> boruvka(const Graph& g) {
    return boruvka(g.get_edges(), g.n());
}

}  // namespace mixdex
