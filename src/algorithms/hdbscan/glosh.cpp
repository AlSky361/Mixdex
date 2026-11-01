// Method references: docs/references.md#density-clustering
#include "mixdex/algorithms/hdbscan/glosh.hpp"

#include <algorithm>

#include "mixdex/utils/constants.hpp"

using namespace mixdex::constants;

namespace mixdex {

std::vector<double> compute_glosh(const std::vector<double>& core_dist,
                                  const std::vector<CondRow>& rows,
                                  const std::vector<int>& cl_parent,
                                  int n,
                                  int next_cl) {
    int n_cl = next_cl - n;

    std::vector<double> cl_lambda_max(n_cl, 0.0);

    for (const auto& r : rows) {
        int ci = r.parent_cl - n;
        if (ci < 0 || ci >= n_cl)
            continue;
        cl_lambda_max[ci] = std::max(cl_lambda_max[ci], r.lambda);
    }

    for (int ci = n_cl - 1; ci >= 0; --ci) {
        int pai = cl_parent[ci];
        if (pai >= 0)
            cl_lambda_max[pai] = std::max(cl_lambda_max[pai], cl_lambda_max[ci]);
    }

    std::vector<const CondRow*> leaf_rows;
    leaf_rows.reserve(n);
    for (const auto& r : rows) {
        if (r.child_cl < n)
            leaf_rows.push_back(&r);
    }

    std::vector<double> scores(n, 1.0);

    // Keep this cheap O(n) loop sequential; repeated point rows would race in parallel.
    for (const CondRow* rp : leaf_rows) {
        const auto& r = *rp;
        int pt = r.child_cl;
        int ci = r.parent_cl - n;
        if (ci < 0 || ci >= n_cl)
            continue;
        double lam_x = r.lambda;
        double lam_max = cl_lambda_max[ci];
        scores[pt] =
            (lam_max > kEpsZero) ? std::max(0.0, std::min(1.0, 1.0 - lam_x / lam_max)) : 1.0;
    }

    return scores;
}

}  // namespace mixdex
