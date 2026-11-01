#include "mixdex/algorithms/dbscan/dbscan.hpp"

#include <stdexcept>

#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/parallel.hpp"

namespace mixdex {

std::vector<int> dbscan(const NeighbourSearch& search, double eps, int min_pts) {
    if (!(eps >= 0.0))
        throw std::invalid_argument("dbscan: eps must be >= 0");
    if (min_pts < 1)
        throw std::invalid_argument("dbscan: min_pts must be >= 1");

    const int n = search.n_rows();
    constexpr int kUnvisited = -2;

    std::vector<char> is_core(n, 0);
    parallel_for(n, [&](int i) {
        is_core[i] = search.has_neighbours_within(i, eps, min_pts - 1) ? 1 : 0;
    });

    std::vector<int> labels(n, kUnvisited);
    std::vector<int> queue;
    std::vector<int> neighbours;
    int cluster = 0;

    for (int s = 0; s < n; ++s) {
        if (labels[s] != kUnvisited)
            continue;
        if (!is_core[s]) {
            labels[s] = constants::kNoiseLabel;
            continue;
        }

        labels[s] = cluster;
        queue.clear();
        queue.push_back(s);

        for (std::size_t h = 0; h < queue.size(); ++h) {
            const int p = queue[h];
            search.radius_ids(p, eps, neighbours);
            for (const int q : neighbours) {
                if (labels[q] == kUnvisited || labels[q] == constants::kNoiseLabel) {
                    const bool was_unvisited = (labels[q] == kUnvisited);
                    labels[q] = cluster;
                    if (was_unvisited && is_core[q])
                        queue.push_back(q);
                }
            }
        }
        ++cluster;
    }

    return labels;
}

std::vector<int> dbscan(
    const Dataset& data, double eps, int min_pts, SearchMethod method, int leaf_size) {
    auto search = make_search(data, method, leaf_size);
    return dbscan(*search, eps, min_pts);
}

}  // namespace mixdex
