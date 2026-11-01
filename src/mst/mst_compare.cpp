#include "mixdex/mst/mst_compare.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "mixdex/utils/constants.hpp"

namespace mixdex {

namespace {
inline std::pair<int, int> canon(const Edge& e) noexcept {
    return e.u < e.v ? std::pair{e.u, e.v} : std::pair{e.v, e.u};
}

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(p.first) << constants::kPairHashShift) ^
               static_cast<std::size_t>(p.second);
    }
};
}  // namespace

double mst_edge_recall(const std::vector<Edge>& candidate, const std::vector<Edge>& reference) {
    if (reference.empty())
        return 0.0;

    std::unordered_set<std::pair<int, int>, PairHash> ref_set;
    ref_set.reserve(reference.size());
    for (const auto& e : reference) ref_set.insert(canon(e));

    long long matched = 0;
    for (const auto& e : candidate) {
        if (ref_set.count(canon(e)))
            ++matched;
    }

    return static_cast<double>(matched) / static_cast<double>(reference.size());
}

}  // namespace mixdex
