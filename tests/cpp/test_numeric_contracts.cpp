#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mixdex/utils/checked_arithmetic.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/parallel.hpp"
#include "mixdex/mst/boruvka.hpp"
#include "../../src/algorithms/hdbscan/cond_tree_internal.hpp"

using namespace mixdex;

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("numeric contract regression");
}

template<class E, class F>
void rejects(F f) {
    try { f(); } catch (const E&) { return; }
    throw std::runtime_error("expected rejection");
}

int main() {
    static_assert(sizeof(Idx) == 4 && sizeof(Code) == 4 && sizeof(Count) == 8);
    static_assert(std::is_signed_v<Idx> && std::is_unsigned_v<Count>);
    static_assert(kInvalidIdx == -1);
    const double inf = constants::kInf;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(constants::f_equal(inf, inf));
    check(constants::f_equal(-inf, -inf));
    check(!constants::f_equal(inf, -inf));
    check(!constants::f_equal(inf, 1));
    check(!constants::f_equal(1, -inf));
    check(!constants::f_equal(nan, nan));
    check(!constants::f_equal(nan, 1));
    check(!constants::f_equal(1, nan));
    check(constants::f_equal(0, -0.0));
    check(constants::f_equal(1, 1 + 1e-10));
    check(!constants::f_equal(1, 2));

    const auto imax = std::numeric_limits<Idx>::max();
    check(detail::checked_index_size(imax) == imax);
    rejects<std::length_error>([&] {
        detail::checked_index_size(static_cast<std::size_t>(imax) + 1);
    });
    const auto smax = std::numeric_limits<std::size_t>::max();
    check(detail::checked_size_product(smax, 1) == smax);
    check(detail::checked_size_product(smax, 0) == 0);
    rejects<std::length_error>([&] { detail::checked_size_product(smax, 2); });

    const Count cmax = std::numeric_limits<Count>::max();
    Count c = cmax - 1;
    detail::add_count(c);
    check(c == cmax);
    rejects<std::overflow_error>([&] { detail::add_count(c); });
    check(c == cmax);
    std::atomic<Count> atomic{cmax - 1};
    detail::increment_count(atomic);
    rejects<std::overflow_error>([&] { detail::increment_count(atomic); });
    check(atomic.load() == cmax);

    check(chunk_begin(imax, imax, imax) == imax);
    check(chunk_begin(10, 3, 1) == 3);
    rejects<std::invalid_argument>([] { chunk_begin(-1, 1, 0); });
    rejects<std::invalid_argument>([] { chunk_begin(1, 0, 0); });
    rejects<std::invalid_argument>([] { chunk_begin(1, 2, -1); });
    rejects<std::invalid_argument>([] { chunk_begin(1, 2, 3); });

    int next = imax - 1;
    check(detail::allocate_cluster_id(next) == imax - 1);
    rejects<std::length_error>([&] { detail::allocate_cluster_id(next); });
    check(next == imax);
    std::vector<CondRow> rows;
    std::vector<double> birth;
    std::vector<int> final;
    rejects<std::length_error>([&] {
        build_condensed_tree({}, imax, 1, {}, rows, birth, next, final);
    });
    rejects<std::invalid_argument>([&] {
        build_condensed_tree({}, -1, 1, {}, rows, birth, next, final);
    });
    // Infinite weights form a group and must not stall equal-weight scanning.
    build_condensed_tree({Edge{0, 1, inf}}, 2, 1, {0, 0}, rows, birth, next, final);
    check(final.size() == 2);
    rejects<std::invalid_argument>([] { boruvka({}, -1); });
    rejects<std::invalid_argument>([] { boruvka({Edge{-1, 0, 1}}, 2); });
    rejects<std::invalid_argument>([&] { boruvka({Edge{0, 1, nan}}, 2); });
    // A generic MST still accepts negative weights.
    check(boruvka({Edge{0, 1, -2}}, 2).size() == 1);
}
