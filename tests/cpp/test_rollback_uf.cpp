// Rollback restores checkpoint state; compare merge decisions with reference UF.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "mixdex/utils/rollback_uf.hpp"
#include "mixdex/utils/uf.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static void check_same_partition(const RollbackUF& uf, UF& ref, int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int t = 0; t < 1500; ++t) {
        const int x = pick(rng), y = pick(rng);
        CHECK((uf.find_root(x) == uf.find_root(y)) ==
              (ref.find_compress(x) == ref.find_compress(y)));
    }
}

int main() {
    constexpr int n = 300;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> pick(0, n - 1);

    for (int trial = 0; trial < 40; ++trial) {
        RollbackUF uf(n);
        std::vector<std::pair<int, int>> ops;
        struct Snap {
            std::size_t log_size;
            std::size_t ops_len;
        };
        std::vector<Snap> snaps;

        for (int t = 0; t < 250; ++t) {
            if (t % 25 == 0)
                snaps.push_back({uf.snapshot(), ops.size()});
            const int a = pick(rng), b = pick(rng);
            UF probe(n);
            for (auto [x, y] : ops) probe.unite(x, y);
            const bool merged = uf.unite(a, b);
            CHECK(merged == (probe.unite(a, b) != -1));
            ops.push_back({a, b});
        }

        // Roll back to a random checkpoint, then continue merging.
        const Snap s = snaps[static_cast<std::size_t>(pick(rng)) % snaps.size()];
        uf.rollback_to(s.log_size);
        ops.resize(s.ops_len);
        CHECK(uf.snapshot() == s.log_size);

        UF ref(n);
        for (auto [x, y] : ops) ref.unite(x, y);
        check_same_partition(uf, ref, n, rng);

        for (int t = 0; t < 60; ++t) {
            const int a = pick(rng), b = pick(rng);
            const bool merged = uf.unite(a, b);
            CHECK(merged == (ref.unite(a, b) != -1));
        }
        check_same_partition(uf, ref, n, rng);

        uf.rollback_to(0);
        UF fresh(n);
        check_same_partition(uf, fresh, n, rng);
    }
    std::puts("test_rollback_uf: OK");
    return 0;
}
