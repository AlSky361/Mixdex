// Compare read-only roots, compression and flattening; verify concurrent reads.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "mixdex/utils/parallel.hpp"
#include "mixdex/utils/uf.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

void check_contracts() {
    try {
        UF invalid(-1);
        CHECK(false);
    } catch (const std::invalid_argument&) {
    }
    UF empty(0);
    CHECK(empty.n_comp() == 0);
    empty.flatten();
    UF one(1);
    CHECK(one.find_root(0) == 0 && one.find_compress(0) == 0);
    CHECK(one.same(0, 0) && one.unite(0, 0) == -1 && one.n_comp() == 1);
}

void check_independent_partition() {
    constexpr int n = 24;
    UF uf(n);
    std::vector<int> labels(n);
    std::iota(labels.begin(), labels.end(), 0);
    std::mt19937 random(361);
    std::uniform_int_distribution<int> pick(0, n - 1);
    int components = n;
    for (int step = 0; step < 300; ++step) {
        const int a = pick(random), b = pick(random);
        const int previous = labels[b], replacement = labels[a];
        const bool distinct = previous != replacement;
        const int result = uf.unite(a, b);
        CHECK((result != -1) == distinct);
        if (distinct) {
            std::replace(labels.begin(), labels.end(), previous, replacement);
            --components;
            CHECK(result == uf.find_root(a) && result == uf.find_root(b));
        }
        CHECK(uf.n_comp() == components);
        if (step % 7 == 0)
            uf.flatten();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) CHECK(uf.same(i, j) == (labels[i] == labels[j]));
    }
}

int main() {
    check_contracts();
    check_independent_partition();
    constexpr int n = 5000;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> pick(0, n - 1);

    UF uf(n);
    UF ref(n);  // Reference uses the same merges and compressed roots.
    for (int t = 0; t < 4000; ++t) {
        const int a = pick(rng), b = pick(rng);
        uf.unite(a, b);
        ref.unite(a, b);
    }

    // Read-only roots must match the reference before flattening.
    std::vector<int> expect(n);
    for (int x = 0; x < n; ++x) expect[x] = ref.find_compress(x);
    for (int x = 0; x < n; ++x) CHECK(uf.find_root(x) == expect[x]);

    // Flattening preserves roots and makes subsequent lookup direct.
    uf.flatten();
    for (int x = 0; x < n; ++x) CHECK(uf.find_root(x) == expect[x]);

    // Concurrent readers must observe the same roots.
    std::vector<int> got(n, -1);
    const UF& cuf = uf;
    parallel_for<Schedule::Dynamic>(n, [&](int x) {
        got[x] = cuf.find_root(x);
    });
    for (int x = 0; x < n; ++x) CHECK(got[x] == expect[x]);

    std::puts("test_uf: OK");
    return 0;
}
