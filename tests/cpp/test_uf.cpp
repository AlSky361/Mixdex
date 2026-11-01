// Compare read-only roots, compression and flattening; verify concurrent reads.
#include <cstdio>
#include <cstdlib>
#include <random>
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

int main() {
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
