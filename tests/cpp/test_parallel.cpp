// Check parallel_for coverage, index types, schedules and exception propagation.
// Enable with MIXDEX_BUILD_TESTS and run with ctest.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mixdex/utils/parallel.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <Schedule S>
void check_schedule() {
    std::vector<int> v(10000, 0);
    parallel_for<S>(static_cast<int>(v.size()), [&](int i) {
        v[i] = i + 1;
    });
    for (int i = 0; i < static_cast<int>(v.size()); ++i) CHECK(v[i] == i + 1);

    std::vector<char> seen(5000, 0);
    parallel_for<S>(static_cast<std::size_t>(seen.size()), [&](std::size_t i) {
        seen[i] = 1;
    });
    for (char c : seen) CHECK(c == 1);

    std::vector<char> small_seen(200, 0);
    parallel_for<S>(std::uint8_t{200}, [&](auto i) {
        small_seen[static_cast<std::size_t>(i)] = 1;
    });
    for (char c : small_seen) CHECK(c == 1);

    std::vector<char> medium_seen(40000, 0);
    parallel_for<S>(std::uint16_t{40000}, [&](auto i) {
        medium_seen[static_cast<std::size_t>(i)] = 1;
    });
    for (char c : medium_seen) CHECK(c == 1);

    bool oversized = false;
    try {
        parallel_for<S>(std::numeric_limits<unsigned long long>::max(), [](auto) {
            CHECK(false);
        });
    } catch (const std::length_error&) {
        oversized = true;
    }
    CHECK(oversized);

    bool caught = false;
    try {
        parallel_for<S>(static_cast<long long>(10000), [&](long long i) {
            if (i == 4321)
                throw std::runtime_error("boom");
        });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "boom";
    }
    CHECK(caught);
}

int main() {
    check_schedule<Schedule::Static>();
    check_schedule<Schedule::Dynamic>();
    check_schedule<Schedule::Guided>();

    // Empty ranges do no work; negative counts are invalid.
    bool called = false;
    parallel_for(0, [&](int) {
        called = true;
    });
    bool negative = false;
    try {
        parallel_for(-5, [&](int) { called = true; });
    } catch (const std::invalid_argument&) {
        negative = true;
    }
    CHECK(negative);
    CHECK(!called);

    // Propagate other exception types and stop unnecessary work.
    bool bad = false;
    try {
        parallel_for(1000, [&](int i) {
            if (i % 7 == 3)
                throw std::bad_alloc();
        });
    } catch (const std::bad_alloc&) {
        bad = true;
    }
    CHECK(bad);

    std::puts("test_parallel: OK");
    return 0;
}
