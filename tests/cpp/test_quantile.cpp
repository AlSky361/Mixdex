// Check definitions, invalid input and agreement of both quantile paths.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "mixdex/quantile.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

static_assert(method_for(NumericType::Discrete) == QuantileMethod::InvertedCDF);
static_assert(method_for(NumericType::Continuous) == QuantileMethod::MedianUnbiased);

static void check_values(const std::vector<double>& sorted,
                         const std::vector<double>& ps,
                         QuantileMethod method,
                         const std::vector<double>& expected,
                         double tolerance = 0.0) {
    const auto result = quantiles(sorted, ps, method);
    auto work = sorted;
    std::reverse(work.begin(), work.end());
    const auto selected = quantiles_select(work, ps, method);
    CHECK(result.size() == expected.size());
    CHECK(selected.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::isfinite(result[i]));
        CHECK(std::fabs(result[i] - expected[i]) <= tolerance);
        CHECK(same_bits(selected[i], result[i]));
    }
}

int main() {
    // Type 1 jumps immediately above a rank, without an epsilon-wide plateau.
    check_values({0.0, 10.0, 20.0},
                 {1.0 / 3.0 - 1e-13, 1.0 / 3.0, 1.0 / 3.0 + 1e-13},
                 QuantileMethod::InvertedCDF,
                 {0.0, 0.0, 10.0});
    check_values({0.0, 6.0, 12.0, 18.0},
                 {0.75, 0.25, 0.5, 0.25},
                 QuantileMethod::MedianUnbiased,
                 {15.5, 2.5, 9.0, 2.5},
                 1e-14);

    const double tiny = std::numeric_limits<double>::denorm_min();
    const double largest = std::numeric_limits<double>::max();
    for (auto method : {QuantileMethod::InvertedCDF, QuantileMethod::MedianUnbiased}) {
        for (double constant : {tiny, -tiny, largest, -largest})
            check_values({constant, constant},
                         {0.0, 0.25, 0.5, 0.75, 1.0},
                         method,
                         std::vector<double>(5, constant));
        check_values({tiny, 2.0 * tiny}, {0.0, 1.0}, method, {tiny, 2.0 * tiny});
        CHECK(quantiles({}, {}, method).empty());
        std::vector<double> empty;
        CHECK(quantiles_select(empty, {}, method).empty());
    }
    check_values({-largest, largest},
                 {0.0, 0.5, 1.0},
                 QuantileMethod::MedianUnbiased,
                 {-largest, 0.0, largest});

    try {
        method_for(static_cast<NumericType>(-1));
        CHECK(false);
    } catch (const std::invalid_argument&) {
    }

    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        for (const auto& ps : {std::vector<double>{}, std::vector<double>{0.5}}) {
            std::vector<double> work{3.0, bad, 1.0};
            const auto before = work;
            try {
                quantiles_select(work, ps, QuantileMethod::MedianUnbiased);
                CHECK(false);
            } catch (const std::invalid_argument&) {
            }
            CHECK(std::memcmp(work.data(), before.data(), work.size() * sizeof(double)) == 0);
        }
    }
    for (double bad : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        std::vector<double> work{3.0, 1.0, 2.0};
        const auto before = work;
        try {
            quantiles_select(work, {0.5, bad}, QuantileMethod::MedianUnbiased);
            CHECK(false);
        } catch (const std::invalid_argument&) {
        }
        CHECK(work == before);
    }

    std::mt19937 rng(5);
    const std::vector<double> ps = {
        0.25, 0.5, 0.75, 0.0, 1.0, 1e-12, 1.0 - 1e-12, 0.9, 0.1, 0.5, 0.333, 0.6667};
    for (QuantileMethod method : {QuantileMethod::InvertedCDF, QuantileMethod::MedianUnbiased}) {
        for (int n = 1; n <= 80; ++n) {
            for (int kind = 0; kind < 3; ++kind) {
                std::vector<double> x(static_cast<std::size_t>(n));
                std::uniform_real_distribution<double> real(-5.0, 5.0);
                std::uniform_int_distribution<int> few(0, 3);
                for (double& v : x) v = kind == 0 ? real(rng) : (kind == 1 ? few(rng) : 7.0);

                std::vector<double> sorted = x;
                std::sort(sorted.begin(), sorted.end());
                const auto expect = quantiles(sorted, ps, method);

                std::vector<double> work = x;
                const auto got = quantiles_select(work, ps, method);
                CHECK(got.size() == expect.size());
                for (std::size_t i = 0; i < ps.size(); ++i) CHECK(same_bits(got[i], expect[i]));
            }
        }
    }

    bool threw = false;
    try {
        std::vector<double> empty;
        quantiles_select(empty, {0.5}, QuantileMethod::MedianUnbiased);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    for (double p : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(),
                     -0.1,
                     1.1}) {
        std::vector<double> x{1, 2, 3};
        for (bool select : {false, true}) {
            bool rejected = false;
            try {
                if (select)
                    quantiles_select(x, {p}, QuantileMethod::MedianUnbiased);
                else
                    quantiles(x, {p}, QuantileMethod::MedianUnbiased);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            CHECK(rejected);
        }
    }

    for (bool select : {false, true}) {
        std::vector<double> x{1, 2, 3};
        bool rejected = false;
        try {
            if (select)
                quantiles_select(x, {0.5}, static_cast<QuantileMethod>(-1));
            else
                quantiles(x, {0.5}, static_cast<QuantileMethod>(-1));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK(rejected);
    }

    std::puts("test_quantile: OK");
    return 0;
}
