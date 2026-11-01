// Match sorted quantiles bit-for-bit across sizes, ties, methods and probability orders.
#include <algorithm>
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

int main() {
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
