#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mixdex/scale.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <class F>
void expect_message(F f, const char* expected) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find(expected) != std::string::npos);
        return;
    }
    CHECK(false);
}

int main() {
    const std::vector<double> constant{3.0, 3.0};
    expect_message(
        [&] {
            fit_scale_params(constant, NumericType::Continuous);
        },
        "min == max");

    const std::vector<double> close{1.0, std::nextafter(1.0, 2.0)};
    CHECK(close[0] != close[1]);
    std::vector<FeatureType> ft{FeatureType::Numeric};
    std::vector<NumericType> nt{NumericType::Continuous};
    Dataset d(close, {}, 2, 1, 0, ft, nt, {"x"});
    expect_message(
        [&] {
            Scaler::fit(d);
        },
        "численно вырожденный масштаб");
    expect_message(
        [&] {
            scale_feature(close, NumericType::Continuous);
        },
        "численно вырожденный масштаб");

    // Small absolute values alone do not imply degeneracy.
    const std::vector<double> tiny{0.0, 1e-20};
    const auto p = fit_scale_params(tiny, NumericType::Continuous);
    CHECK(p.scale > 0.0 && std::isfinite(p.scale));
    CHECK(p.level == ScaleLevel::Iqr);

    // Fall back to IDR when IQR is zero but IDR is usable.
    std::vector<double> ladder(100, 0.0);
    for (int i = 80; i < 100; ++i) ladder[i] = 1.0;
    CHECK(fit_scale_params(ladder, NumericType::Discrete).level == ScaleLevel::Idr);

    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        const std::vector<double> values{0.0, bad, 2.0};
        expect_message(
            [&] {
                fit_scale_params(values, NumericType::Continuous);
            },
            "finite");
        expect_message(
            [&] {
                scale_feature(values, NumericType::Discrete);
            },
            "finite");
    }
    expect_message(
        [&] {
            fit_scale_params(tiny, static_cast<NumericType>(-1));
        },
        "unknown numeric type");

    // A wrapped m * n_num must fail before an empty buffer can be accessed.
    Dataset two_columns({0.0, 0.0, 1.0, 2.0},
                        {},
                        2,
                        2,
                        0,
                        {FeatureType::Numeric, FeatureType::Numeric},
                        {NumericType::Continuous, NumericType::Continuous},
                        {"a", "b"});
    const auto scaler = Scaler::fit(two_columns);
    CHECK(scaler.transform_rows({}, 0).empty());
    try {
        scaler.transform_rows({}, std::numeric_limits<std::size_t>::max() / 2 + 1);
        CHECK(false);
    } catch (const std::length_error& error) {
        CHECK(std::string(error.what()) == "buffer size overflow");
    }

    std::puts("test_scale_validation: OK");
    return 0;
}
