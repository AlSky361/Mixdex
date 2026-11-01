// Reject exact duplicate coordinates, including signed zeros, in any row order.
// Nearby rows and numeric matches with different categories remain distinct.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mixdex/types.hpp"

using namespace mixdex;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

Dataset make(std::vector<double> num,
             std::vector<Code> cat,
             std::size_t n,
             std::size_t n_num,
             std::size_t n_cat) {
    std::vector<FeatureType> ft(n_num, FeatureType::Numeric);
    for (std::size_t i = 0; i < n_cat; ++i) ft.push_back(FeatureType::Categorical);
    std::vector<NumericType> nt(n_num, NumericType::Continuous);
    std::vector<std::string> names;
    for (std::size_t i = 0; i < n_num + n_cat; ++i) names.push_back("f" + std::to_string(i));
    return Dataset(std::move(num),
                   std::move(cat),
                   n,
                   n_num,
                   n_cat,
                   std::move(ft),
                   std::move(nt),
                   std::move(names));
}

bool throws(const std::function<void()>& f) {
    try {
        f();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

}  // namespace

int main() {
    // Exact numeric duplicate.
    CHECK(throws([] {
        make({1.0, 2.0, 1.0, 2.0}, {}, 2, 2, 0);
    }));

    // Nearby rows are distinct; keep the second feature nonconstant.
    CHECK(!throws([] {
        make({1000.0, 2000.0, 1000.0 + 1000.0 * 1e-12, 2001.0}, {}, 2, 2, 0);
    }));
    CHECK(!throws([] {
        make({1.0, 2.0, std::nextafter(1.0, 2.0), 2.1}, {}, 2, 2, 0);
    }));

    // Signed zeros are equal and must hash identically.
    CHECK(throws([] {
        make({0.0, 5.0, -0.0, 5.0}, {}, 2, 2, 0);
    }));

    // Distinct rows must remain distinct.
    CHECK(!throws([] {
        make({1.0, 2.0, 1.5, 2.5}, {}, 2, 2, 0);
    }));

    // Exact category codes determine equality; keep the control feature nonconstant.
    CHECK(throws([] {
        make({}, {5, 7, 5, 7}, 2, 0, 2);
    }));
    CHECK(!throws([] {
        make({}, {5, 7, 6, 8}, 2, 0, 2);
    }));

    // Matching numeric values with different categories are distinct.
    // The third row keeps the numeric feature nonconstant.
    CHECK(!throws([] {
        make({1.0, 9.0, 1.0, 9.5, 2.0, 9.0}, {3, 4, 3}, 3, 2, 1);
    }));
    // Matching numeric and categorical values are duplicates.
    CHECK(throws([] {
        make({1.0, 9.0, 1.0, 9.0, 2.0, 9.5}, {3, 3, 3}, 3, 2, 1);
    }));

    // Detect duplicates separated in original row order.
    CHECK(throws([] {
        make({5.0, 1.0, 5.0}, {}, 3, 1, 0);
    }));

    // Three distinct rows must not trigger duplicate detection.
    CHECK(!throws([] {
        make({1.0, 2.0, 3.0}, {}, 3, 1, 0);
    }));

    // A single matching coordinate does not make duplicate rows.
    // The third row prevents constant-column rejection.
    CHECK(!throws([] {
        make({1.0, 2.0, 1.0, 3.0, 5.0, 6.0}, {}, 3, 2, 0);
    }));
    CHECK(!throws([] {
        make({1.0, 2.0, 7.0, 2.0, 5.0, 6.0}, {}, 3, 2, 0);
    }));

    // Find a duplicate at the end of a large unique input.
    CHECK(throws([] {
        std::vector<double> v;
        for (int i = 0; i < 5000; ++i) {
            v.push_back(i * 0.5);
            v.push_back(i % 7);
        }
        v.push_back(1234 * 0.5);
        v.push_back(1234 % 7);
        make(std::move(v), {}, 5001, 2, 0);
    }));

    std::puts("OK");
    return 0;
}
