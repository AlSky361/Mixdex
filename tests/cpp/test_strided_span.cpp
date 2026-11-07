#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mixdex/utils/strided_span.hpp"

namespace {
void check(bool condition) {
    if (!condition)
        throw std::runtime_error("strided span regression");
}

template <class F>
void rejects(F operation) {
    try {
        operation();
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error("expected out_of_range");
}

struct ThrowingAssignment {
    ThrowingAssignment& operator=(const ThrowingAssignment&) {
        throw std::runtime_error("assignment failed");
    }
};
}  // namespace

static_assert(std::forward_iterator<mixdex::StridedIterator<int>>);
static_assert(std::ranges::forward_range<mixdex::StridedSpan<int>>);
static_assert(!std::contiguous_iterator<mixdex::StridedIterator<int>>);
static_assert(noexcept(std::declval<mixdex::StridedSpan<int>>().copy_to(nullptr)));
static_assert(!noexcept(std::declval<mixdex::StridedSpan<ThrowingAssignment>>().copy_to(nullptr)));

int main() {
    const int rows[]{1, 2, 3, 4, 5, 6};
    mixdex::StridedSpan<int> col(rows + 1, 3, 2);
    check(col.at(0) == 2 && col.at(1) == 5);
    check(!col.is_contiguous());
    rejects([&] {
        col.at(2);
    });
    check(col.to_vector() == std::vector<int>({2, 5}));
    check(std::distance(col.begin(), col.end()) == 2);
    std::vector<int> found;
    std::ranges::copy(col, std::back_inserter(found));
    check(found == std::vector<int>({2, 5}));
    auto it = col.begin();
    auto old = it++;
    check(*old == 2 && *it == 5 && *old.operator->() == 2);
    auto other = col.begin();
    other.stride = 1;
    check(other != col.begin());

    check(col.subspan(1, 1).at(0) == 5);
    auto tail = col.subspan(col.size(), 0);
    check(tail.empty() && tail.data() == col.data() && tail.begin() == tail.end());
    check(tail.is_contiguous());
    rejects([&] {
        col.subspan(3, 0);
    });
    rejects([&] {
        col.subspan(1, 2);
    });
    rejects([&] {
        col.subspan(1, std::numeric_limits<std::size_t>::max());
    });

    std::array<int, 4> dest{};
    col.copy_to(std::span<int>(dest), 2);
    check(dest == std::array<int, 4>{2, 0, 5, 0});
    rejects([&] {
        col.copy_to(std::span<int>(dest).first(2), 2);
    });
    rejects([&] {
        col.copy_to(std::span<int>(dest), 0);
    });
    rejects([&] {
        col.copy_to(std::span<int>(dest), std::numeric_limits<std::size_t>::max());
    });
    rejects([&] {
        col.copy_to(std::span<int>{});
    });
    col.copy_to(dest.data());
    check(dest[0] == 2 && dest[1] == 5);

    mixdex::StridedSpan<int> contiguous(rows, 6);
    check(contiguous.is_contiguous());
    rejects([&] {
        contiguous.copy_to(std::span<int>(dest));
    });
    std::array<int, 6> full{};
    contiguous.copy_to(std::span<int>(full));
    check(std::ranges::equal(full, rows));
    mixdex::StridedSpan<int> empty;
    check(empty.begin() == empty.end() && empty.to_vector().empty());
    check(empty.subspan(0, 0).empty());
    empty.copy_to(nullptr);
    empty.copy_to(std::span<int>{});
    rejects([&] {
        empty.at(0);
    });
    rejects([&] {
        empty.subspan(1, 0);
    });
    ThrowingAssignment source, target;
    mixdex::StridedSpan<ThrowingAssignment> throwing(&source, 1);
    bool propagated = false;
    try {
        throwing.copy_to(&target);
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    check(propagated);
    return EXIT_SUCCESS;
}
