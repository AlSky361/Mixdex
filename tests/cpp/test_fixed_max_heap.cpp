#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mixdex/utils/fm_heap.hpp"

namespace {
thread_local int allocations_before_failure = -1;

void check(bool condition) {
    if (!condition) {
        std::fputs("fixed_max_heap regression failed\n", stderr);
        std::exit(1);
    }
}

template <class Exception, class F>
void expect(F&& action) {
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    check(false);
}

using Entry = std::pair<int, double>;

bool closer(const Entry& a, const Entry& b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
}

void check_selection() {
    std::mt19937 random(361);
    const double inf = std::numeric_limits<double>::infinity();
    for (std::size_t capacity : {0, 1, 2, 7, 20}) {
        mixdex::FixedMaxHeap heap(capacity);
        std::vector<Entry> reference;
        check(heap.size() == 0 && heap.max_distance() == inf && heap.farthest() == 0);
        for (int i = 0; i < 300; ++i) {
            const double distance = random() % 13 == 0 ? inf : static_cast<double>(random() % 11);
            heap.try_insert(distance, i);
            reference.emplace_back(i, distance);
            std::sort(reference.begin(), reference.end(), closer);
            if (reference.size() > capacity)
                reference.resize(capacity);
            check(heap.sorted() == reference);
            check(heap.size() == reference.size());
            check(heap.full() == (reference.size() == capacity));
            check(heap.farthest() == (reference.empty() ? 0 : reference.back().second));
            check(heap.max_distance() ==
                  (reference.empty() || !heap.full() ? inf : reference.back().second));
        }
        for (std::size_t next : {3, 0, 8, 2, 2}) {
            heap.reset(next);
            check(heap.size() == 0 && heap.sorted().empty());
            check(heap.full() == (next == 0));
            check(heap.max_distance() == inf && heap.farthest() == 0);
            heap.try_insert(3, 1);
            check(heap.size() == (next == 0 ? 0u : 1u));
        }
    }
    mixdex::FixedMaxHeap tied(2);
    tied.try_insert(5, 9);
    tied.try_insert(5, 8);
    tied.try_insert(5, 1);
    check(tied.sorted() == std::vector<Entry>{{1, 5}, {8, 5}});
}

void check_allocation_failures() {
    for (int fail_after : {0, 1}) {
        allocations_before_failure = fail_after;
        expect<std::bad_alloc>([] {
            mixdex::FixedMaxHeap heap(4);
        });
        check(allocations_before_failure == -1);

        mixdex::FixedMaxHeap heap(4);
        for (int i = 0; i < 4; ++i) heap.try_insert(i + 2, i);
        const auto before = heap.sorted();
        allocations_before_failure = fail_after;
        expect<std::bad_alloc>([&] {
            heap.reset(64);
        });
        check(allocations_before_failure == -1);
        check(heap.sorted() == before && heap.size() == 4 && heap.full());
        check(heap.max_distance() == 5 && heap.farthest() == 5);
        heap.try_insert(1, 9);
        check(heap.sorted() == std::vector<Entry>{{9, 1}, {0, 2}, {1, 3}, {2, 4}});
        heap.reset(64);
        check(heap.size() == 0 && !heap.full());
    }

    mixdex::FixedMaxHeap heap(2);
    heap.try_insert(1, 0);
    heap.try_insert(2, 1);
    expect<std::length_error>([&] {
        heap.reset(std::numeric_limits<std::size_t>::max());
    });
    check(heap.size() == 2 && heap.full() && heap.max_distance() == 2);
    heap.try_insert(0, 2);
    check(heap.sorted() == std::vector<Entry>{{2, 0}, {0, 1}});
    expect<std::length_error>([] {
        mixdex::FixedMaxHeap oversized(std::numeric_limits<std::size_t>::max());
    });

    mixdex::FixedMaxHeap growing(4);
    growing.reset(5);
    allocations_before_failure = 0;
    growing.reset(6);  // Geometric growth leaves room without another allocation.
    check(allocations_before_failure == 0);
    allocations_before_failure = -1;
    check(growing.size() == 0 && !growing.full());
}
}  // namespace

// Fail either vector allocation without requesting excessive physical memory.
void* operator new(std::size_t bytes) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0)
        --allocations_before_failure;
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes))
        return p;
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

static_assert(!noexcept(mixdex::FixedMaxHeap(1)));

int main() {
    check_selection();
    check_allocation_failures();
    std::puts("test_fixed_max_heap: OK");
    return 0;
}
