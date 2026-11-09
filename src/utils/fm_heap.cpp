#include "mixdex/utils/fm_heap.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

#include "mixdex/utils/constants.hpp"

using namespace mixdex::constants;

namespace mixdex {

namespace {
inline bool less_key(double d1, int i1, double d2, int i2) noexcept {
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

template <class T>
void reserve_growth(std::vector<T>& buffer, std::size_t required) {
    if (required <= buffer.capacity())
        return;
    const auto limit = buffer.max_size();
    const auto grown = buffer.capacity() <= limit / 2 ? 2 * buffer.capacity() : limit;
    buffer.reserve(std::max(required, grown));
}
}  // namespace

FixedMaxHeap::FixedMaxHeap(std::size_t capacity) :
    capacity_(capacity), size_(0), distances_(capacity), indices_(capacity) {
}

void FixedMaxHeap::sift_down(std::size_t i) noexcept {
    while (true) {
        std::size_t left = 2 * i + 1;
        if (left >= size_)
            break;

        std::size_t right = left + 1;
        std::size_t largest = i;

        if (less_key(distances_[largest], indices_[largest], distances_[left], indices_[left]))
            largest = left;
        if (right < size_ &&
            less_key(distances_[largest], indices_[largest], distances_[right], indices_[right]))
            largest = right;

        if (largest == i)
            break;

        std::swap(distances_[i], distances_[largest]);
        std::swap(indices_[i], indices_[largest]);
        i = largest;
    }
}

void FixedMaxHeap::try_insert(double distance, int index) {
    assert(distance >= 0.0 && index >= 0);
    if (capacity_ == 0)
        return;

    if (size_ < capacity_) {
        distances_[size_] = distance;
        indices_[size_] = index;
        ++size_;

        if (size_ == capacity_)
            build_heap();
        return;
    }

    if (!less_key(distance, index, distances_[0], indices_[0]))
        return;

    distances_[0] = distance;
    indices_[0] = index;
    sift_down(0);
}

std::size_t FixedMaxHeap::size() const noexcept {
    return size_;
}

bool FixedMaxHeap::full() const noexcept {
    return size_ == capacity_;
}

double FixedMaxHeap::max_distance() const noexcept {
    return size_ > 0 && full() ? distances_[0] : kInf;
}

void FixedMaxHeap::reset(std::size_t capacity) {
    // Reserve both buffers before resizing; scalar initialization cannot throw afterwards.
    reserve_growth(distances_, capacity);
    reserve_growth(indices_, capacity);
    distances_.resize(capacity);
    indices_.resize(capacity);
    capacity_ = capacity;
    size_ = 0;
}

double FixedMaxHeap::farthest() const noexcept {
    if (size_ == 0)
        return 0.0;
    if (size_ == capacity_)
        return distances_[0];
    return *std::max_element(distances_.begin(),
                             distances_.begin() + static_cast<std::ptrdiff_t>(size_));
}

void FixedMaxHeap::build_heap() noexcept {
    if (size_ == 0)
        return;

    for (std::size_t i = size_ / 2; i > 0; --i) {
        sift_down(i - 1);
    }
}

std::vector<std::pair<int, double>> FixedMaxHeap::sorted() const {
    std::vector<std::size_t> order(size_);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return less_key(distances_[a], indices_[a], distances_[b], indices_[b]);
    });

    std::vector<std::pair<int, double>> out;
    out.reserve(size_);

    for (std::size_t idx : order) out.emplace_back(indices_[idx], distances_[idx]);

    return out;
}

}  // namespace mixdex
