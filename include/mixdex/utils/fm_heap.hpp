#pragma once
#include <cstddef>
#include <utility>
#include <vector>

namespace mixdex {

// Keep the nearest capacity entries; the farthest retained entry is the heap root.
class FixedMaxHeap {
public:
    explicit FixedMaxHeap(std::size_t capacity) noexcept;

    void try_insert(double distance, int index);

    std::size_t size() const noexcept;
    bool full() const noexcept;
    double max_distance() const noexcept;

    void reset(std::size_t capacity);

    double farthest() const noexcept;

    std::vector<std::pair<int, double>> sorted() const;

private:
    std::size_t capacity_;
    std::size_t size_;
    std::vector<double> distances_;
    std::vector<int> indices_;

    void sift_down(std::size_t i) noexcept;
    void build_heap() noexcept;
};

}  // namespace mixdex
