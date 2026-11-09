#pragma once
#include <cstddef>
#include <utility>
#include <vector>

namespace mixdex {

// Keep the nearest capacity entries; build the max-heap once capacity is reached.
class FixedMaxHeap {
public:
    explicit FixedMaxHeap(std::size_t capacity);

    // Requires distance >= 0 (infinity allowed) and index >= 0; Debug asserts both.
    void try_insert(double distance, int index);

    std::size_t size() const noexcept;
    bool full() const noexcept;
    double max_distance() const noexcept;  // Pruning cutoff; infinity until full or when empty.

    // Clear entries; allocation failure preserves the retained candidates.
    void reset(std::size_t capacity);

    double farthest() const noexcept;  // Actual farthest retained distance; zero when empty.

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
