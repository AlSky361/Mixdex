#pragma once
#include <cassert>
#include <cstddef>
#include <vector>

namespace mixdex {

// Disjoint sets: union by size; find_compress mutates paths, find_root only reads.
// Indices must be in [0, n); Debug asserts this precondition.
struct UF {
    explicit UF(int n);  // Negative sizes throw invalid_argument; zero is valid.

    int find_compress(int x) noexcept;

    int find_root(int x) const noexcept;
    void flatten() noexcept;
    int unite(int a, int b) noexcept;
    bool same(int a, int b) noexcept {
        return find_compress(a) == find_compress(b);
    }
    int n_comp() const noexcept {
        return n_comp_;
    }

protected:
    void assert_index([[maybe_unused]] int x) const noexcept {
        assert(x >= 0 && static_cast<std::size_t>(x) < parent_.size());
    }

    std::vector<int> parent_;
    std::vector<int> size_;
    int n_comp_{0};
};

}  // namespace mixdex
