#include "mixdex/utils/uf.hpp"

#include <numeric>
#include <stdexcept>

namespace mixdex {
namespace {
std::size_t checked_size(int n) {
    if (n < 0)
        throw std::invalid_argument("UF: size must be nonnegative");
    return static_cast<std::size_t>(n);
}
}  // namespace

UF::UF(int n) : parent_(checked_size(n)), size_(parent_.size(), 1), n_comp_(n) {
    std::iota(parent_.begin(), parent_.end(), 0);
}

int UF::find_compress(int x) noexcept {
    assert_index(x);
    int p = parent_[x];
    while (p != x) {
        int gp = parent_[p];
        parent_[x] = gp;
        x = p;
        p = gp;
    }

    return x;
}

int UF::find_root(int x) const noexcept {
    assert_index(x);
    while (parent_[x] != x) x = parent_[x];
    return x;
}

void UF::flatten() noexcept {
    for (int x = 0, n = static_cast<int>(parent_.size()); x < n; ++x) parent_[x] = find_compress(x);
}

int UF::unite(int a, int b) noexcept {
    a = find_compress(a);
    b = find_compress(b);

    if (a == b)
        return -1;

    if (size_[a] < size_[b])
        std::swap(a, b);

    parent_[b] = a;
    size_[a] += size_[b];
    --n_comp_;
    return a;
}
}  // namespace mixdex
