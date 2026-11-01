#include "mixdex/utils/uf.hpp"

#include <numeric>

namespace mixdex {

UF::UF(int n) : parent_(n), size_(n, 1), n_comp_(n) {
    std::iota(parent_.begin(), parent_.end(), 0);
}

int UF::find_compress(int x) noexcept {
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
