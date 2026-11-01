#pragma once
#include <vector>

#include "mixdex/utils/uf.hpp"

namespace mixdex {

// Disjoint sets with member lists updated on each successful merge.
struct UFMembers : UF {
    explicit UFMembers(int n);

    int unite(int a, int b) noexcept;
    int size_of(int x) noexcept;
    const std::vector<int>& members_of(int x) noexcept;

private:
    std::vector<std::vector<int>> members_;
};

}  // namespace mixdex
