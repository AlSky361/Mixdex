#include "mixdex/utils/uf_members.hpp"

namespace mixdex {

UFMembers::UFMembers(int n) : UF(n), members_(n) {
    for (int i = 0; i < n; ++i) {
        members_[i] = {i};
    }
}

int UFMembers::unite(int a, int b) noexcept {
    a = find_compress(a);
    b = find_compress(b);

    if (a == b)
        return a;

    if (size_[a] < size_[b])
        std::swap(a, b);

    parent_[b] = a;

    members_[a].reserve(size_[a] + size_[b]);
    members_[a].insert(members_[a].end(),
                       std::make_move_iterator(members_[b].begin()),
                       std::make_move_iterator(members_[b].end()));
    members_[b].clear();

    size_[a] += size_[b];
    --n_comp_;

    return a;
}

int UFMembers::size_of(int x) noexcept {
    return size_[find_compress(x)];
}

const std::vector<int>& UFMembers::members_of(int x) noexcept {
    return members_[find_compress(x)];
}

}  // namespace mixdex
