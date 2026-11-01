#pragma once
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "mixdex/utils/aliases.hpp"

namespace mixdex {

// Rollback union-find uses ranks without path compression.
// find_root is read-only and permits concurrent readers.
class RollbackUF {
public:
    explicit RollbackUF(Idx n) : parent_(static_cast<std::size_t>(n)), rank_(parent_.size(), 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    Idx find_root(Idx x) const noexcept {
        while (parent_[static_cast<std::size_t>(x)] != x) x = parent_[static_cast<std::size_t>(x)];
        return x;
    }

    // Return true only when two distinct sets are joined.
    bool unite(Idx a, Idx b) {
        a = find_root(a);
        b = find_root(b);
        if (a == b)
            return false;
        if (rank_[static_cast<std::size_t>(a)] < rank_[static_cast<std::size_t>(b)])
            std::swap(a, b);
        const bool bump = rank_[static_cast<std::size_t>(a)] == rank_[static_cast<std::size_t>(b)];
        log_.push_back({b, a, bump});
        parent_[static_cast<std::size_t>(b)] = a;
        if (bump)
            ++rank_[static_cast<std::size_t>(a)];
        return true;
    }

    // Rollback checkpoint: number of recorded merges.
    std::size_t snapshot() const noexcept {
        return log_.size();
    }

    // Undo merges after the checkpoint.
    void rollback_to(std::size_t snapshot) noexcept {
        while (log_.size() > snapshot) {
            const Op& op = log_.back();
            parent_[static_cast<std::size_t>(op.child)] = op.child;
            if (op.rank_bumped)
                --rank_[static_cast<std::size_t>(op.parent)];
            log_.pop_back();
        }
    }

private:
    struct Op {
        Idx child;
        Idx parent;
        bool rank_bumped;
    };

    std::vector<Idx> parent_;
    std::vector<int> rank_;
    std::vector<Op> log_;
};

}  // namespace mixdex
