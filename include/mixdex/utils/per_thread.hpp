#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

#include "mixdex/utils/aliases.hpp"
#include "mixdex/utils/constants.hpp"
#include "mixdex/utils/omp_macros.hpp"

namespace mixdex {

// One aligned slot per worker; thread IDs must fit the allocated slot count.
// Read and combine worker results after the parallel region.
template <class T>
class PerThread {
public:
    explicit PerThread(int threads) : slots_(static_cast<std::size_t>(std::max(threads, 1))) {
    }

    int threads() const noexcept {
        return static_cast<int>(slots_.size());
    }

    T& local() noexcept {
        return slots_[static_cast<std::size_t>(MIXDEX_OMP_THREAD_NUM())].value;
    }

    const T& slot(int t) const noexcept {
        return slots_[static_cast<std::size_t>(t)].value;
    }

private:
    struct alignas(constants::kCacheLineBytes) Slot {
        T value{};
    };
    std::vector<Slot> slots_;
};

}  // namespace mixdex
