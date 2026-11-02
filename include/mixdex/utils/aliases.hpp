#pragma once
#include <cstdint>

namespace mixdex {

// Compact signed indices; buffer sizes remain std::size_t.
using Idx = std::int32_t;    // Row / feature / node / component index.
using Count = std::uint64_t; // Nonnegative statistics and pair counts.
using Code = std::int32_t;   // Category code; prepared codes are nonnegative.

static_assert(sizeof(int) == sizeof(Idx), "Mixdex requires 32-bit int interfaces");

inline constexpr Idx kInvalidIdx = -1; // Absence only where the API permits it.

}  // namespace mixdex
