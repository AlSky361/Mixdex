#pragma once

namespace mixdex {

// Internal indices/codes use int; flat-buffer sizes use size_t; counters use long long.
using Idx = int;          // Row / feature / node / component index; distinct from class Index.
using Count = long long;  // Statistics, pair counts and call counts.
using Code = int;         // Category code.

}  // namespace mixdex
