#pragma once
#include <vector>

#include "mixdex/utils/graph.hpp"

namespace mixdex {

/// Minimum spanning forest; exact weight ties use ordered endpoint IDs.
std::vector<Edge> boruvka(const std::vector<Edge>& edges, int n);

std::vector<Edge> boruvka(const Graph& g);

}  // namespace mixdex
