#pragma once
#include <vector>

#include "mixdex/utils/graph.hpp"

namespace mixdex {

// Point IDs are below n; cluster IDs start at n. Lambda is inverse edge weight.
struct CondRow {
    int parent_cl;
    int child_cl;
    double lambda;
    int child_cl_size;  // One for a point; cluster size otherwise.
};

/// Prune branches below min_cluster_size while recording point departures.
void build_condensed_tree(const std::vector<Edge>& reversed_MSText,
                          int n,
                          int min_cluster_size,
                          const std::vector<double>& core,
                          std::vector<CondRow>& rows,
                          std::vector<double>& lambda_birth,
                          int& next_cl,
                          std::vector<int>& final_cluster);

/// Select clusters by excess of mass; labels -1 denote noise.
void extract_clusters(const std::vector<CondRow>& rows,
                      const std::vector<double>& lambda_birth,
                      int n,
                      int next_cl,
                      const std::vector<int>& final_cluster,
                      std::vector<int>& out_labels,
                      std::vector<double>& out_stabilities);

}  // namespace mixdex
