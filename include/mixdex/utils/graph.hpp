#pragma once
#include <vector>

namespace mixdex {

struct Edge {
    int u, v;
    double w;

    Edge() = default;
    Edge(int u_, int v_, double w_) : u(u_), v(v_), w(w_) {
    }
};

inline bool operator<(const Edge& a, const Edge& b) {
    return a.w < b.w;
}

/// Undirected weighted graph; adjacency stores IDs into the shared edge array.
class Graph {
public:
    Graph() = default;

    explicit Graph(int n) : Graph(n, 0) {
    }

    Graph(int n, int expected_degree);

    int n() const noexcept;
    void add_edge(int u, int v, double w);
    void remove_edge(int u, int v);
    bool is_connected() const;
    const std::vector<Edge>& get_edges() const noexcept;
    const std::vector<int>& incident_edges(int u) const;
    std::vector<int> compute_components() const;

    std::vector<int> compute_components_subset(const std::vector<int>& subset) const;
    bool edge_removed(int id) const noexcept {
        return removed_[id] != 0;
    }

private:
    int n_{0};
    std::vector<Edge> edges;
    // Only condensed-tree code deletes edges; keep deletion flags out of shared Edge records.
    std::vector<char> removed_;
    std::vector<std::vector<int>> adj_;
};

}  // namespace mixdex
