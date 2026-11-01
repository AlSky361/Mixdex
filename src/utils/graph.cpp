#include "mixdex/utils/graph.hpp"

#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mixdex {

Graph::Graph(int n, int expected_degree) : n_(n) {
    if (n < 0 || expected_degree < 0)
        throw std::invalid_argument("graph sizes must be nonnegative");
    const auto size = static_cast<std::size_t>(n);
    const auto degree = static_cast<std::size_t>(expected_degree);
    if (size != 0 && degree > edges.max_size() / size)
        throw std::length_error("graph reserve exceeds the container limit");
    adj_.resize(size);
    edges.reserve(size * degree);
    removed_.reserve(edges.capacity());
    for (int i = 0; i < n; ++i) adj_[i].reserve(expected_degree);
}

int Graph::n() const noexcept {
    return n_;
}

void Graph::add_edge(int u, int v, double w) {
    if (u < 0 || u >= n_ || v < 0 || v >= n_)
        throw std::invalid_argument("graph vertex out of range");
    if (edges.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("graph edge count exceeds the index range");
    int id = static_cast<int>(edges.size());
    edges.emplace_back(u, v, w);
    removed_.push_back(0);
    adj_[u].push_back(id);
    adj_[v].push_back(id);
}

void Graph::remove_edge(int u, int v) {
    for (int id : adj_[u]) {
        const Edge& e = edges[id];
        if (!removed_[id] && ((e.u == u && e.v == v) || (e.u == v && e.v == u))) {
            removed_[id] = 1;
            return;
        }
    }
}

bool Graph::is_connected() const {
    if (n_ == 0)
        return true;

    std::vector<char> visited(n_, 0);
    std::vector<int> stack = {0};
    visited[0] = 1;
    int count = 1;

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        for (int id : adj_[u]) {
            if (removed_[id])
                continue;
            int v = (edges[id].u == u ? edges[id].v : edges[id].u);
            if (!visited[v]) {
                visited[v] = 1;
                stack.push_back(v);
                ++count;
            }
        }
    }
    return count == n_;
}

std::vector<int> Graph::compute_components() const {
    std::vector<int> components(n_, -1);
    int cid = 0;

    for (int i = 0; i < n_; ++i) {
        if (components[i] != -1)
            continue;

        std::vector<int> stack = {i};
        components[i] = cid;

        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            for (int eid : adj_[x]) {
                if (removed_[eid])
                    continue;
                const Edge& e = edges[eid];
                int y = (e.u == x ? e.v : e.u);
                if (components[y] == -1) {
                    components[y] = cid;
                    stack.push_back(y);
                }
            }
        }
        ++cid;
    }
    return components;
}

std::vector<int> Graph::compute_components_subset(const std::vector<int>& subset) const {
    std::unordered_map<int, int> components;
    components.reserve(subset.size() * 2);

    std::unordered_set<int> in_subset(subset.begin(), subset.end());

    int cid = 0;
    for (int start : subset) {
        if (components.count(start))
            continue;

        std::vector<int> stack = {start};
        components[start] = cid;

        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            for (int eid : adj_[x]) {
                if (removed_[eid])
                    continue;
                const Edge& e = edges[eid];
                int y = (e.u == x ? e.v : e.u);
                if (in_subset.count(y) == 0)
                    continue;
                if (components.find(y) == components.end()) {
                    components[y] = cid;
                    stack.push_back(y);
                }
            }
        }
        ++cid;
    }

    std::vector<int> result;
    result.reserve(subset.size());
    for (int v : subset) {
        result.push_back(components[v]);
    }
    return result;
}

const std::vector<Edge>& Graph::get_edges() const noexcept {
    return edges;
}

const std::vector<int>& Graph::incident_edges(int u) const {
    return adj_[u];
}

}  // namespace mixdex
