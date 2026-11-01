#include <limits>
#include <stdexcept>

#include "mixdex/utils/graph.hpp"

using mixdex::Graph;

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("graph validation regression");
}

template <class Exception, class Fn>
void rejects(Fn fn) {
    bool caught = false;
    try {
        fn();
    } catch (const Exception&) {
        caught = true;
    }
    check(caught);
}

int main() {
    rejects<std::invalid_argument>([] {
        Graph g(-1);
    });
    rejects<std::invalid_argument>([] {
        Graph g(2, -1);
    });
    rejects<std::length_error>([] {
        Graph g(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    });

    Graph empty;
    rejects<std::invalid_argument>([&] {
        empty.add_edge(0, 0, 1);
    });
    Graph graph(3);
    rejects<std::invalid_argument>([&] {
        graph.add_edge(-1, 0, 1);
    });
    rejects<std::invalid_argument>([&] {
        graph.add_edge(0, 3, 1);
    });
    check(graph.get_edges().empty());
    graph.add_edge(0, 1, 2);
    graph.add_edge(1, 2, 3);
    check(graph.is_connected());
    check(graph.incident_edges(1).size() == 2);
    graph.remove_edge(1, 2);
    check(!graph.is_connected());
}
