#include "common.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/mst/boruvka.hpp"
#include "mixdex/mst/dt_boruvka.hpp"
#include "mixdex/mst/mst_compare.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/graph.hpp"

namespace mixdex::pybind {
using namespace mixdex;

namespace {

// Build the core-distance tree shared by dual-tree Boruvka and diagnostics.
BallTree make_core_tree(const Dataset& data, const std::vector<double>& core, int leaf_size) {
    BallTree tree(data, leaf_size);
    tree.set_core_dist(core);
    return tree;
}

}  // namespace

void bind_mst(py::module_& m) {
    py::class_<Edge>(m, "Edge")
        .def(py::init<int, int, double>())
        .def_readwrite("u", &Edge::u)
        .def_readwrite("v", &Edge::v)
        .def_readwrite("w", &Edge::w);

    py::class_<Graph>(m, "Graph")
        .def(py::init<int>())
        .def("n", &Graph::n)
        .def("add_edge", &Graph::add_edge)
        .def("is_connected", &Graph::is_connected)
        .def("get_edges", &Graph::get_edges);

    m.def(
        "boruvka",
        [](const Graph& g) {
            return boruvka(g);
        },
        py::arg("graph"));

    m.def(
        "boruvka_edges",
        [](const std::vector<Edge>& edges, int n) {
            return boruvka(edges, n);
        },
        py::arg("edges"),
        py::arg("n"));

    m.def(
        "dt_boruvka",
        [](const Dataset& data,
           const std::vector<double>& core,
           int leaf_size,
           bool deterministic) {
            return dual_tree_boruvka(make_core_tree(data, core, leaf_size), deterministic);
        },
        py::arg("data"),
        py::arg("core"),
        py::arg("leaf_size") = constants::kDefaultLeafSize,
        py::arg("deterministic") = true,
        "Dual-tree Boruvka MST. deterministic=True: the edge set does not depend on thread count "
        "or scheduling. deterministic=False: still a valid MST (same weights), but among "
        "equal-weight "
        "alternatives the chosen edges may vary between runs; faster on data without mass ties.");
    py::class_<DTBoruvkaStats>(m, "DTBoruvkaStats")
        .def_readonly("total_calls", &DTBoruvkaStats::total_calls)
        .def_readonly("pruned_calls", &DTBoruvkaStats::pruned_calls)
        .def_readonly("zero_lb_calls", &DTBoruvkaStats::zero_lb_calls)
        .def_readonly("leaf_pairs", &DTBoruvkaStats::leaf_pairs)
        .def_readonly("metric_calls", &DTBoruvkaStats::metric_calls)
        .def_readonly("fast_skipped", &DTBoruvkaStats::fast_skipped)
        .def_readonly("iterations", &DTBoruvkaStats::iterations)
        .def("pruned_ratio", &DTBoruvkaStats::pruned_ratio)
        .def("zero_lb_ratio", &DTBoruvkaStats::zero_lb_ratio)
        .def("__repr__", [](const DTBoruvkaStats& s) {
            return "DTBoruvkaStats(total_calls=" + std::to_string(s.total_calls) +
                   ", pruned_calls=" + std::to_string(s.pruned_calls) + " (" +
                   std::to_string(s.pruned_ratio() * 100.0) +
                   "%)"
                   ", zero_lb_calls=" +
                   std::to_string(s.zero_lb_calls) +
                   ", leaf_pairs=" + std::to_string(s.leaf_pairs) +
                   ", metric_calls=" + std::to_string(s.metric_calls) +
                   ", fast_skipped=" + std::to_string(s.fast_skipped) +
                   ", iterations=" + std::to_string(s.iterations) + ")";
        });

    py::class_<DTBoruvkaResult>(m, "DTBoruvkaResult")
        .def_readonly("edges", &DTBoruvkaResult::edges)
        .def_readonly("stats", &DTBoruvkaResult::stats);

    m.def(
        "dt_boruvka_with_stats",
        [](const Dataset& data,
           const std::vector<double>& core,
           int leaf_size,
           bool deterministic) {
            return dual_tree_boruvka_with_stats(make_core_tree(data, core, leaf_size),
                                                deterministic);
        },
        py::arg("data"),
        py::arg("core"),
        py::arg("leaf_size") = constants::kDefaultLeafSize,
        py::arg("deterministic") = true,
        "Same as dt_boruvka(), but also returns DTBoruvkaStats (pruning "
        "efficiency counters) alongside the MST edges.");

    m.def("mst_edge_recall",
          &mst_edge_recall,
          py::arg("candidate"),
          py::arg("reference"),
          "Fraction of candidate's edges also present in reference (as "
          "unordered pairs, weight-agnostic). Returns 0.0 if reference is "
          "empty.");
    m.def("check_lb_violations",
          [](const Dataset& data, const std::vector<double>& core, int leaf_size) {
              const BallTree tree = make_core_tree(data, core, leaf_size);

              std::vector<std::tuple<int, int, int, int, double, double>> violations;

              for (int qi = 0; qi < static_cast<int>(tree.nodes().size()); ++qi) {
                  for (int ri = qi; ri < static_cast<int>(tree.nodes().size()); ++ri) {
                      const auto& qnode = tree.nodes()[qi];
                      const auto& rnode = tree.nodes()[ri];
                      if (!qnode.is_leaf() || !rnode.is_leaf())
                          continue;

                      double lb = tree.lower_bound_mreach_node(qi, ri);

                      for (int q = qnode.low; q < qnode.high; ++q) {
                          for (int r = rnode.low; r < rnode.high; ++r) {
                              int qi_idx = tree.indices()[q];
                              int ri_idx = tree.indices()[r];
                              if (qi_idx >= ri_idx)
                                  continue;
                              double real = tree.call_metric(qi_idx, ri_idx);
                              if (real < lb)  // Check the bound directly, without a tolerance.
                                  violations.emplace_back(qi, ri, qi_idx, ri_idx, lb, real);
                          }
                      }
                  }
              }
              return violations;
          });
}

}  // namespace mixdex::pybind
