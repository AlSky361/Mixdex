#include "common.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/index/brute_force/brute_force.hpp"
#include "mixdex/index/graph/find_k/find_k.hpp"
#include "mixdex/index/graph/knn_graph/knn_graph.hpp"
#include "mixdex/index/index.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"

namespace mixdex::pybind {
using namespace mixdex;

void bind_index(py::module_& m) {
    // Build an index once and reuse it across algorithms.
    py::class_<Index>(m, "Index")
        .def(py::init<const Dataset&, SearchMethod, int>(),
             py::arg("data"),
             py::arg("method") = SearchMethod::BallTree,
             py::arg("leaf_size") = constants::kDefaultLeafSize,
             py::keep_alive<1, 2>(),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("n_rows", &Index::n_rows)
        .def_property_readonly("method", &Index::method)
        .def_property_readonly("leaf_size", &Index::leaf_size);

    m.def(
        "knn_filtered",
        [](const Index& index,
           std::vector<double> num,
           std::vector<int> cat,
           int k,
           py::array_t<unsigned char, py::array::c_style> active,
           std::vector<int> excluded,
           std::vector<std::vector<Code>> sets,
           bool include_ties) {
            if (active.ndim() != 1)
                throw std::invalid_argument("active must be a one-dimensional uint8 mask");
            const auto& search = index.search();
            py::gil_scoped_release release;
            const RowView q{num, cat, sets};
            auto result = search.knn_filtered(
                q, k, {active.data(), static_cast<std::size_t>(active.size())}, excluded);
            if (include_ties && result.size() == static_cast<std::size_t>(k)) {
                auto tied = search.radius_query(q, result.back().second);
                std::erase_if(tied, [&](const auto& item) {
                    return !active.data()[item.first] ||
                           std::binary_search(excluded.begin(), excluded.end(), item.first);
                });
                return tied;
            }
            return result;
        },
        py::arg("index"),
        py::arg("numeric"),
        py::arg("categorical"),
        py::arg("k"),
        py::arg("active"),
        py::arg("excluded_rows") = std::vector<int>{},
        py::arg("categorical_sets") = std::vector<std::vector<Code>>{},
        py::arg("include_ties") = false);

    // kNN and radius queries accept either Index or Dataset.
    def_indexed(
        m,
        "knn",
        [](const NeighbourSearch& s, int query_idx, int k) {
            return s.knn(query_idx, k);
        },
        py::arg("query_idx"),
        py::arg("k"));

    def_indexed(
        m,
        "knn_all",
        [](const NeighbourSearch& s, int k) {
            py::gil_scoped_release release;
            return s.knn_batch(k);
        },
        py::arg("k"));

    def_indexed(
        m,
        "knn_query",
        [](const NeighbourSearch& s, const py::object& X_num, const py::object& X_cat, int k) {
            return knn_query_points(s, X_num, X_cat, k);
        },
        py::arg("X_num"),
        py::arg("X_cat"),
        py::arg("k"));

    def_indexed(
        m,
        "radius_query_points",
        [](const NeighbourSearch& s, const py::object& X_num, const py::object& X_cat, double r) {
            return radius_query_points(s, X_num, X_cat, r);
        },
        py::arg("X_num"),
        py::arg("X_cat"),
        py::arg("r"));

    // Indexed graph operations
    def_indexed(
        m,
        "knn_graph",
        [](const NeighbourSearch& s, int k) {
            return build_knn_graph(s, k);
        },
        py::arg("k"));

    def_indexed(
        m,
        "find_k_connected",
        [](const NeighbourSearch& s, int m_pts, int k_max) {
            return find_k_connected(s, m_pts, k_max);
        },
        py::arg("m_pts"),
        py::arg("k_max"));

    // Test-only node view combines geometry with separately stored mreach data.
    struct NodeView {
        int center_idx, left, right, low, high;
        double radius_num, radius_mreach, min_core;
        bool is_leaf() const {
            return left == -1 && right == -1;
        }
    };
    py::class_<NodeView>(m, "BallTreeNode")
        .def_readonly("center_idx", &NodeView::center_idx)
        .def_readonly("radius_mreach", &NodeView::radius_mreach)
        .def_readonly("radius_num", &NodeView::radius_num)
        .def_readonly("left", &NodeView::left)
        .def_readonly("right", &NodeView::right)
        .def_readonly("low", &NodeView::low)
        .def_readonly("high", &NodeView::high)
        .def_readonly("min_core", &NodeView::min_core)
        .def("is_leaf", &NodeView::is_leaf);

    py::class_<BallTree>(m, "BallTree")
        .def(py::init<const Dataset&, int>(),
             py::arg("data"),
             py::arg("leaf_size"),
             py::keep_alive<1, 2>())
        .def(
            "knn",
            [](const BallTree& t, int query_idx, int k) {
                return t.knn(query_idx, k);
            },
            py::arg("query_idx"),
            py::arg("k"))
        .def(
            "knn_batch",
            [](const BallTree& t, int k) {
                return t.knn_batch(k);
            },
            py::arg("k"))
        .def(
            "knn_query",
            [](const BallTree& t, const py::object& X_num, const py::object& X_cat, int k) {
                return knn_query_points(t, X_num, X_cat, k);
            },
            py::arg("X_num"),
            py::arg("X_cat"),
            py::arg("k"))
        .def(
            "radius_query_points",
            [](const BallTree& t, const py::object& X_num, const py::object& X_cat, double r) {
                return radius_query_points(t, X_num, X_cat, r);
            },
            py::arg("X_num"),
            py::arg("X_cat"),
            py::arg("r"))
        .def("n_rows", &BallTree::n_rows)
        .def("lower_bound", &BallTree::lower_bound, py::arg("query_idx"), py::arg("node_idx"))
        .def("lower_bound_mreach",
             &BallTree::lower_bound_mreach,
             py::arg("query_idx"),
             py::arg("node_idx"))
        .def("lower_bound_mreach_node",
             &BallTree::lower_bound_mreach_node,
             py::arg("qnode_idx"),
             py::arg("rnode_idx"))
        .def("call_metric", &BallTree::call_metric, py::arg("i"), py::arg("j"))
        .def(
            "radius_query",
            [](const BallTree& t, int query_idx, double r) {
                return t.radius_query(query_idx, r);
            },
            py::arg("query_idx"),
            py::arg("r"))
        .def("set_core_dist", &BallTree::set_core_dist)
        .def("bbox_lo",
             [](const BallTree& t, int node_idx) {
                 auto s = t.bbox_lo(node_idx);
                 return std::vector<double>(s.begin(), s.end());
             })
        .def("bbox_hi",
             [](const BallTree& t, int node_idx) {
                 auto s = t.bbox_hi(node_idx);
                 return std::vector<double>(s.begin(), s.end());
             })
        .def_property_readonly("nodes",
                               [](const BallTree& t) {
                                   const auto& ns = t.nodes();
                                   const auto& mr = t.mreach_nodes();
                                   std::vector<NodeView> out;
                                   out.reserve(ns.size());
                                   for (std::size_t i = 0; i < ns.size(); ++i) {
                                       const auto& n = ns[i];
                                       const bool has = i < mr.size();
                                       out.push_back({n.center_idx,
                                                      n.left,
                                                      n.right,
                                                      n.low,
                                                      n.high,
                                                      n.radius_num,
                                                      has ? mr[i].radius : 0.0,
                                                      has ? mr[i].min_core : 0.0});
                                   }
                                   return out;
                               })
        .def_property_readonly("indices", &BallTree::indices)
        .def_property_readonly("root", &BallTree::root);

    m.def("full_mreach_edges", [](const Dataset& data, const std::vector<double>& core) {
        auto metric = [&](int i, int j) {
            return mixdex::distance(data, i, j);
        };
        return build_full_mreach_edges(data, core, metric);
    });

    // Node-visit profiling is enabled only by MIXDEX_PROFILE_KNN at startup.
    m.def(
        "_knn_nodes_visited",
        []() {
            return g_knn_nodes_visited.load();
        },
        "BallTree nodes visited since reset; zero unless MIXDEX_PROFILE_KNN was set at startup.");
    m.def(
        "_knn_nodes_reset",
        []() {
            g_knn_nodes_visited.store(0);
        },
        "Reset the kNN node-visit counter.");
}

}  // namespace mixdex::pybind
