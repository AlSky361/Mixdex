#include "common.hpp"
#include "mixdex/algorithms/dbscan/dbscan.hpp"
#include "mixdex/algorithms/knn/knn_classifier.hpp"
#include "mixdex/algorithms/outlier/knn_outlier.hpp"
#include "mixdex/algorithms/outlier/lof.hpp"
#include "mixdex/index/index.hpp"

namespace mixdex::pybind {
using namespace mixdex;

void bind_algorithms(py::module_& m) {
    def_indexed(
        m,
        "dbscan",
        [](const NeighbourSearch& s, double eps, int min_pts) {
            py::gil_scoped_release release;
            return dbscan(s, eps, min_pts);
        },
        py::arg("eps"),
        py::arg("min_pts"));

    // kNN classification
    m.attr("UNLABELED") = kUnlabeled;
    def_indexed(
        m,
        "knn_classify",
        [](const NeighbourSearch& s,
           const std::vector<int>& labels,
           int k,
           bool distance_weighted) {
            py::gil_scoped_release release;
            return knn_classify(s, labels, k, distance_weighted);
        },
        py::arg("labels"),
        py::arg("k"),
        py::arg("distance_weighted") = false);

    def_indexed(
        m,
        "knn_predict",
        [](const NeighbourSearch& s,
           const std::vector<int>& labels,
           const py::object& X_num,
           const py::object& X_cat,
           int k,
           bool distance_weighted) {
            auto b = make_query_batch(s.dataset(), X_num, X_cat);
            py::gil_scoped_release release;
            return knn_predict(s, labels, b.num_span(), b.cat_span(), b.m, k, distance_weighted);
        },
        py::arg("labels"),
        py::arg("X_num"),
        py::arg("X_cat"),
        py::arg("k"),
        py::arg("distance_weighted") = false);

    // Indexed outlier scores
    def_indexed(
        m,
        "knn_outlier_scores",
        [](const NeighbourSearch& s, int k) {
            py::gil_scoped_release release;
            return knn_distance_scores(s, k);
        },
        py::arg("k"));

    def_indexed(
        m,
        "knn_outlier_scores_points",
        [](const NeighbourSearch& s, const py::object& X_num, const py::object& X_cat, int k) {
            auto b = make_query_batch(s.dataset(), X_num, X_cat);
            py::gil_scoped_release release;
            return knn_distance_scores(s, b.num_span(), b.cat_span(), b.m, k);
        },
        py::arg("X_num"),
        py::arg("X_cat"),
        py::arg("k"));

    py::class_<LOF>(m, "LOF")
        .def(py::init<const Dataset&, int, SearchMethod, int>(),
             py::arg("data"),
             py::arg("k"),
             py::arg("method") = SearchMethod::BallTree,
             py::arg("leaf_size") = constants::kDefaultLeafSize,
             py::keep_alive<1, 2>(),
             py::call_guard<py::gil_scoped_release>())
        .def(py::init([](const Index& idx, int k) {
                 auto shared = idx.shared();
                 return std::make_unique<LOF>(std::move(shared), k);
             }),
             py::arg("index"),
             py::arg("k"),
             py::keep_alive<1, 2>(),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("k", &LOF::k)
        .def_property_readonly("k_distance", &LOF::k_distance)
        .def_property_readonly("lrd", &LOF::lrd)
        .def_property_readonly("scores", &LOF::scores)
        .def(
            "score",
            [](const LOF& lof, const py::object& X_num, const py::object& X_cat) {
                auto b = make_query_batch(lof.dataset(), X_num, X_cat);
                py::gil_scoped_release release;
                return lof.score(b.num_span(), b.cat_span(), b.m);
            },
            py::arg("X_num"),
            py::arg("X_cat"));
}

}  // namespace mixdex::pybind
