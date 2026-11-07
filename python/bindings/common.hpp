#pragma once
// Shared binding helpers.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include <tuple>
#include <utility>

#include "mixdex/index/index.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"

namespace py = pybind11;

namespace mixdex::pybind {

void check_index(int index, int count);

// External NumPy batches: float64 numeric and int32 categorical matrices.
// None denotes an absent block; hidden visibility avoids array_t export warnings.
#if defined(__GNUC__) || defined(__clang__)
#define MIXDEX_HIDDEN __attribute__((visibility("hidden")))
#else
#define MIXDEX_HIDDEN
#endif
struct MIXDEX_HIDDEN QueryBatch {
    py::array_t<double, py::array::c_style | py::array::forcecast> num;
    py::array_t<int, py::array::c_style | py::array::forcecast> cat;
    std::size_t m{0};
    std::span<const double> num_span() const {
        return {num.size() ? num.data() : nullptr, static_cast<std::size_t>(num.size())};
    }
    std::span<const int> cat_span() const {
        return {cat.size() ? cat.data() : nullptr, static_cast<std::size_t>(cat.size())};
    }
};

QueryBatch make_query_batch(const Dataset& data, const py::object& X_num, const py::object& X_cat);

// Index construction uses no Python objects and releases the GIL.
inline Index build_index(const Dataset& data, SearchMethod method, int leaf_size) {
    py::gil_scoped_release release;
    return Index(data, method, leaf_size);
}

// Shared external-batch kNN/radius implementation for Index and BallTree.
inline auto knn_query_points(const NeighbourSearch& s,
                             const py::object& X_num,
                             const py::object& X_cat,
                             int k) {
    auto b = make_query_batch(s.dataset(), X_num, X_cat);
    py::gil_scoped_release release;
    return s.knn_query_batch(b.num_span(), b.cat_span(), b.m, k);
}

inline auto radius_query_points(const NeighbourSearch& s,
                                const py::object& X_num,
                                const py::object& X_cat,
                                double r) {
    auto b = make_query_batch(s.dataset(), X_num, X_cat);
    py::gil_scoped_release release;
    return s.radius_query_batch(b.num_span(), b.cat_span(), b.m, r);
}

// Register Index and Dataset overloads from one handler to keep defaults aligned.
// Dataset overloads build an index; handlers manage GIL release.
template <class Fn, class R, class... A, class... Extra>
void def_indexed_impl(py::module_& m,
                      const char* name,
                      Fn fn,
                      R (Fn::*)(const NeighbourSearch&, A...) const,
                      Extra... extra) {
    m.def(
        name,
        [fn](const Dataset& data, A... a, SearchMethod method, int leaf_size) {
            const Index idx = build_index(data, method, leaf_size);
            return fn(idx.search(), std::forward<A>(a)...);
        },
        py::arg("data"),
        extra...,
        py::arg("method") = SearchMethod::BallTree,
        py::arg("leaf_size") = constants::kDefaultLeafSize);
    m.def(
        name,
        [fn](const Index& idx, A... a) {
            return fn(idx.search(), std::forward<A>(a)...);
        },
        py::arg("index"),
        extra...);
}

template <class Fn, class... Extra>
void def_indexed(py::module_& m, const char* name, Fn fn, Extra... extra) {
    def_indexed_impl(m, name, fn, &Fn::operator(), extra...);
}

// Registrations called from the Python module entry point.
void bind_data(py::module_& m);  // Enums, features, Dataset, quantiles, scaling and synthetic data.
void bind_encoding(py::module_& m);    // Scalar and set label dictionaries.
void bind_density(py::module_& m);     // Distances and density hierarchy.
void bind_index(py::module_& m);       // Index, BallTree, queries and graphs.
void bind_mst(py::module_& m);         // Graphs and minimum spanning trees.
void bind_algorithms(py::module_& m);  // DBSCAN, kNN classification, outliers and LOF.
void bind_hdbscan(py::module_& m);     // Condensed tree, GLOSH and HDBSCAN.

}  // namespace mixdex::pybind
