#include <cmath>
#include <stdexcept>

#include "common.hpp"
#include "mixdex/density/core/core.hpp"
#include "mixdex/density/hierarchy.hpp"
#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/types.hpp"

namespace mixdex::pybind {
using namespace mixdex;

void bind_density(py::module_& m) {
    m.def("distance", [](const Dataset& d, int i, int j) {
        check_index(i, d.n_rows());
        check_index(j, d.n_rows());
        return distance(d, i, j);
    });

    m.def(
        "distance_matrix",
        [](const Dataset& data) {
            const int n = data.n_rows();
            std::vector<std::vector<double>> mat(n, std::vector<double>(n, 0.0));
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) {
                    double d = distance(data, i, j);
                    mat[i][j] = d;
                    mat[j][i] = d;
                }
            return mat;
        },
        py::arg("data"));

    m.def("mreach_distance",
          [](const std::vector<double>& core, const Dataset& data, int i, int j) {
              check_index(i, data.n_rows());
              check_index(j, data.n_rows());
              if (core.size() != static_cast<std::size_t>(data.n_rows()))
                  throw std::invalid_argument("one core distance per row is required");
              for (double value : {core[i], core[j]})
                  if (!std::isfinite(value) || value < 0)
                      throw std::invalid_argument("core distances must be finite and nonnegative");
              auto metric = [&](int a, int b) {
                  return mixdex::distance(data, a, b);
              };
              return mreach_distance(core, i, j, metric);
          });

    def_indexed(
        m,
        "core_distances",
        [](const NeighbourSearch& s, int m_pts) {
            return core_distances(s, m_pts);
        },
        py::arg("m_pts"));

    // Density hierarchy and DBSCAN*
    py::class_<DensityHierarchy>(m, "DensityHierarchy")
        .def(py::init<const Dataset&, int, HDBSCANMode, SearchMethod, int, int, bool>(),
             py::arg("data"),
             py::arg("m_pts"),
             py::arg("mode"),
             py::arg("method"),
             py::arg("leaf_size") = constants::kDefaultLeafSize,
             py::arg("k_opt") = constants::kKUnset,
             py::arg("deterministic_mst") = true,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("m_pts", &DensityHierarchy::m_pts)
        .def_property_readonly("core_distances", &DensityHierarchy::core_distances)
        .def_property_readonly("mst_edges",
                               [](const DensityHierarchy& h) {
                                   std::vector<std::tuple<int, int, double>> out;
                                   out.reserve(h.mst_edges().size());
                                   for (const Edge& e : h.mst_edges())
                                       out.emplace_back(e.u, e.v, e.w);
                                   return out;
                               })
        .def("dbscan_star",
             &DensityHierarchy::dbscan_star,
             py::arg("eps"),
             py::arg("min_cluster_size") = 1);
}

}  // namespace mixdex::pybind
