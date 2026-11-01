#include <cmath>

#include "common.hpp"
#include "mixdex/density/hierarchy.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/quantile.hpp"
#include "mixdex/scale.hpp"
#include "mixdex/synthetic/synthetic.hpp"
#include "mixdex/types.hpp"

namespace mixdex::pybind {
using namespace mixdex;

namespace {

SyntheticParams make_params(std::size_t n,
                            std::size_t d_num,
                            int n_clusters,
                            double separation,
                            double cat_separation,
                            double p_cat,
                            int cardinality,
                            double cluster_density,
                            double elongation,
                            double cluster_std,
                            std::uint64_t seed) {
    return SyntheticParams{n,
                           d_num,
                           n_clusters,
                           separation,
                           cat_separation,
                           p_cat,
                           cardinality,
                           cluster_density,
                           elongation,
                           cluster_std,
                           seed};
}

// Expose g argument types so pybind11 can infer the composed signature.
template <class F, class R, class... A>
auto compose(F f, R (*g)(A...)) {
    return [f, g](A... a) {
        return f(g(a...));
    };
}

}  // namespace

void bind_data(py::module_& m) {
    py::register_exception<std::invalid_argument>(m, "InvalidArgument", PyExc_ValueError);
    py::register_exception<std::runtime_error>(m, "RuntimeError", PyExc_RuntimeError);

    py::enum_<SearchMethod>(m, "SearchMethod")
        .value("BallTree", SearchMethod::BallTree)
        .value("BruteForce", SearchMethod::BruteForce);

    py::enum_<NumericType>(m, "NumericType")
        .value("Continuous", NumericType::Continuous)
        .value("Discrete", NumericType::Discrete)
        .export_values();

    py::enum_<ScaleLevel>(m, "ScaleLevel")
        .value("Iqr", ScaleLevel::Iqr)
        .value("Idr", ScaleLevel::Idr)
        .value("Range", ScaleLevel::Range);

    py::enum_<FeatureType>(m, "FeatureType")
        .value("Numeric", FeatureType::Numeric)
        .value("Categorical", FeatureType::Categorical)
        .value("CategoricalSet", FeatureType::CategoricalSet)
        .export_values();

    py::enum_<QuantileMethod>(m, "QuantileMethod")
        .value("MedianUnbiased", QuantileMethod::MedianUnbiased)
        .value("InvertedCDF", QuantileMethod::InvertedCDF)
        .export_values();

    py::enum_<MSTMode>(m, "MSTMode")
        .value("Exact", MSTMode::Exact)
        .value("Approximate", MSTMode::Approximate)
        .export_values();
    m.attr("HDBSCANMode") = m.attr("MSTMode");  // Compatibility alias.

    py::class_<NumFeature>(m, "NumFeature")
        .def(py::init<>())
        .def_readwrite("array", &NumFeature::array)
        .def_readwrite("type", &NumFeature::type)
        .def("size", &NumFeature::size);

    py::class_<CategoricalFeature>(m, "CategoricalFeature")
        .def(py::init<>())
        .def_readwrite("encoded", &CategoricalFeature::encoded);

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<>())
        .def(py::init<const std::vector<NumFeature>&,
                      const std::vector<CategoricalFeature>&,
                      const std::vector<std::string>&>(),
             py::arg("num_features"),
             py::arg("cat_features"),
             py::arg("feat_names"))
        .def(py::init<std::vector<double>,
                      std::vector<int>,
                      int,
                      int,
                      int,
                      std::vector<FeatureType>,
                      std::vector<NumericType>,
                      std::vector<std::string>>(),
             py::arg("num_data"),
             py::arg("cat_data"),
             py::arg("n"),
             py::arg("n_num"),
             py::arg("n_cat"),
             py::arg("feat_types"),
             py::arg("num_types"),
             py::arg("feat_names"))
        .def("n_rows", &Dataset::n_rows)
        .def("n_num", &Dataset::n_num)
        .def("n_cat", &Dataset::n_cat)
        .def("n_cols", &Dataset::n_cols)
        .def("feat_type",
             [](const Dataset& d, int k) {
                 check_index(k, d.n_cols());
                 return d.feat_type(k);
             })
        .def("num_type",
             [](const Dataset& d, int k) {
                 check_index(k, d.n_num());
                 return d.num_type(k);
             })
        .def("feat_name",
             [](const Dataset& d, int k) {
                 check_index(k, d.n_cols());
                 return d.feat_name(k);
             })
        .def("num_data",
             [](const Dataset& d) {
                 return d.num_data();
             })
        .def("cat_data",
             [](const Dataset& d) {
                 return d.cat_data();
             })
        .def("row_num",
             [](const Dataset& d, int i) {
                 check_index(i, d.n_rows());
                 auto span = d.row_num_span(i);
                 return std::vector<double>(span.begin(), span.end());
             })
        .def("col_num",
             [](const Dataset& d, int k) {
                 check_index(k, d.n_num());
                 auto span = d.col_num_span(k);
                 return span.to_vector();
             })
        .def("set_weights", &Dataset::set_weights, py::arg("weights"))
        .def("set_category_sets", &Dataset::set_category_sets, py::arg("k"), py::arg("sets"))
        .def(
            "weight",
            [](const Dataset& d, int k) {
                check_index(k, d.n_cols());
                return d.weight(k);
            },
            py::arg("k"))
        .def("inv_weights_sum", &Dataset::inv_weights_sum)
        .def("has_weights", &Dataset::has_weights);

    m.def(
        "quantiles_select",
        [](std::vector<double> values, const std::vector<double>& ps, QuantileMethod method) {
            if (method != QuantileMethod::InvertedCDF && method != QuantileMethod::MedianUnbiased)
                throw std::invalid_argument("unknown quantile method");
            for (double value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("quantiles_select requires finite values");
            for (double p : ps)
                if (!std::isfinite(p) || p < 0 || p > 1)
                    throw std::invalid_argument("quantile probabilities must be in [0, 1]");
            return quantiles_select(values, ps, method);
        },
        py::arg("values"),
        py::arg("ps"),
        py::arg("method"),
        "Select quantiles from an unsorted copy; inputs are unchanged.");
    m.def("quantiles",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> x,
             std::vector<double> ps,
             QuantileMethod method) {
              auto buf = x.request();
              if (buf.ndim != 1)
                  throw std::invalid_argument("quantiles: ожидается 1D-массив, получено ndim=" +
                                              std::to_string(buf.ndim));
              return quantiles(std::span<const double>(static_cast<const double*>(buf.ptr),
                                                       static_cast<size_t>(buf.size)),
                               ps,
                               method);
          });

    m.def("scale_feature",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> x, NumericType type) {
              auto buf = x.request();
              if (buf.ndim != 1)
                  throw std::invalid_argument("scale_feature: ожидается 1D-массив, получено ndim=" +
                                              std::to_string(buf.ndim));
              std::span<const double> span(static_cast<const double*>(buf.ptr),
                                           static_cast<size_t>(buf.size));
              return scale_feature(span, type);
          });

    m.def("scale",
          &scale,
          py::arg("data"),
          "Center and scale numeric features; reject numerically unusable widths.");

    py::class_<Scaler>(m, "Scaler")
        .def(py::init<>())
        .def_static(
            "fit",
            &Scaler::fit,
            py::arg("data"),
            "Fit medians and positive widths using IQR, interdecile range, then full range.")
        .def("transform", &Scaler::transform, py::arg("data"))
        .def(
            "transform_rows",
            [](const Scaler& sc, py::array_t<double, py::array::c_style | py::array::forcecast> X) {
                if (X.ndim() != 2 || X.shape(1) != sc.n_num())
                    throw std::invalid_argument("X must be a 2D array with n_num columns");
                const std::size_t rows = static_cast<std::size_t>(X.shape(0));
                auto out = sc.transform_rows({X.data(), static_cast<std::size_t>(X.size())}, rows);
                py::array_t<double> res({rows, static_cast<std::size_t>(sc.n_num())});
                std::copy(out.begin(), out.end(), res.mutable_data());
                return res;
            },
            py::arg("X_num"))
        .def_property_readonly("n_num", &Scaler::n_num)
        .def_property_readonly("medians", &Scaler::medians)
        .def_property_readonly("scales", &Scaler::scales)
        .def_property_readonly("levels", &Scaler::levels);

    // Share SyntheticParams field order, factory and argument defaults between both entry points.
    auto with_synthetic_args = [](auto&& register_with) {
        const SyntheticParams d;
        register_with(py::arg("n") = d.n,
                      py::arg("d_num") = d.d_num,
                      py::arg("n_clusters") = d.n_clusters,
                      py::arg("separation") = d.separation,
                      py::arg("cat_separation") = d.cat_separation,
                      py::arg("p_cat") = d.p_cat,
                      py::arg("cardinality") = d.cardinality,
                      py::arg("cluster_density") = d.cluster_density,
                      py::arg("elongation") = d.elongation,
                      py::arg("cluster_std") = d.cluster_std,
                      py::arg("seed") = d.seed);
    };

    py::class_<SyntheticParams> params_cls(m, "SyntheticParams");
    with_synthetic_args([&](auto... args) {
        params_cls.def(py::init(&make_params), args...);
    });
    params_cls.def_readwrite("n", &SyntheticParams::n)
        .def_readwrite("d_num", &SyntheticParams::d_num)
        .def_readwrite("n_clusters", &SyntheticParams::n_clusters)
        .def_readwrite("separation", &SyntheticParams::separation)
        .def_readwrite("cat_separation", &SyntheticParams::cat_separation)
        .def_readwrite("p_cat", &SyntheticParams::p_cat)
        .def_readwrite("cardinality", &SyntheticParams::cardinality)
        .def_readwrite("cluster_density", &SyntheticParams::cluster_density)
        .def_readwrite("elongation", &SyntheticParams::elongation)
        .def_readwrite("cluster_std", &SyntheticParams::cluster_std)
        .def_readwrite("seed", &SyntheticParams::seed)
        .def("__repr__", [](const SyntheticParams& p) {
            return "SyntheticParams(n=" + std::to_string(p.n) +
                   ", d_num=" + std::to_string(p.d_num) +
                   ", n_clusters=" + std::to_string(p.n_clusters) +
                   ", separation=" + std::to_string(p.separation) +
                   ", cat_separation=" + std::to_string(p.cat_separation) +
                   ", p_cat=" + std::to_string(p.p_cat) +
                   ", cardinality=" + std::to_string(p.cardinality) +
                   ", cluster_density=" + std::to_string(p.cluster_density) +
                   ", elongation=" + std::to_string(p.elongation) +
                   ", cluster_std=" + std::to_string(p.cluster_std) +
                   ", seed=" + std::to_string(p.seed) + ")";
        });

    py::class_<SyntheticResult>(m, "SyntheticResult")
        .def_readonly("dataset", &SyntheticResult::dataset)
        .def_readonly("true_labels", &SyntheticResult::true_labels);

    m.def("make_synthetic",
          &make_synthetic,
          py::arg("params"),
          "Build a synthetic mixed-type Dataset with controllable cluster geometry.");

    with_synthetic_args([&](auto... args) {
        m.def("synth",
              compose(&make_synthetic, &make_params),
              args...,
              "Shortcut for make_synthetic(SyntheticParams(...)) -- same keyword "
              "arguments as SyntheticParams, returns SyntheticResult directly.");
    });
}

}  // namespace mixdex::pybind
