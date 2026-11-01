#include "common.hpp"
#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/algorithms/hdbscan/glosh.hpp"
#include "mixdex/algorithms/hdbscan/hdbscan.hpp"
#include "mixdex/types.hpp"

#ifdef MIXDEX_PROFILE_FASTPATH
namespace mixdex {
extern long long g_fp_hot_steps, g_fp_reason_multicluster, g_fp_reason_multinonself,
    g_fp_reason_group_gt2, g_fp_reason_self_unrelated, g_fp_reason_cap_bothbig, g_fp_reason_ok;
extern long long g_fp_multinonself_edgesum, g_fp_multinonself_sizesum, g_fp_multinonself_count;
}  // namespace mixdex
#endif  // MIXDEX_PROFILE_FASTPATH

namespace mixdex::pybind {
using namespace mixdex;

void bind_hdbscan(py::module_& m) {
    py::class_<CondRow>(m, "CondRow")
        .def_readonly("parent_cl", &CondRow::parent_cl)
        .def_readonly("child_cl", &CondRow::child_cl)
        .def_readonly("lambda_", &CondRow::lambda)
        .def_readonly("size", &CondRow::child_cl_size);

    m.def(
        "extract_clusters",
        [](const std::vector<CondRow>& rows,
           const std::vector<double>& lambda_birth,
           int n,
           int next_cl,
           const std::vector<int>& final_cluster) {
            std::vector<int> labels;
            std::vector<double> stabilities;
            extract_clusters(rows, lambda_birth, n, next_cl, final_cluster, labels, stabilities);
            return std::make_tuple(labels, stabilities);
        },
        py::arg("rows"),
        py::arg("lambda_birth"),
        py::arg("n"),
        py::arg("next_cl"),
        py::arg("final_cluster"));

    m.def(
        "compute_glosh",
        [](const std::vector<double>& core,
           const std::vector<CondRow>& rows,
           const std::vector<int>& cl_parent,
           int n,
           int next_cl) {
            return compute_glosh(core, rows, cl_parent, n, next_cl);
        },
        py::arg("core"),
        py::arg("rows"),
        py::arg("cl_parent"),
        py::arg("n"),
        py::arg("next_cl"));

    m.def(
        "build_condensed_tree",
        [](const std::vector<Edge>& mst, int n, int mcs, const std::vector<double>& core) {
            std::vector<CondRow> rows;
            std::vector<double> lambda_birth;
            std::vector<int> final_cluster;
            int next_cl = n + 1;

            build_condensed_tree(mst, n, mcs, core, rows, lambda_birth, next_cl, final_cluster);

            return std::make_tuple(rows, lambda_birth, next_cl, final_cluster);
        },
        py::arg("mst"),
        py::arg("n"),
        py::arg("min_cluster_size"),
        py::arg("core"));

    py::class_<HDBSCANResult>(m, "HDBSCANResult")
        .def_readonly("labels", &HDBSCANResult::labels)
        .def_readonly("outlier_scores", &HDBSCANResult::outlier_scores)
        .def_readonly("cluster_stabilities", &HDBSCANResult::cluster_stabilities);

    m.def(
        "hdbscan",
        [](const Dataset& data,
           int m_pts,
           int min_cluster_size,
           HDBSCANMode mode,
           SearchMethod method,
           int leaf_size,
           int k_opt,
           bool deterministic_mst) {
            py::gil_scoped_release release;  // Only C++ data is accessed while the GIL is released.
            return hdbscan(
                data, m_pts, min_cluster_size, mode, method, leaf_size, k_opt, deterministic_mst);
        },
        py::arg("data"),
        py::arg("m_pts"),
        py::arg("min_cluster_size"),
        py::arg("mode"),
        py::arg("method"),
        py::arg("leaf_size") = constants::kDefaultLeafSize,
        py::arg("k_opt") = constants::kKUnset,
        py::arg("deterministic_mst") = true);

#ifdef MIXDEX_PROFILE_FASTPATH
    m.def("_fp_stats", []() {
        py::dict d;
        d["hot_steps"] = mixdex::g_fp_hot_steps;
        d["multicluster"] = mixdex::g_fp_reason_multicluster;
        d["multinonself"] = mixdex::g_fp_reason_multinonself;
        d["group_gt2"] = mixdex::g_fp_reason_group_gt2;
        d["self_unrelated"] = mixdex::g_fp_reason_self_unrelated;
        d["cap_bothbig"] = mixdex::g_fp_reason_cap_bothbig;
        d["ok"] = mixdex::g_fp_reason_ok;
        d["multinonself_avg_edges"] =
            mixdex::g_fp_multinonself_count
                ? (double)mixdex::g_fp_multinonself_edgesum / mixdex::g_fp_multinonself_count
                : 0.0;
        d["multinonself_avg_size"] =
            mixdex::g_fp_multinonself_count
                ? (double)mixdex::g_fp_multinonself_sizesum / mixdex::g_fp_multinonself_count
                : 0.0;
        return d;
    });
    m.def("_fp_reset", []() {
        mixdex::g_fp_hot_steps = mixdex::g_fp_reason_multicluster =
            mixdex::g_fp_reason_multinonself = 0;
        mixdex::g_fp_reason_group_gt2 = mixdex::g_fp_reason_self_unrelated =
            mixdex::g_fp_reason_cap_bothbig = 0;
        mixdex::g_fp_reason_ok = 0;
        mixdex::g_fp_multinonself_edgesum = mixdex::g_fp_multinonself_sizesum =
            mixdex::g_fp_multinonself_count = 0;
    });
#endif
}

}  // namespace mixdex::pybind
