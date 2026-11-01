#include <map>
#include <unordered_map>
#include <vector>

#include "cond_tree_internal.hpp"

namespace mixdex {
namespace detail {

void slow_path(CondTreeState& st, const StepInfo& step, const ClusterPoints& cluster_points) {
    const auto& edges = st.edges;
    const int n = st.n;
    const int min_cluster_size = st.min_cluster_size;
    const auto& core = st.core;
    Graph& g = st.g;
    std::vector<int>& cluster = st.cluster;
    std::vector<std::vector<int>>& cl_members = st.cl_members;
    std::vector<CondRow>& rows = st.rows;
    std::vector<double>& lambda_birth = st.lambda_birth;
    int& next_cl = st.next_cl;
    const std::size_t i = step.i;
    const std::size_t j = step.j;
    const double w = step.w;
    const double lambda = step.lambda;
    RollbackUF& uf = st.uf;
    (void)edges;
    (void)i;
    (void)j;

    std::vector<int> all_affected_points;
    {
        std::size_t total = 0;
        for (auto& [cl_id, pts] : cluster_points) total += pts->size();
        all_affected_points.reserve(total);
        for (auto& [cl_id, pts] : cluster_points) {
            all_affected_points.insert(all_affected_points.end(), pts->begin(), pts->end());
        }
    }

    std::unordered_map<int, int> comp_id;
    comp_id.reserve(all_affected_points.size() * 2);
    for (int p : all_affected_points) {
        comp_id[p] = uf.find_root(p);
    }

    for (auto& [cl_id, pts_ptr] : cluster_points) {
        const std::vector<int>& pts = *pts_ptr;
        // Ordered UF roots keep component traversal stable across standard libraries.
        std::map<int, std::vector<int>> comp_to_pts;
        for (int p : pts) {
            comp_to_pts[comp_id[p]].push_back(p);
        }

        struct CompInfo {
            int cid;
            int size;
            bool has_edges;
            std::vector<int> pts;
        };

        std::vector<CompInfo> comps;

        for (auto& [cid, cpts] : comp_to_pts) {
            bool has_edges = false;

            for (int p : cpts) {
                for (int ei : g.incident_edges(p)) {
                    if (g.edge_removed(ei))
                        continue;
                    const Edge& ed = g.get_edges()[ei];
                    int other = (ed.u == p ? ed.v : ed.u);
                    auto other_it = comp_id.find(other);
                    if (other_it != comp_id.end() && other_it->second == cid) {
                        has_edges = true;
                        break;
                    }
                }
                if (has_edges)
                    break;
            }

            if (!has_edges) {
                for (int p : cpts) {
                    if (core[p] > w) {
                        has_edges = true;
                        break;
                    }
                }
            }

            comps.push_back({cid, static_cast<int>(cpts.size()), has_edges, std::move(cpts)});
        }

        std::vector<CompInfo*> alive;
        for (auto& c : comps) {
            if (c.has_edges && c.size >= min_cluster_size) {
                alive.push_back(&c);
            } else {
                for (int p : c.pts) {
                    shed_point(st, cl_id, p, lambda);
                }
            }
        }

        cl_members[cl_id - n].clear();
        cl_members[cl_id - n].shrink_to_fit();

        if (alive.size() == 1) {
            // Retain the cluster ID; only shed points have left.
            cl_members[cl_id - n] = std::move(alive[0]->pts);
        } else {
            for (auto* cinfo : alive) {
                int new_id = next_cl++;
                rows.push_back({cl_id, new_id, lambda, cinfo->size});
                lambda_birth.push_back(lambda);

                cl_members.push_back(std::move(cinfo->pts));

                for (int p : cl_members.back()) {
                    cluster[p] = new_id;
                }
            }
        }
    }
}

}  // namespace detail
}  // namespace mixdex
