#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cond_tree_internal.hpp"

namespace mixdex {

#ifdef MIXDEX_PROFILE_FASTPATH
long long g_fp_hot_steps = 0;
long long g_fp_reason_multicluster = 0;
long long g_fp_reason_multinonself = 0;
long long g_fp_reason_group_gt2 = 0;
long long g_fp_reason_self_unrelated = 0;
long long g_fp_reason_cap_bothbig = 0;
long long g_fp_reason_ok = 0;
long long g_fp_multinonself_edgesum = 0;
long long g_fp_multinonself_sizesum = 0;
long long g_fp_multinonself_count = 0;
#endif

namespace detail {

bool try_fast_path(CondTreeState& st,
                   const StepInfo& step,
                   const std::unordered_set<int>& affected_clusters,
                   const ClusterPoints& cluster_points) {
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

    bool fast_path_ok = false;
    int fast_path_cl_id = -1;
    std::vector<int> fp_small_side;
    bool fp_small_is_u = true;
    const std::size_t cap = static_cast<std::size_t>(min_cluster_size) + 1;

#ifdef MIXDEX_PROFILE_FASTPATH
    {
        std::size_t max_affected_size = 0;
        for (auto& kv : cluster_points) {
            max_affected_size = std::max(max_affected_size, kv.second->size());
        }
        if (max_affected_size > 50) {
            std::size_t k_self = 0, k_nonself = 0;
            for (std::size_t k = i; k < j; ++k) {
                if (edges[k].u == edges[k].v)
                    ++k_self;
                else
                    ++k_nonself;
            }
            ++g_fp_hot_steps;
            if (affected_clusters.size() != 1)
                ++g_fp_reason_multicluster;
            else if (k_nonself != 1) {
                ++g_fp_reason_multinonself;
                g_fp_multinonself_edgesum += static_cast<long long>(k_nonself);
                g_fp_multinonself_sizesum += static_cast<long long>(max_affected_size);
                ++g_fp_multinonself_count;
            } else if ((j - i) > 2)
                ++g_fp_reason_group_gt2;
        }
    }
#endif

    bool entered_fastpath_branch = false;
    int fp_u = -1, fp_v = -1;
    {
        std::size_t k_self = 0, k_nonself = 0;
        int nonself_idx = -1;
        for (std::size_t k = i; k < j; ++k) {
            if (edges[k].u == edges[k].v)
                ++k_self;
            else {
                ++k_nonself;
                nonself_idx = static_cast<int>(k);
            }
        }

        if (affected_clusters.size() == 1 && k_nonself == 1 && (j - i) <= 2 && k_self <= 1) {
            bool self_ok = true;
            if (k_self == 1) {
                for (std::size_t k = i; k < j; ++k) {
                    if (edges[k].u == edges[k].v) {
                        int sp = edges[k].u;
                        if (sp != edges[nonself_idx].u && sp != edges[nonself_idx].v)
                            self_ok = false;
                    }
                }
            }
            if (self_ok) {
                entered_fastpath_branch = true;
                fp_u = edges[nonself_idx].u;
                fp_v = edges[nonself_idx].v;
            }
#ifdef MIXDEX_PROFILE_FASTPATH
            else {
                std::size_t max_affected_size = 0;
                for (auto& kv : cluster_points)
                    max_affected_size = std::max(max_affected_size, kv.second->size());
                if (max_affected_size > 50) {
                    ++g_fp_reason_self_unrelated;
                }
            }
#endif
        }
    }

    if (entered_fastpath_branch) {
        fast_path_cl_id = *affected_clusters.begin();
        int u0 = fp_u, v0 = fp_v;

        std::unordered_set<int> visited_u{u0}, visited_v{v0};
        std::vector<int> frontier_u{u0}, frontier_v{v0};
        bool u_exhausted = false, v_exhausted = false;

        auto step_frontier = [&](std::vector<int>& frontier,
                                 std::unordered_set<int>& visited,
                                 std::unordered_set<int>& other_visited) -> bool {
            std::vector<int> next;
            for (int x : frontier) {
                for (int ei : g.incident_edges(x)) {
                    if (g.edge_removed(ei))
                        continue;
                    const Edge& ed = g.get_edges()[ei];
                    if (ed.u == ed.v)
                        continue;
                    int y = (ed.u == x ? ed.v : ed.u);
                    if (cluster[y] != fast_path_cl_id)
                        continue;
                    if (visited.count(y))
                        continue;

                    visited.insert(y);
                    next.push_back(y);
                }
            }
            frontier = std::move(next);
            return frontier.empty();
        };

        while (visited_u.size() <= cap && visited_v.size() <= cap && !u_exhausted && !v_exhausted) {
            u_exhausted = step_frontier(frontier_u, visited_u, visited_v);
            if (u_exhausted || visited_u.size() > cap)
                break;
            v_exhausted = step_frontier(frontier_v, visited_v, visited_u);
        }

        if (!u_exhausted && visited_u.size() <= cap) {
            while (!u_exhausted && visited_u.size() <= cap) {
                u_exhausted = step_frontier(frontier_u, visited_u, visited_v);
            }
        }
        if (!v_exhausted && visited_v.size() <= cap && visited_u.size() > cap) {
            while (!v_exhausted && visited_v.size() <= cap) {
                v_exhausted = step_frontier(frontier_v, visited_v, visited_u);
            }
        }

        bool u_small = u_exhausted && visited_u.size() <= cap;
        bool v_small = v_exhausted && visited_v.size() <= cap;

        if (u_small && !(v_small && visited_v.size() < visited_u.size())) {
            fp_small_side.assign(visited_u.begin(), visited_u.end());
            std::sort(fp_small_side.begin(),
                      fp_small_side.end());  // Keep row order independent of hash iteration.
            fp_small_is_u = true;
            fast_path_ok = true;
        } else if (v_small) {
            fp_small_side.assign(visited_v.begin(), visited_v.end());
            std::sort(fp_small_side.begin(),
                      fp_small_side.end());  // Keep row order independent of hash iteration.
            fp_small_is_u = false;
            fast_path_ok = true;
        }

        auto idx_it0 = cluster_points.find(fast_path_cl_id);
        if (fast_path_ok && idx_it0 != cluster_points.end() &&
            fp_small_side.size() >= idx_it0->second->size()) {
            fast_path_ok = false;
        }
#ifdef MIXDEX_PROFILE_FASTPATH
        {
            std::size_t max_affected_size = 0;
            for (auto& kv : cluster_points)
                max_affected_size = std::max(max_affected_size, kv.second->size());
            if (max_affected_size > 50) {
                if (fast_path_ok)
                    ++g_fp_reason_ok;
                else
                    ++g_fp_reason_cap_bothbig;
            }
        }
#endif
    }

    if (fast_path_ok) {
        (void)fp_small_is_u;

        int small_size = static_cast<int>(fp_small_side.size());
        bool small_has_edges = (small_size > 1);
        if (!small_has_edges) {
            int p = fp_small_side[0];
            if (core[p] > w)
                small_has_edges = true;
        }

        auto& full_pts = cl_members[fast_path_cl_id - n];
        int full_size = static_cast<int>(full_pts.size());
        int big_size = full_size - small_size;
        bool big_has_edges = big_size > 1;
        if (!big_has_edges && big_size == 1) {
            std::unordered_set<int> small_set(fp_small_side.begin(), fp_small_side.end());
            for (int p : full_pts) {
                if (!small_set.count(p)) {
                    if (core[p] > w)
                        big_has_edges = true;
                    break;
                }
            }
        }

        bool small_alive = small_has_edges && small_size >= min_cluster_size;
        if (!small_alive) {
            for (int p : fp_small_side) {
                shed_point(st, fast_path_cl_id, p, lambda);
            }
        }

        bool big_alive = big_has_edges && big_size >= min_cluster_size;

        if (small_alive && big_alive) {
            std::unordered_set<int> small_set(fp_small_side.begin(), fp_small_side.end());
            std::vector<int> big_pts;
            big_pts.reserve(big_size);
            for (int p : full_pts) {
                if (!small_set.count(p))
                    big_pts.push_back(p);
            }

            int small_id = allocate_cluster_id(next_cl);
            rows.push_back({fast_path_cl_id, small_id, lambda, small_size});
            lambda_birth.push_back(lambda);
            cl_members.push_back(fp_small_side);
            for (int p : cl_members.back()) cluster[p] = small_id;

            int big_id = allocate_cluster_id(next_cl);
            rows.push_back({fast_path_cl_id, big_id, lambda, big_size});
            lambda_birth.push_back(lambda);
            cl_members.push_back(std::move(big_pts));
            for (int p : cl_members.back()) cluster[p] = big_id;

            full_pts.clear();
            full_pts.shrink_to_fit();
        } else if (big_alive) {
            std::unordered_set<int> small_set(fp_small_side.begin(), fp_small_side.end());
            std::vector<int> big_pts;
            big_pts.reserve(big_size);
            for (int p : full_pts) {
                if (!small_set.count(p))
                    big_pts.push_back(p);
            }
            full_pts = std::move(big_pts);  // Retain the continuing cluster ID.
        } else if (small_alive) {
            std::unordered_set<int> small_set(fp_small_side.begin(), fp_small_side.end());
            for (int p : full_pts) {
                if (!small_set.count(p))
                    shed_point(st, fast_path_cl_id, p, lambda);
            }
            full_pts = fp_small_side;  // Retain the continuing cluster ID.
        } else {
            std::unordered_set<int> small_set(fp_small_side.begin(), fp_small_side.end());
            for (int p : full_pts) {
                if (!small_set.count(p))
                    shed_point(st, fast_path_cl_id, p, lambda);
            }
            full_pts.clear();
            full_pts.shrink_to_fit();
        }

        return true;
    }

    return false;
}

}  // namespace detail
}  // namespace mixdex
