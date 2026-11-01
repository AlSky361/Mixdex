// Excess-of-mass (EOM) cluster selection; construction lives in cond_tree.cpp.
#include <algorithm>
#include <vector>

#include "mixdex/algorithms/hdbscan/cond_tree.hpp"
#include "mixdex/utils/constants.hpp"

namespace mixdex {
namespace {

// Cluster indices are id - n; the root index is zero.
struct ClusterTree {
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
};

// Stability sums lifetime contributions, including size-weighted child departures.
std::vector<double> compute_stability(const std::vector<CondRow>& rows,
                                      const std::vector<double>& lambda_birth,
                                      int n,
                                      int n_cl) {
    std::vector<double> stability(n_cl, 0.0);

    for (const auto& r : rows) {
        if (r.child_cl >= n)
            continue;
        const int ci = r.parent_cl - n;
        if (ci < 0 || ci >= n_cl)
            continue;
        stability[ci] += (r.lambda - lambda_birth[ci]);
    }

    for (const auto& r : rows) {
        if (r.child_cl < n)
            continue;
        const int parent = r.parent_cl - n;
        if (parent >= 0 && parent < n_cl)
            stability[parent] += (r.lambda - lambda_birth[parent]) * r.child_cl_size;
    }
    return stability;
}

ClusterTree build_cluster_tree(const std::vector<CondRow>& rows, int n, int n_cl) {
    ClusterTree tree{std::vector<int>(n_cl, -1), std::vector<std::vector<int>>(n_cl)};
    for (const auto& r : rows) {
        if (r.child_cl < n)
            continue;
        const int chi = r.child_cl - n;
        const int pai = r.parent_cl - n;
        if (chi >= 0 && chi < n_cl && pai >= 0 && pai < n_cl && tree.parent[chi] == -1) {
            tree.parent[chi] = pai;
            tree.children[pai].push_back(chi);
        }
    }
    return tree;
}

// Bottom-up EOM compares parent stability with the selected child subtrees.
// Ties keep the parent; the root is excluded.
std::vector<bool> select_clusters_eom(const std::vector<double>& stability,
                                      const ClusterTree& tree) {
    const int n_cl = static_cast<int>(stability.size());
    std::vector<bool> selected(n_cl, false);
    std::vector<double> eff_stab(stability);

    std::vector<int> order;
    std::vector<int> stack = {0};
    while (!stack.empty()) {
        const int ci = stack.back();
        stack.pop_back();
        order.push_back(ci);
        for (int ch : tree.children[ci]) stack.push_back(ch);
    }
    std::reverse(order.begin(), order.end());

    eff_stab[0] = 0.0;
    for (int ci : order) {
        if (ci == 0)
            continue;
        double child_sum = 0.0;
        for (int ch : tree.children[ci]) child_sum += eff_stab[ch];

        if (stability[ci] >= child_sum) {
            selected[ci] = true;
            eff_stab[ci] = stability[ci];

            std::vector<int> descendants = tree.children[ci];
            while (!descendants.empty()) {
                const int c = descendants.back();
                descendants.pop_back();
                selected[c] = false;
                for (int cc : tree.children[c]) descendants.push_back(cc);
            }
        } else {
            eff_stab[ci] = child_sum;
        }
    }
    return selected;
}

// Assign selected-cluster labels in ID order and align their stabilities.
std::vector<int> number_selected_clusters(const std::vector<bool>& selected,
                                          const std::vector<double>& stability,
                                          std::vector<double>& out_stabilities) {
    const int n_cl = static_cast<int>(selected.size());
    std::vector<int> cl_id_to_label(n_cl, -1);
    out_stabilities.clear();
    for (int ci = 0; ci < n_cl; ++ci) {
        if (selected[ci]) {
            cl_id_to_label[ci] = static_cast<int>(out_stabilities.size());
            out_stabilities.push_back(stability[ci]);
        }
    }
    return cl_id_to_label;
}

// Use the nearest selected ancestor; otherwise mark the point as noise.
std::vector<int> label_points(const std::vector<int>& final_cluster,
                              int n,
                              const std::vector<bool>& selected,
                              const std::vector<int>& cl_parent,
                              const std::vector<int>& cl_id_to_label) {
    const int n_cl = static_cast<int>(selected.size());
    std::vector<int> labels(n, constants::kNoiseLabel);

    std::vector<int> resolved_cache(n_cl, -2);
    std::vector<int> path;
    for (int p = 0; p < n; ++p) {
        int cl = final_cluster[p] - n;
        path.clear();
        int result = -1;
        while (cl >= 0) {
            if (selected[cl]) {
                result = cl;
                break;
            }
            if (resolved_cache[cl] != -2) {
                result = resolved_cache[cl];
                break;
            }
            path.push_back(cl);
            cl = cl_parent[cl];
        }
        for (int c : path) resolved_cache[c] = result;

        if (result >= 0)
            labels[p] = cl_id_to_label[result];
    }
    return labels;
}

// Canonical labels follow first point occurrence, independent of tied MST choices.
// Apply the same permutation to stabilities; keep condensed-tree IDs unchanged.
void canonicalize_labels(std::vector<int>& labels, std::vector<double>& stabilities) {
    const int n_labels = static_cast<int>(stabilities.size());
    std::vector<int> remap(n_labels, -1);
    int next_label = 0;
    for (int l : labels) {
        if (l != constants::kNoiseLabel && remap[l] < 0)
            remap[l] = next_label++;
    }
    for (int l = 0; l < n_labels; ++l) {  // Retain a slot for an unexpected empty cluster.
        if (remap[l] < 0)
            remap[l] = next_label++;
    }
    std::vector<double> stab_canonical(n_labels);
    for (int l = 0; l < n_labels; ++l) stab_canonical[remap[l]] = stabilities[l];
    stabilities.swap(stab_canonical);
    for (int& l : labels) {
        if (l != constants::kNoiseLabel)
            l = remap[l];
    }
}

}  // namespace

void extract_clusters(const std::vector<CondRow>& rows,
                      const std::vector<double>& lambda_birth,
                      int n,
                      int next_cl,
                      const std::vector<int>& final_cluster,
                      std::vector<int>& out_labels,
                      std::vector<double>& out_stabilities) {
    const int n_cl = next_cl - n;

    const std::vector<double> stability = compute_stability(rows, lambda_birth, n, n_cl);
    const ClusterTree tree = build_cluster_tree(rows, n, n_cl);
    const std::vector<bool> selected = select_clusters_eom(stability, tree);
    const std::vector<int> cl_id_to_label =
        number_selected_clusters(selected, stability, out_stabilities);

    out_labels = label_points(final_cluster, n, selected, tree.parent, cl_id_to_label);
    canonicalize_labels(out_labels, out_stabilities);
}

}  // namespace mixdex
