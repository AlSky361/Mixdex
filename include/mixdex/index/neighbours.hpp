#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>

#include "mixdex/types.hpp"
#include "mixdex/utils/parallel.hpp"

namespace mixdex {

enum class SearchMethod { BallTree, BruteForce };

enum class MetricType { Composite, MReach };

inline void check_k(int k) {
    if (k < 1)
        throw std::invalid_argument("k must be >= 1");
}
inline void check_radius(double r) {
    if (std::isnan(r))
        throw std::invalid_argument("radius must not be NaN");
}

inline void sort_neighbours(std::vector<std::pair<int, double>>& v) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
    });
}

/// Shared query interface; external rows must use the indexed schema and transform.
class NeighbourSearch {
public:
    virtual ~NeighbourSearch() = default;

    // Mask is immutable for the duration of this query. Exclusions are row positions.
    virtual std::vector<std::pair<int, double>> knn_filtered(
        const RowView& q,
        int k,
        std::span<const unsigned char> active,
        std::span<const int> excluded = {}) const;

    virtual std::vector<std::pair<int, double>> knn(int query_idx, int k) const = 0;

    virtual std::vector<std::pair<int, double>> radius_query(int query_idx, double r) const = 0;

    virtual bool has_neighbours_within(int query_idx, double r, int at_least) const {
        check_radius(r);
        return at_least <= 0 || static_cast<int>(radius_query(query_idx, r).size()) >= at_least;
    }

    virtual void radius_ids(int query_idx, double r, std::vector<int>& out) const {
        out.clear();
        for (const auto& nb : radius_query(query_idx, r)) out.push_back(nb.first);
    }

    virtual double kth_distance(int query_idx, int k) const {
        const auto nb = knn(query_idx, k);
        return nb.empty() ? 0.0 : nb.back().second;
    }

    std::vector<std::vector<std::pair<int, double>>> knn_batch(int k) const {
        check_k(k);
        const int n = n_rows();
        std::vector<std::vector<std::pair<int, double>>> out(n);
        parallel_for(n, [&](int i) {
            out[i] = knn(i, k);
        });
        return out;
    }

    virtual int n_rows() const noexcept = 0;

    virtual const Dataset& dataset() const noexcept = 0;

    virtual std::vector<std::pair<int, double>> knn(const RowView& q, int k) const = 0;

    virtual std::vector<std::pair<int, double>> radius_query(const RowView& q, double r) const = 0;

    virtual double kth_distance(const RowView& q, int k) const {
        const auto nb = knn(q, k);
        return nb.empty() ? 0.0 : nb.back().second;
    }

    virtual void check_external_ready() const {
    }

    std::vector<double> kth_distance_all(int k) const {
        check_k(k);
        const int n = n_rows();
        std::vector<double> out(n);
        parallel_for(n, [&](int i) {
            out[i] = kth_distance(i, k);
        });
        return out;
    }

    std::vector<double> kth_distance_batch(std::span<const double> num,
                                           std::span<const int> cat,
                                           std::size_t m,
                                           int k) const {
        check_k(k);
        validate_batch(num, cat, m);
        std::vector<double> out(m);
        parallel_for(m, [&](long long i) {
            out[i] = kth_distance(batch_row(num, cat, static_cast<std::size_t>(i)), k);
        });
        return out;
    }

    void validate_batch(std::span<const double> num,
                        std::span<const int> cat,
                        std::size_t m) const {
        const Dataset& d = dataset();
        if (num.size() != m * static_cast<std::size_t>(d.n_num()) ||
            cat.size() != m * static_cast<std::size_t>(d.n_cat()))
            throw std::invalid_argument("query batch does not match the dataset schema");
        check_external_ready();
        for (std::size_t i = 0; i < m; ++i) check_row(batch_row(num, cat, i));
    }

    std::vector<std::vector<std::pair<int, double>>> knn_query_batch(std::span<const double> num,
                                                                     std::span<const int> cat,
                                                                     std::size_t m,
                                                                     int k) const {
        check_k(k);
        validate_batch(num, cat, m);
        std::vector<std::vector<std::pair<int, double>>> out(m);
        parallel_for(m, [&](long long i) {
            out[i] = knn(batch_row(num, cat, static_cast<std::size_t>(i)), k);
        });
        return out;
    }

    std::vector<std::vector<std::pair<int, double>>> radius_query_batch(std::span<const double> num,
                                                                        std::span<const int> cat,
                                                                        std::size_t m,
                                                                        double r) const {
        check_radius(r);
        validate_batch(num, cat, m);
        std::vector<std::vector<std::pair<int, double>>> out(m);
        parallel_for(m, [&](long long i) {
            out[i] = radius_query(batch_row(num, cat, static_cast<std::size_t>(i)), r);
        });
        return out;
    }

protected:
    void check_core_distances(std::span<const double> values) const {
        if (values.size() != static_cast<std::size_t>(n_rows()))
            throw std::invalid_argument("one core distance per row is required");
        for (double value : values)
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("core distances must be finite and nonnegative");
    }
    void check_row(const RowView& q) const {
        const Dataset& d = dataset();
        if (q.num.size() != static_cast<std::size_t>(d.n_num()) ||
            q.cat.size() != static_cast<std::size_t>(d.n_cat()))
            throw std::invalid_argument("query row does not match the dataset schema");
        if (!q.cat_sets.empty() && q.cat_sets.size() != q.cat.size())
            throw std::invalid_argument("query set block does not match categorical columns");
        for (const auto& values : q.cat_sets) {
            if (!std::is_sorted(values.begin(), values.end()) ||
                std::adjacent_find(values.begin(), values.end()) != values.end() ||
                (!values.empty() && values.front() < 0))
                throw std::invalid_argument(
                    "query sets must have sorted distinct nonnegative codes");
        }
        for (double v : q.num)
            if (!std::isfinite(v))
                throw std::invalid_argument("query row contains NaN or inf");
    }

private:
    RowView batch_row(std::span<const double> num, std::span<const int> cat, std::size_t i) const {
        const std::size_t nn = dataset().n_num();
        const std::size_t no = dataset().n_cat();
        return {num.subspan(i * nn, nn), cat.subspan(i * no, no)};
    }
};

class MutualReachabilitySearch : public NeighbourSearch {
public:
    virtual void set_core_dist(const std::vector<double>& cd) = 0;
};

std::unique_ptr<NeighbourSearch> make_search(const Dataset&, SearchMethod, int leaf_size);
std::unique_ptr<MutualReachabilitySearch> make_mreach_search(const Dataset&,
                                                             SearchMethod,
                                                             int leaf_size);

}  // namespace mixdex
