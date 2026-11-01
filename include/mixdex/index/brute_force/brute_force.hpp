#pragma once
#include <vector>

#include "mixdex/density/mreach/mreach.hpp"
#include "mixdex/distance.hpp"
#include "mixdex/index/neighbours.hpp"
#include "mixdex/types.hpp"
#include "mixdex/utils/fm_heap.hpp"

namespace mixdex {

/// Scan all rows using a borrowed Dataset and the supplied metric.
template <typename Metric>
class BruteForceSearch : public MutualReachabilitySearch {
public:
    BruteForceSearch(const Dataset& dataset, Metric metric_fn) : data(dataset), metric(metric_fn) {
    }

    std::vector<std::pair<int, double>> knn(int query_idx, int k) const override;

    std::vector<std::pair<int, double>> radius_query(int query_idx, double r) const override;

    std::vector<std::pair<int, double>> knn(const RowView& q, int k) const override;
    std::vector<std::pair<int, double>> radius_query(const RowView& q, double r) const override;

    const Dataset& dataset() const noexcept override {
        return data;
    }
    void check_external_ready() const override {
        if (!core_.empty())
            throw std::invalid_argument("external queries are not supported in mreach mode");
    }

    int n_rows() const noexcept override {
        return data.n_rows();
    }

    void set_core_dist(const std::vector<double>& cd) override {
        check_core_distances(cd);
        core_ = cd;
    }

    const Dataset& data;

private:
    Metric metric;
    std::vector<double> core_;

    double dist(int i, int j) const {
        if (core_.empty())
            return metric(i, j);
        return mreach_distance(core_, i, j, metric);
    }
};

}  // namespace mixdex

#include "mixdex/index/brute_force/brute_force.ipp"
