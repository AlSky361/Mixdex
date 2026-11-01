#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "mixdex/distance.hpp"
#include "mixdex/index/ball_tree/ball_tree.hpp"
#include "mixdex/types.hpp"
using namespace mixdex;
void check(bool b) {
    if (!b)
        throw std::runtime_error("weight regression");
}
Dataset make() {
    return Dataset({0, 10, 0, 20, 1, 30},
                   {0, 3, 0, 4, 1, 5},
                   3,
                   2,
                   2,
                   {FeatureType::Numeric,
                    FeatureType::Numeric,
                    FeatureType::Categorical,
                    FeatureType::Categorical},
                   {NumericType::Continuous, NumericType::Discrete},
                   {"n0", "n1", "c0", "c1"});
}
int main() {
    for (auto w : std::vector<std::vector<double>>{
             {-1, 1, 1, 1},
             {0, 0, 0, 0},
             {1, 2},
             {NAN, 1, 1, 1},
             {INFINITY, 1, 1, 1},
             {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 1, 1}}) {
        auto d = make();
        bool rejected = false;
        try {
            d.set_weights(w);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && d.n_cols() == 4 && d.n_rows() == 3 && !d.has_weights());
    }
    auto d = make();
    d.set_weights({2, 0, 3, -0.0});
    check(d.n_num() == 1 && d.n_cat() == 1 && d.n_rows() == 2);
    check(d.num_data() == std::vector<double>({0, 1}) && d.cat_data() == std::vector<Code>({0, 1}));
    check(d.feat_name(0) == "n0" && d.feat_name(1) == "c0" && d.categories_size(0) == 2);
    check(d.weight(0) == 2 && d.weight(1) == 3 && d.inv_weights_sum() == .2);
    check(d.preprocess_report().rows_dropped_duplicate == 1);
    check(d.preprocess_report().columns_dropped_zero_weight_idx == std::vector<int>({1, 3}));
    d.validate();
    check(distance(d, 0, 1) == 1.0);
    BallTree tree(d, 1);
    check(tree.knn(0, 1).at(0).first == 1);
    d.set_weights({0, 4});
    check(d.n_num() == 0 && d.n_cat() == 1 && d.weight(0) == 4);
    d.validate();
    auto n = make();
    n.set_weights({0, 5, 0, 0});
    check(n.n_num() == 1 && n.n_cat() == 0 && n.num_type(0) == NumericType::Discrete);
    check(n.num_data() == std::vector<double>({10, 20, 30}));
    n.validate();
    std::cout << "Weight validation, projection, metadata, duplicate removal and kNN passed\n";
}
