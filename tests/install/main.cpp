#include <mixdex/distance.hpp>
#include <mixdex/types.hpp>

int main() {
    mixdex::Dataset data({}, {0, 1}, 2, 0, 1, {mixdex::FeatureType::Categorical}, {}, {"genres"});
    data.set_category_sets(0, {{10}, {10, 20}});
    return mixdex::distance(data, 0, 1) == 0.5 ? 0 : 1;
}
