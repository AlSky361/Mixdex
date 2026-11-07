#include <mixdex/distance.hpp>
#include <mixdex/encoding.hpp>
#include <mixdex/types.hpp>

int main() {
    mixdex::Dataset data({}, {0, 1}, 2, 0, 1, {mixdex::FeatureType::Categorical}, {}, {"genres"});
    auto scalar = mixdex::LabelEncoder::fit({"b", "a"});
    if (scalar.encode("a") != 0 || scalar.decode(1) != "b")
        return 2;
    auto sets = mixdex::SetEncoder::fit({{"a"}, {"a", "b"}});
    data.set_category_sets(0, sets.dictionary());
    return mixdex::distance(data, 0, 1) == 0.5 ? 0 : 1;
}
