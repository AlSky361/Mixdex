#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "mixdex/distance.hpp"
#include "mixdex/encoding.hpp"
#include "mixdex/types.hpp"

namespace {
using namespace mixdex;
void check(bool value) {
    if (!value)
        throw std::runtime_error("encoding regression");
}
template <class E, class F>
void rejects(F operation) {
    try {
        operation();
    } catch (const E&) {
        return;
    }
    throw std::runtime_error("expected encoding rejection");
}

void scalar_labels() {
    auto encoder = LabelEncoder::fit({"b", "a", "b", ""});
    check(encoder.classes() == std::vector<std::string>({"", "a", "b"}));
    check(encoder.transform({"b", "", "a"}) == std::vector<Code>({2, 0, 1}));
    check(encoder.decode(1) == "a");
    check(encoder.encode_query("new") == 3);
    rejects<std::invalid_argument>([&] {
        encoder.encode("new");
    });
    rejects<std::out_of_range>([&] {
        encoder.decode(-1);
    });
    rejects<std::out_of_range>([&] {
        encoder.decode(3);
    });
    check(encoder.classes().size() == 3);
    auto unicode = LabelEncoder::fit({"\xf0\x9f\x98\x80", "\xce\xb1", "a", "\xc3\xa9"});
    check(unicode.classes() ==
          std::vector<std::string>({"a", "\xc3\xa9", "\xce\xb1", "\xf0\x9f\x98\x80"}));
    auto empty = LabelEncoder::fit({});
    check(empty.classes().empty() && empty.encode_query("x") == 0);
    rejects<std::invalid_argument>([&] {
        empty.encode("x");
    });
}

void set_labels() {
    auto encoder = SetEncoder::fit({{"b", "a", "a"}, {}, {"a"}, {"b", "a"}});
    check(encoder.classes() == std::vector<std::vector<std::string>>({{}, {"a"}, {"a", "b"}}));
    check(encoder.member_classes() == std::vector<std::string>({"a", "b"}));
    check(encoder.dictionary() == std::vector<std::vector<Code>>({{}, {0}, {0, 1}}));
    check(encoder.encode({"b", "a", "b"}) == 2);
    check(encoder.decode(2) == std::vector<std::string>({"a", "b"}));
    rejects<std::invalid_argument>([&] {
        encoder.encode({"a", "new"});
    });
    auto query = encoder.encode_query({"z", "a", "new", "z"});
    check(query.code == 3 && query.members == std::vector<Code>({0, 2, 3}));
    check(encoder.encode_query({}).code == 0);
    check(encoder.encode_query({"b", "a"}).code == 2);
    check(encoder.classes().size() == 3 && encoder.member_classes().size() == 2);

    Dataset data({}, {0, 1, 2}, 3, 0, 1, {FeatureType::Categorical}, {}, {"set"});
    data.set_category_sets(0, encoder.dictionary());
    auto partial = encoder.encode_query({"a", "z"});
    check(data.categorical_distance(0, partial.members, 1) == 0.5);
    check(data.categorical_distance(0, partial.members, 2) == 2.0 / 3.0);
    check(distance(data, 1, 2) == 0.5);

    std::vector<std::vector<std::string>> values{
        {"ab", "c"}, {"a", "bc"}, {""}, {}, {"1:a"}, {"1", "a"}, {std::string("a\0b", 3)}};
    auto delimiters = SetEncoder::fit(values);
    check(delimiters.classes().size() == values.size());
    for (const auto& value : values)
        check(delimiters.decode(delimiters.encode(value)) == canonicalize_category_set(value));
    auto empty = SetEncoder::fit({});
    check(empty.encode_query({}).code == 0 && empty.encode_query({}).members.empty());
    rejects<std::out_of_range>([&] {
        encoder.decode(3);
    });
}
}  // namespace
int main() {
    scalar_labels();
    set_labels();
}
