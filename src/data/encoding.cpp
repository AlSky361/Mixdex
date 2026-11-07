#include "mixdex/encoding.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "mixdex/utils/checked_arithmetic.hpp"

namespace mixdex {
namespace {

// Length prefixes distinguish empty labels, delimiters and embedded zero bytes.
std::string set_key(const std::vector<std::string>& labels) {
    std::string key;
    for (const auto& label : labels) {
        key += std::to_string(label.size());
        key += ':';
        key += label;
    }
    return key;
}

const std::string& scalar_key(const std::string& label) {
    return label;
}

// Hash first, then sort only unique classes; repeated labels do not add a log(n) factor.
template <class T, class Key>
std::vector<T> canonical_classes(std::vector<T> values, Key key) {
    const bool sorted = std::is_sorted(values.begin(), values.end());
    if (sorted && std::adjacent_find(values.begin(), values.end()) == values.end())
        return values;
    std::vector<T> result;
    if (sorted) {
        for (auto& value : values)
            if (result.empty() || result.back() != value)
                result.push_back(std::move(value));
    } else {
        std::unordered_set<std::string> seen;
        for (auto& value : values)
            if (seen.insert(key(value)).second)
                result.push_back(std::move(value));
        std::sort(result.begin(), result.end());
    }
    return result;
}

Code query_code(const std::unordered_map<std::string, Code>& codes, const std::string& key) {
    const auto it = codes.find(key);
    return it == codes.end() ? detail::checked_index_size(codes.size()) : it->second;
}

Code known_code(const std::unordered_map<std::string, Code>& codes, const std::string& key) {
    const auto it = codes.find(key);
    if (it == codes.end())
        throw std::invalid_argument("category was not observed during fitting");
    return it->second;
}

template <class T>
const T& decoded(const std::vector<T>& classes, Code code) {
    if (code < 0 || static_cast<std::size_t>(code) >= classes.size())
        throw std::out_of_range("category code out of range");
    return classes[static_cast<std::size_t>(code)];
}

template <class T, class Key>
std::unordered_map<std::string, Code> make_codes(const std::vector<T>& classes, Key key) {
    detail::checked_index_size(classes.size());
    std::unordered_map<std::string, Code> codes;
    codes.reserve(classes.size());
    for (std::size_t i = 0; i < classes.size(); ++i)
        codes.emplace(key(classes[i]), static_cast<Code>(i));
    return codes;
}

}  // namespace

std::vector<std::string> canonicalize_category_set(std::vector<std::string> labels) {
    return canonical_classes(std::move(labels), scalar_key);
}

LabelEncoder LabelEncoder::fit(std::vector<std::string> labels) {
    labels = canonical_classes(std::move(labels), scalar_key);
    LabelEncoder result;
    result.codes_ = make_codes(labels, scalar_key);
    result.classes_ = std::move(labels);
    return result;
}

Code LabelEncoder::encode(const std::string& label) const {
    return known_code(codes_, label);
}

std::vector<Code> LabelEncoder::transform(const std::vector<std::string>& labels) const {
    std::vector<Code> result;
    result.reserve(labels.size());
    for (const auto& label : labels) result.push_back(encode(label));
    return result;
}

Code LabelEncoder::encode_query(const std::string& label) const {
    return query_code(codes_, label);
}

const std::string& LabelEncoder::decode(Code code) const {
    return decoded(classes_, code);
}

SetEncoder SetEncoder::fit(std::vector<std::vector<std::string>> sets) {
    for (auto& labels : sets) labels = canonicalize_category_set(std::move(labels));
    sets = canonical_classes(std::move(sets), set_key);
    SetEncoder result;
    result.codes_ = make_codes(sets, set_key);
    std::vector<std::string> labels;
    for (const auto& values : sets) labels.insert(labels.end(), values.begin(), values.end());
    result.members_ = LabelEncoder::fit(std::move(labels));
    result.dictionary_.reserve(sets.size());
    for (const auto& values : sets) result.dictionary_.push_back(result.members_.transform(values));
    result.classes_ = std::move(sets);
    return result;
}

Code SetEncoder::encode(std::vector<std::string> labels) const {
    return known_code(codes_, set_key(canonicalize_category_set(std::move(labels))));
}

std::vector<Code> SetEncoder::transform(const std::vector<std::vector<std::string>>& sets) const {
    std::vector<Code> result;
    result.reserve(sets.size());
    for (const auto& labels : sets) result.push_back(encode(labels));
    return result;
}

EncodedSetQuery SetEncoder::encode_query(std::vector<std::string> labels) const {
    labels = canonicalize_category_set(std::move(labels));
    EncodedSetQuery result{query_code(codes_, set_key(labels)), {}};
    result.members.reserve(labels.size());
    std::size_t next_unknown = members_.classes().size();
    for (const auto& label : labels) {
        Code code = members_.encode_query(label);
        if (static_cast<std::size_t>(code) == members_.classes().size()) {
            code = detail::checked_index_size(next_unknown);
            ++next_unknown;
        }
        result.members.push_back(code);
    }
    if (!std::is_sorted(result.members.begin(), result.members.end()))
        std::sort(result.members.begin(), result.members.end());
    return result;
}

const std::vector<std::string>& SetEncoder::decode(Code code) const {
    return decoded(classes_, code);
}

}  // namespace mixdex
