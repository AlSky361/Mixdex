#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mixdex/utils/aliases.hpp"

namespace mixdex {

/// Sort and deduplicate labels; sorted inputs need only a linear pass.
std::vector<std::string> canonicalize_category_set(std::vector<std::string> labels);

/// Fitted scalar labels; query encoding never changes the dictionary.
class LabelEncoder {
public:
    static LabelEncoder fit(std::vector<std::string> labels);
    Code encode(const std::string& label) const;
    std::vector<Code> transform(const std::vector<std::string>& labels) const;
    Code encode_query(const std::string& label) const;
    const std::string& decode(Code code) const;
    const std::vector<std::string>& classes() const noexcept {
        return classes_;
    }

private:
    std::vector<std::string> classes_;
    std::unordered_map<std::string, Code> codes_;
};

struct EncodedSetQuery {
    Code code;                  // Whole-set code, or classes().size() for a novel set.
    std::vector<Code> members;  // Unknown member codes are local to this query.
};

/// Fitted whole sets and member labels; distinct sets have distinct stored codes.
class SetEncoder {
public:
    static SetEncoder fit(std::vector<std::vector<std::string>> sets);
    Code encode(std::vector<std::string> labels) const;
    std::vector<Code> transform(const std::vector<std::vector<std::string>>& sets) const;
    EncodedSetQuery encode_query(std::vector<std::string> labels) const;
    const std::vector<std::string>& decode(Code code) const;
    const std::vector<std::vector<std::string>>& classes() const noexcept {
        return classes_;
    }
    const std::vector<std::string>& member_classes() const noexcept {
        return members_.classes();
    }
    const std::vector<std::vector<Code>>& dictionary() const noexcept {
        return dictionary_;
    }

private:
    std::vector<std::vector<std::string>> classes_;
    std::unordered_map<std::string, Code> codes_;
    LabelEncoder members_;
    std::vector<std::vector<Code>> dictionary_;
};

}  // namespace mixdex
