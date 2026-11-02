#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mixdex/types.hpp"

namespace mixdex {

namespace {

// Compare finite values exactly; -0.0 and +0.0 are equal.
bool row_equal(const double* a_num,
               const double* b_num,
               std::size_t n_num,
               const Code* a_cat,
               const Code* b_cat,
               std::size_t n_cat) {
    for (std::size_t k = 0; k < n_num; ++k) {
        if (a_num[k] != b_num[k])
            return false;
    }
    for (std::size_t k = 0; k < n_cat; ++k) {
        if (a_cat[k] != b_cat[k])
            return false;
    }
    return true;
}

std::uint64_t mix64(std::uint64_t z) {  // splitmix64 finalizer.
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

std::uint64_t row_hash(const double* num, std::size_t n_num, const Code* cat, std::size_t n_cat) {
    std::uint64_t h = 0;
    for (std::size_t k = 0; k < n_num; ++k) {
        double x = num[k];
        if (x == 0.0)
            x = 0.0;  // Equal signed zeros must hash identically.
        std::uint64_t bits;
        std::memcpy(&bits, &x, sizeof bits);
        h = mix64(h ^ bits);
    }
    for (std::size_t k = 0; k < n_cat; ++k)
        h = mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(cat[k])));
    return h;
}

}  // namespace

void Dataset::check_duplicates(DatasetPolicy policy,
                               const std::vector<unsigned char>& rows_dropped_already,
                               std::vector<unsigned char>* rows_duplicate) const {
    if (rows_duplicate)
        rows_duplicate->assign(n_, 0);
    if (n_ < 2)
        return;

    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();
    const bool skip_dropped = !rows_dropped_already.empty();

    // Hashes group candidates; coordinate comparisons confirm equality.
    // Skip missing rows already marked for removal.
    std::vector<std::pair<std::uint64_t, std::size_t>> keys;
    keys.reserve(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        if (skip_dropped && rows_dropped_already[i])
            continue;
        keys.push_back(
            {row_hash(
                 num_data_.data() + i * n_num_f, n_num_f, cat_data_.data() + i * n_cat_f, n_cat_f),
             i});
    }
    std::sort(keys.begin(), keys.end());

    for (std::size_t a = 0; a < keys.size();) {
        std::size_t b = a + 1;
        while (b < keys.size() && keys[b].first == keys[a].first) ++b;
        for (std::size_t p = a; p < b; ++p) {
            const std::size_t i = keys[p].second;
            if (rows_duplicate && (*rows_duplicate)[i])
                continue;
            for (std::size_t q = p + 1; q < b; ++q) {
                const std::size_t j = keys[q].second;
                if (row_equal(num_data_.data() + i * n_num_f,
                              num_data_.data() + j * n_num_f,
                              n_num_f,
                              cat_data_.data() + i * n_cat_f,
                              cat_data_.data() + j * n_cat_f,
                              n_cat_f)) {
                    const std::size_t lo = std::min(i, j), hi = std::max(i, j);
                    if (policy == DatasetPolicy::Strict) {
                        throw std::invalid_argument(
                            "Dataset: объекты-дубликаты (все координаты равны): строки " +
                            std::to_string(lo) + " и " + std::to_string(hi) +
                            "; один из них должен быть исключён до кластеризации (см. §3.1)");
                    }
                    // Keep the first original row and mark later duplicates.
                    if (rows_duplicate)
                        (*rows_duplicate)[hi] = 1;
                }
            }
        }
        a = b;
    }
}

void Dataset::drop_rows(const std::vector<unsigned char>& rows_missing,
                        const std::vector<unsigned char>& rows_duplicate) {
    const std::size_t n_num_f = n_num();
    const std::size_t n_cat_f = n_cat();

    const bool has_missing = !rows_missing.empty();
    const bool has_duplicate = !rows_duplicate.empty();
    if (!has_missing && !has_duplicate)
        return;

    std::size_t kept = 0;
    for (std::size_t i = 0; i < n_; ++i) {
        const bool drop = (has_missing && rows_missing[i]) || (has_duplicate && rows_duplicate[i]);
        if (drop) {
            if (has_missing && rows_missing[i])
                ++report_.rows_dropped_missing;
            else
                ++report_.rows_dropped_duplicate;
            continue;
        }
        if (kept != i) {
            std::copy(num_data_.begin() + static_cast<std::ptrdiff_t>(i * n_num_f),
                      num_data_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_num_f),
                      num_data_.begin() + static_cast<std::ptrdiff_t>(kept * n_num_f));
            std::copy(cat_data_.begin() + static_cast<std::ptrdiff_t>(i * n_cat_f),
                      cat_data_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_cat_f),
                      cat_data_.begin() + static_cast<std::ptrdiff_t>(kept * n_cat_f));
        }
        ++kept;
    }
    num_data_.resize(kept * n_num_f);
    cat_data_.resize(kept * n_cat_f);
    n_ = kept;
}

}  // namespace mixdex
