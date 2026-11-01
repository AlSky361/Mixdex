#include "common.hpp"

#include <stdexcept>

namespace mixdex::pybind {

void check_index(int index, int count) {
    if (index < 0 || index >= count)
        throw std::invalid_argument("index out of range");
}

QueryBatch make_query_batch(const Dataset& data, const py::object& X_num, const py::object& X_cat) {
    QueryBatch b;
    const std::size_t nn = data.n_num();
    const std::size_t no = data.n_cat();
    bool have_m = false;
    auto take_m = [&](std::size_t rows) {
        if (have_m && rows != b.m)
            throw std::invalid_argument("X_num and X_cat have different numbers of rows");
        b.m = rows;
        have_m = true;
    };
    if (nn > 0) {
        if (X_num.is_none())
            throw std::invalid_argument("X_num is required: dataset has numeric features");
        b.num = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X_num);
        if (!b.num || b.num.ndim() != 2 || static_cast<std::size_t>(b.num.shape(1)) != nn)
            throw std::invalid_argument("X_num must be a 2D array with n_num columns");
        take_m(static_cast<std::size_t>(b.num.shape(0)));
    } else if (!X_num.is_none()) {
        auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(X_num);
        if (!a || a.ndim() != 2 || a.shape(1) != 0)
            throw std::invalid_argument("dataset has no numeric features: X_num must be None");
        take_m(static_cast<std::size_t>(a.shape(0)));
    }
    if (no > 0) {
        if (X_cat.is_none())
            throw std::invalid_argument("X_cat is required: dataset has categorical features");
        b.cat = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(X_cat);
        if (!b.cat || b.cat.ndim() != 2 || static_cast<std::size_t>(b.cat.shape(1)) != no)
            throw std::invalid_argument("X_cat must be a 2D array with n_cat columns");
        take_m(static_cast<std::size_t>(b.cat.shape(0)));
    } else if (!X_cat.is_none()) {
        auto a = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(X_cat);
        if (!a || a.ndim() != 2 || a.shape(1) != 0)
            throw std::invalid_argument("dataset has no categorical features: X_cat must be None");
        take_m(static_cast<std::size_t>(a.shape(0)));
    }
    if (!have_m)
        throw std::invalid_argument("empty dataset schema");
    return b;
}

}  // namespace mixdex::pybind
