#include "common.hpp"
#include "mixdex/encoding.hpp"

namespace mixdex::pybind {
namespace {

std::string read_label(py::handle value) {
    if (!py::isinstance<py::str>(value))
        throw std::invalid_argument("categorical labels must be strings");
    return py::cast<std::string>(value);
}

std::vector<std::string> read_labels(py::iterable values) {
    if (py::isinstance<py::str>(values) || py::isinstance<py::bytes>(values))
        throw std::invalid_argument("a collection of labels is required");
    std::vector<std::string> result;
    for (auto value : values) result.push_back(read_label(value));
    return result;
}

std::vector<std::vector<std::string>> read_sets(py::iterable values) {
    if (py::isinstance<py::str>(values) || py::isinstance<py::bytes>(values))
        throw std::invalid_argument("a collection of sets is required");
    std::vector<std::vector<std::string>> result;
    for (auto value : values) {
        if (!py::isinstance<py::iterable>(value))
            throw std::invalid_argument("each set must be a collection of labels");
        result.push_back(read_labels(py::reinterpret_borrow<py::iterable>(value)));
    }
    return result;
}

}  // namespace

void bind_encoding(py::module_& m) {
    m.def(
        "canonicalize_category_set",
        [](py::iterable values) {
            auto labels = read_labels(values);
            py::gil_scoped_release release;
            return canonicalize_category_set(std::move(labels));
        },
        py::arg("labels"));

    py::class_<EncodedSetQuery>(m, "EncodedSetQuery")
        .def_readonly("code", &EncodedSetQuery::code)
        .def_readonly("members", &EncodedSetQuery::members);

    py::class_<LabelEncoder>(m, "LabelEncoder")
        .def_static(
            "fit",
            [](py::iterable values) {
                auto labels = read_labels(values);
                py::gil_scoped_release release;
                return LabelEncoder::fit(std::move(labels));
            },
            py::arg("labels"))
        .def(
            "encode",
            [](const LabelEncoder& encoder, py::handle value) {
                return encoder.encode(read_label(value));
            },
            py::arg("label"))
        .def(
            "transform",
            [](const LabelEncoder& encoder, py::iterable values) {
                auto labels = read_labels(values);
                py::gil_scoped_release release;
                return encoder.transform(labels);
            },
            py::arg("labels"))
        .def(
            "encode_query",
            [](const LabelEncoder& encoder, py::handle value) {
                return encoder.encode_query(read_label(value));
            },
            py::arg("label"))
        .def(
            "decode",
            [](const LabelEncoder& encoder, Code code) {
                return encoder.decode(code);
            },
            py::arg("code"))
        .def_property_readonly("classes", [](const LabelEncoder& encoder) {
            return encoder.classes();
        });

    py::class_<SetEncoder>(m, "SetEncoder")
        .def_static(
            "fit",
            [](py::iterable values) {
                auto sets = read_sets(values);
                py::gil_scoped_release release;
                return SetEncoder::fit(std::move(sets));
            },
            py::arg("sets"))
        .def(
            "encode",
            [](const SetEncoder& encoder, py::iterable values) {
                auto labels = read_labels(values);
                py::gil_scoped_release release;
                return encoder.encode(std::move(labels));
            },
            py::arg("labels"))
        .def(
            "transform",
            [](const SetEncoder& encoder, py::iterable values) {
                auto sets = read_sets(values);
                py::gil_scoped_release release;
                return encoder.transform(sets);
            },
            py::arg("sets"))
        .def(
            "encode_query",
            [](const SetEncoder& encoder, py::iterable values) {
                auto labels = read_labels(values);
                py::gil_scoped_release release;
                return encoder.encode_query(std::move(labels));
            },
            py::arg("labels"))
        .def(
            "decode",
            [](const SetEncoder& encoder, Code code) {
                return encoder.decode(code);
            },
            py::arg("code"))
        .def_property_readonly("classes",
                               [](const SetEncoder& encoder) {
                                   return encoder.classes();
                               })
        .def_property_readonly("member_classes",
                               [](const SetEncoder& encoder) {
                                   return encoder.member_classes();
                               })
        .def_property_readonly("dictionary", [](const SetEncoder& encoder) {
            return encoder.dictionary();
        });
}

}  // namespace mixdex::pybind
