// Python entry point; registrations live in bind_*.cpp and share common.hpp.
#include "common.hpp"

PYBIND11_MODULE(mixdex, m) {
    m.doc() = "mixdex bindings";

    namespace mp = mixdex::pybind;
    mp::bind_data(m);  // Register enums and Dataset before dependent bindings.
    mp::bind_encoding(m);
    mp::bind_density(m);
    mp::bind_mst(m);
    mp::bind_index(m);
    mp::bind_algorithms(m);
    mp::bind_hdbscan(m);
}
