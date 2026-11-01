cmake_minimum_required(VERSION 3.20)
project(mixdex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_OPENMP "Enable OpenMP" OFF)
option(MIXDEX_BUILD_PYTHON "Build Python bindings" ON)
option(USE_NATIVE "Enable -march=native" OFF)
option(USE_O3 "Enable -O3 instead of -O2" ON)

if(USE_OPENMP)
    include(cmake/mixdexOpenMP.cmake)
endif()

if(MIXDEX_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# The static library is linked into a Python shared module; Linux requires PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mixdex_lib STATIC
    src/types.cpp
    src/quantile.cpp
    src/scale.cpp
    src/distance.cpp
    src/synthetic/synthetic.cpp
    
    src/utils/fm_heap.cpp
    src/utils/graph.cpp
    src/utils/uf.cpp
    src/utils/uf_members.cpp

    src/algorithms/dbscan/dbscan.cpp
    src/density/hierarchy.cpp
    src/algorithms/knn/knn_classifier.cpp
    src/algorithms/outlier/knn_outlier.cpp
    src/algorithms/outlier/lof.cpp
    src/algorithms/hdbscan/cond_tree.cpp
    src/algorithms/hdbscan/cond_tree_extract.cpp
    src/algorithms/hdbscan/cond_tree_fastpath.cpp
    src/algorithms/hdbscan/cond_tree_slowpath.cpp
    src/algorithms/hdbscan/glosh.cpp
    src/algorithms/hdbscan/hdbscan.cpp
    
    src/index/ball_tree.cpp
    src/index/factory.cpp
    src/index/index.cpp
    src/index/filtered.cpp
    
    src/mst/dt_boruvka.cpp
    src/mst/boruvka.cpp
    src/mst/mst_compare.cpp
)

target_include_directories(mixdex_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(mixdex_lib PUBLIC Threads::Threads)
target_compile_features(mixdex_lib PUBLIC cxx_std_20)
set_target_properties(mixdex_lib PROPERTIES EXPORT_NAME mixdex)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS mixdex_lib EXPORT mixdexTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT CppDevelopment)
install(DIRECTORY include/mixdex DESTINATION include COMPONENT CppDevelopment)
install(EXPORT mixdexTargets NAMESPACE mixdex:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mixdex
    COMPONENT CppDevelopment)
configure_package_config_file(cmake/mixdexConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mixdexConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mixdex)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/mixdexConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES cmake/mixdexOpenMP.cmake ${CMAKE_CURRENT_BINARY_DIR}/mixdexConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/mixdexConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mixdex COMPONENT CppDevelopment)

if(MSVC)
    # Use UTF-8 source encoding; prevent Windows min/max macros.
    target_compile_options(mixdex_lib PRIVATE
        /W4 /permissive- /utf-8
        $<$<CONFIG:Release>:/O2>
    )
    target_compile_definitions(mixdex_lib PUBLIC NOMINMAX)
else()
    if(USE_O3)
        target_compile_options(mixdex_lib PRIVATE $<$<CONFIG:Release>:-O3>)
    else()
        target_compile_options(mixdex_lib PRIVATE $<$<CONFIG:Release>:-O2>)
    endif()

    if(USE_NATIVE)
        target_compile_options(mixdex_lib PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()

    target_compile_options(mixdex_lib PRIVATE $<$<CONFIG:Debug>:-O0 -g>)

    target_compile_options(mixdex_lib PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -Wconversion
        -Wsign-compare
        -Wno-unused-parameter
        # Keep signed internal indices; Dataset checks INT_MAX. Suppress Clang sign-conversion noise.
        -Wno-sign-conversion
    )

    target_compile_options(mixdex_lib PRIVATE
        $<$<CONFIG:Debug>:
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        >
    )
    # Propagate sanitizer runtimes to executables linking the static library.
    target_link_options(mixdex_lib PUBLIC
        $<$<CONFIG:Debug>:
            -fsanitize=address,undefined
        >
    )
endif()

if(USE_OPENMP)
    if(MSVC)
        target_compile_options(mixdex_lib PUBLIC ${MIXDEX_MSVC_OPENMP_FLAG})
    else()
        target_link_libraries(mixdex_lib PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

if(MIXDEX_BUILD_PYTHON)
pybind11_add_module(mixdex
    python/bindings/bindings.cpp
    python/bindings/common.cpp
    python/bindings/bind_data.cpp
    python/bindings/bind_density.cpp
    python/bindings/bind_index.cpp
    python/bindings/bind_mst.cpp
    python/bindings/bind_algorithms.cpp
    python/bindings/bind_hdbscan.cpp
)

# Python catalog facade is distributed next to the extension module.
configure_file(python/mixdex_catalog.py ${CMAKE_CURRENT_BINARY_DIR}/mixdex_catalog.py COPYONLY)
configure_file(python/mixdex_preprocessing.py ${CMAKE_CURRENT_BINARY_DIR}/mixdex_preprocessing.py COPYONLY)

target_link_libraries(mixdex PRIVATE
    mixdex_lib
)
target_compile_options(mixdex PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

set_target_properties(mixdex PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(mixdex PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}")
endforeach()
install(TARGETS mixdex LIBRARY DESTINATION . COMPONENT PythonRuntime
    RUNTIME DESTINATION . COMPONENT PythonRuntime)
install(FILES python/mixdex_catalog.py python/mixdex_preprocessing.py
    DESTINATION . COMPONENT PythonRuntime)
endif()

# Enable C++ tests with -DMIXDEX_BUILD_TESTS=ON; run with ctest.
option(MIXDEX_BUILD_TESTS "Build C++ tests" OFF)
if(MIXDEX_BUILD_TESTS)
    enable_testing()
    add_executable(test_graph tests/cpp/test_graph.cpp)
    target_link_libraries(test_graph PRIVATE mixdex_lib)
    add_test(NAME graph COMMAND test_graph)
    add_executable(test_tree_value_semantics tests/cpp/test_tree_value_semantics.cpp)
    target_link_libraries(test_tree_value_semantics PRIVATE mixdex_lib)
    add_test(NAME tree_value_semantics COMMAND test_tree_value_semantics)
    add_executable(test_strided_span tests/cpp/test_strided_span.cpp)
    target_link_libraries(test_strided_span PRIVATE mixdex_lib)
    add_test(NAME strided_span COMMAND test_strided_span)
    add_executable(test_parallel tests/cpp/test_parallel.cpp)
    target_link_libraries(test_parallel PRIVATE mixdex_lib)
    add_test(NAME parallel COMMAND test_parallel)
    add_executable(test_uf tests/cpp/test_uf.cpp)
    target_link_libraries(test_uf PRIVATE mixdex_lib)
    add_test(NAME uf COMMAND test_uf)
    add_executable(test_rollback_uf tests/cpp/test_rollback_uf.cpp)
    target_link_libraries(test_rollback_uf PRIVATE mixdex_lib)
    add_test(NAME rollback_uf COMMAND test_rollback_uf)
    add_executable(test_quantile tests/cpp/test_quantile.cpp)
    target_link_libraries(test_quantile PRIVATE mixdex_lib)
    add_test(NAME quantile COMMAND test_quantile)
    add_executable(test_scale_validation tests/cpp/test_scale_validation.cpp)
    target_link_libraries(test_scale_validation PRIVATE mixdex_lib)
    add_test(NAME scale_validation COMMAND test_scale_validation)
    add_executable(test_dt_boruvka tests/cpp/test_dt_boruvka.cpp)
    target_link_libraries(test_dt_boruvka PRIVATE mixdex_lib)
    add_test(NAME dt_boruvka COMMAND test_dt_boruvka)
    add_executable(test_duplicates tests/cpp/test_duplicates.cpp)
    target_link_libraries(test_duplicates PRIVATE mixdex_lib)
    add_test(NAME duplicates COMMAND test_duplicates)
    add_executable(test_weights tests/cpp/test_weights.cpp)
    target_link_libraries(test_weights PRIVATE mixdex_lib)
    add_test(NAME weights COMMAND test_weights)
    add_executable(test_cond_tree_snapshot tests/cpp/test_cond_tree_snapshot.cpp)
    target_link_libraries(test_cond_tree_snapshot PRIVATE mixdex_lib)
    add_test(NAME cond_tree_snapshot COMMAND test_cond_tree_snapshot)
    add_executable(test_cond_tree_mcs1 tests/cpp/test_cond_tree_mcs1.cpp)
    target_link_libraries(test_cond_tree_mcs1 PRIVATE mixdex_lib)
    add_test(NAME cond_tree_mcs1 COMMAND test_cond_tree_mcs1)
    add_executable(test_mst_determinism tests/cpp/test_mst_determinism.cpp)
    target_link_libraries(test_mst_determinism PRIVATE mixdex_lib)
    add_test(NAME mst_determinism COMMAND test_mst_determinism)
    add_executable(test_hdbscan_determinism tests/cpp/test_hdbscan_determinism.cpp)
    target_link_libraries(test_hdbscan_determinism PRIVATE mixdex_lib)
    add_test(NAME hdbscan_determinism COMMAND test_hdbscan_determinism)
    add_executable(test_parallel_algorithms tests/cpp/test_parallel_algorithms.cpp)
    target_link_libraries(test_parallel_algorithms PRIVATE mixdex_lib)
    add_test(NAME parallel_algorithms COMMAND test_parallel_algorithms)

    # Compile each header independently to catch hidden include-order dependencies.
    file(GLOB_RECURSE MIXDEX_HEADERS RELATIVE ${CMAKE_SOURCE_DIR}/include
         ${CMAKE_SOURCE_DIR}/include/mixdex/*.hpp ${CMAKE_SOURCE_DIR}/include/mixdex/*.ipp)
    set(MIXDEX_HEADER_CHECK_SOURCES "")
    foreach(h ${MIXDEX_HEADERS})
        string(MAKE_C_IDENTIFIER ${h} h_id)
        set(gen ${CMAKE_BINARY_DIR}/header_check/${h_id}.cpp)
        file(WRITE ${gen} "#include \"${h}\"\n")
        list(APPEND MIXDEX_HEADER_CHECK_SOURCES ${gen})
    endforeach()
    add_library(mixdex_header_check OBJECT ${MIXDEX_HEADER_CHECK_SOURCES})
    target_link_libraries(mixdex_header_check PRIVATE mixdex_lib)
endif()
