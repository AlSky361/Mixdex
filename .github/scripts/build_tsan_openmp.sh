#!/usr/bin/env bash
set -euo pipefail

destination="${1:?Usage: build_tsan_openmp.sh DESTINATION}"
version=19.1.7
mkdir -p "$destination/source"

for component in openmp cmake; do
    curl --fail --location --retry 3 \
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/$component-$version.src.tar.xz" \
        --output "$destination/$component.tar.xz"
done
printf '%s  %s\n' \
    bd7e6901ab086fd268750363017935fd4a717c153dad3c2aab86cb0140d9e3fe "$destination/openmp.tar.xz" \
    11c5a28f90053b0c43d0dec3d0ad579347fc277199c005206b963c19aae514e3 "$destination/cmake.tar.xz" \
    | sha256sum --check

tar -xJf "$destination/openmp.tar.xz" -C "$destination/source"
mkdir -p "$destination/source/cmake"
tar -xJf "$destination/cmake.tar.xz" -C "$destination/source/cmake" --strip-components=1

# Track runtime atomics; keep full memory instrumentation in Mixdex itself.
cmake -S "$destination/source/openmp-$version.src" -B "$destination/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-clang-19}" \
    -DCMAKE_CXX_COMPILER="${CXX:-clang++-19}" \
    -DCMAKE_CXX_FLAGS='-fsanitize=thread -mllvm -tsan-instrument-memory-accesses=false' \
    -DLIBOMP_OMPT_SUPPORT=ON -DLIBOMP_OMPD_SUPPORT=OFF \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_INSTALL_PREFIX="$destination/install"
cmake --build "$destination/build" -j 4
cmake --install "$destination/build"
test -f "$destination/install/lib/libomp.so"
test -f "$destination/install/lib/libarcher.so"
