#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-fair}
case "$mode" in
    fair) args=(-DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DUSE_NATIVE=OFF) ;;
    fast) args=(-DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON -DUSE_NATIVE=ON) ;;
    debug) args=(-DCMAKE_BUILD_TYPE=Debug -DUSE_OPENMP=OFF -DUSE_NATIVE=OFF) ;;
    *) echo 'Usage: ./build.sh [fair|fast|debug]'; exit 2 ;;
esac
py=${PYTHON_EXECUTABLE:-python3}
pybind_dir=$("$py" -m pybind11 --cmakedir)
cmake -S . -B "build/$mode" "${args[@]}" -DPython3_EXECUTABLE="$py" -Dpybind11_DIR="$pybind_dir"
cmake --build "build/$mode" --config "$(test "$mode" = debug && echo Debug || echo Release)" --parallel
printf 'Module directory: %s\n' "$PWD/build/$mode"
