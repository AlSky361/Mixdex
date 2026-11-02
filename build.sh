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
if [[ $(uname -s) == Darwin ]]; then
    export DEVELOPER_DIR=${DEVELOPER_DIR:-$(/usr/bin/xcode-select -p)}
    # Resolve SDK and compiler shims through the same developer directory.
    sdk=${SDKROOT:-$(/usr/bin/xcrun --no-cache --sdk macosx --show-sdk-path)}
    if [[ ! -d "$sdk" ]]; then
        printf 'macOS SDK directory does not exist: %s\n' "$sdk" >&2
        exit 1
    fi
    args+=("-DCMAKE_OSX_SYSROOT:PATH=$sdk")
    printf 'macOS SDK: %s\n' "$sdk"
fi
py=${PYTHON_EXECUTABLE:-python3}
pybind_dir=$("$py" -m pybind11 --cmakedir)
cmake -S . -B "build/$mode" "${args[@]}" -DPython3_EXECUTABLE="$py" -Dpybind11_DIR="$pybind_dir"
cmake --build "build/$mode" --config "$(test "$mode" = debug && echo Debug || echo Release)" --parallel
printf 'Module directory: %s\n' "$PWD/build/$mode"
