param([ValidateSet("Release", "Debug")][string]$Configuration = "Release")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $pybind = python -m pybind11 --cmakedir
    if ($LASTEXITCODE -ne 0) { throw "Install pybind11 first." }
    cmake -S . -B build/windows -DUSE_OPENMP=OFF -DUSE_NATIVE=OFF "-Dpybind11_DIR=$pybind"
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed." }
    cmake --build build/windows --config $Configuration --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
} finally { Pop-Location }
