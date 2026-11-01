# mixdex

C++ and Python tools for exact neighbor search and density-based analysis of
mixed data. The Python catalog API adds stable IDs, query filters, logical
removal, snapshot persistence and replacement.

This is a pre-release source tree. CI has passed on macOS, Windows (MSVC)
and Linux, including source builds, tests and installed-wheel checks.
See [CI runs](https://github.com/AlSky361/Mixdex/actions/workflows/ci.yml)
for the result of each revision.

## Distance

The ordinary distance combines numeric L1, categorical Hamming and set Jaccard:

    distance(x, y) = sum_j weight[j] * component_distance[j](x[j], y[j]) / sum_j weight[j]

Every component has its own feature indices. Numeric features use the fitted
scale; categorical values use equality. Set features use Jaccard, including
empty-to-empty distance zero. Positive retained weights define a metric on
representations, not necessarily on distinct external IDs.

The old gower_* API is renamed to distance*; update imports and header includes
to `mixdex/distance.hpp`. Historical experiment reports retain the old names.

For set features, encode each distinct set as a categorical code and attach its
dictionary before building the index:

```python
import mixdex

d = mixdex.Dataset([], [0, 1, 2], 3, 0, 1,
                   [mixdex.FeatureType.Categorical], [], ["genres"])
d.set_category_sets(0, [[], [10], [10, 20]])
print(mixdex.distance(d, 1, 2))  # 0.5
index = mixdex.Index(d)
print(mixdex.knn(index, 1, 2))
```

Dictionary sets are canonicalized and must be distinct; members are nonnegative
integer labels. Rebuild indices after changing dictionaries. Unregistered query
codes represent isolated unknown values, not arbitrary new sets. For the catalog facade declare `categorical_set_columns=[0]` and pass collections
of string labels in that column; see CATALOG.md. External set queries may contain
new combinations and members. Unknown data and known empty sets are different.

## Python installation

Run these commands from the repository root with Python 3.12, CMake >= 3.20
and a C++20 compiler installed. On Windows, use the Visual Studio C++ workload.

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

Use that environment's Python for your scripts and tests. The library is imported
by a Python program; it does not start a server or open a user interface.
See [Python quick start](docs/python-api.md) for complete native-API examples and
[Catalog API](CATALOG.md) for raw records, stable IDs and filters.

This builds a platform-specific wheel. It does not provide a universal binary
or publish anything to PyPI. Only NumPy is a runtime Python dependency.

## Build from source

Install a C++20 compiler, CMake >= 3.20, Python development files and dependencies:

```sh
python -m pip install -r requirements-test.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMIXDEX_BUILD_TESTS=ON -DUSE_OPENMP=OFF -DUSE_NATIVE=OFF -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

PowerShell configuration (Visual Studio C++ workload required):

```powershell
$pybind = python -m pybind11 --cmakedir
cmake -S . -B build -DMIXDEX_BUILD_TESTS=ON -DUSE_OPENMP=OFF -DUSE_NATIVE=OFF "-Dpybind11_DIR=$pybind"
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
$env:MIXDEX_BUILD_DIR = "$PWD/build"
python -m pytest -q
```

On macOS/Linux set `MIXDEX_BUILD_DIR=$PWD/build` before pytest. Add the build
directory to PYTHONPATH to import both mixdex and mixdex_catalog in applications.
`./build.sh` and `./build.ps1` are convenience wrappers; builds do not format
sources or delete another configuration. C++-only builds use
`-DMIXDEX_BUILD_PYTHON=OFF` and do not require Python or pybind11.

OpenMP is optional. For macOS install libomp and enable `USE_OPENMP`; for Linux
use a compiler with the matching OpenMP runtime. MSVC task-based builds require
an OpenMP implementation supporting tasks; start with the sequential build.
For heavily oversubscribed OpenMP tests, use OMP_WAIT_POLICY=PASSIVE and
KMP_BLOCKTIME=0 to avoid expensive spinning. Native CPU flags are off by default. Python packaging uses scikit-build-core; wheels are specific to the Python ABI,
OS and architecture used to build them.

## C++ installation

```sh
cmake --install build --config Release --prefix /your/prefix --component CppDevelopment
```

Consumers use `find_package(mixdex CONFIG REQUIRED)` and link `mixdex::mixdex`.
The exported target carries C++20 and thread/OpenMP dependencies. A relocatable
install is tested with the separate project under `tests/install`.

## Catalog API

See [CATALOG.md](CATALOG.md) for build, search_by_id, filters, remove/restore,
save/load and SnapshotStore. The library retrieves candidates; it is not a
complete recommendation service. Numeric missing values and negative category
codes are rejected or removed according to Dataset policy. Zero-weight columns
are removed and resulting duplicate rows are merged.

## Development

Optional fitted imputation is available in the Python preprocessing layer:
`CatalogSnapshot.build(..., impute=True, numeric_types=['Continuous', 'Discrete'])`.
It uses type-specific numeric medians, scalar category modes and frequent-element
sets whose size is the discrete median of known set sizes. Parameters are fixed
for queries and stored with snapshots. See [imputation.md](docs/imputation.md)
for the standalone Imputer and fully missing column behavior.

Tests cover exact search, bounds, weights, preprocessing, density algorithms
and snapshots. Build outputs and editor caches are excluded from version control.

License selection remains open at the author's request.
