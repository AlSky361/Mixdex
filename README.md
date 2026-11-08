# Mixdex

C++20 and Python library for neighbor search, clustering and outlier detection
on mixed numeric, categorical and set-valued data. The Python catalog layer adds
stable record IDs, filters, logical removal and persistent snapshots.

## Input model

Each row describes one object; columns have fixed meanings across all rows.

| Feature | Raw Python value | Distance component |
| :--- | :--- | :--- |
| Numeric, continuous or discrete | Finite number | Absolute difference after fitted scaling |
| Scalar categorical | String | 0 for equal labels, 1 otherwise |
| Set categorical | Collection of strings | Jaccard distance; two empty sets have distance 0 |

Each feature has a fixed weight. The final distance is

```text
d(x, y) = sum_j weight[j] * component_distance[j](x[j], y[j]) / sum_j weight[j]
```

Weights must be finite and nonnegative, with a positive finite total and finite
reciprocal total. Zero-weight features are removed. In exact arithmetic, positive
retained weights produce a metric on complete representations; different record IDs can have
the same representation and distance zero.

Use `CatalogSnapshot` for raw row matrices and external IDs. Use native `Dataset`
for prepared numeric values and nonnegative integer category codes. Category
codes represent equality, not an order or numeric magnitude; vocabularies are
independent for each column. Set columns must be declared explicitly.

## Data preparation

`CatalogSnapshot.build` fits category encoders and numeric scales, removes
constant or zero-weight columns, and groups identical representations while
preserving their IDs. Missing active values exclude the affected source records
by default. An empty set is a known value, distinct from a missing value.

Optional `impute=True` fills numbers with type-specific medians, scalar
categories with modes, and sets with frequent elements. Fitted fills, dictionaries
and scales are reused for queries; they do not change on each search.

Native `Dataset` uses `Strict` to reject missing values, duplicates and constants,
or `Clean` to remove them. It does not automatically encode labels, impute values
or fit scales. Fit a `Scaler` once and reuse it for stored and external rows.

## What you can compute

| Task | Entry points |
| :--- | :--- |
| Nearest neighbors and radius queries | `Index`, `knn`, `knn_query`, `radius_query_points` |
| Filtered catalog retrieval | `CatalogSnapshot.search`, `search_by_id`, `prepare_filter` |
| Core distances and minimum spanning trees | `core_distances`, `boruvka`, `dt_boruvka` |
| Density clustering | `dbscan`, `hdbscan` |
| Nearest-neighbor classification | `knn_classify`, `knn_predict` |
| Outlier scores | `knn_outlier_scores`, `LOF`, HDBSCAN GLOSH scores |
| Reusable preprocessing | `LabelEncoder`, `SetEncoder`, `Imputer`, `Scaler` |

Recommendation ranking, business rules and rebuild scheduling belong to the
application. Logical catalog removal filters retrieval; it does not update an
existing MST or clustering hierarchy.

## Installation and first example

Python 3.10+, a C++20 compiler and CMake 3.20+ are required for a source build.
On Windows install the Visual Studio C++ workload. OpenMP is optional and off by
default. NumPy is the only runtime Python dependency.

macOS/Linux, from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

Run examples using that environment's Python. Installation builds a wheel for
the current Python ABI, OS and architecture.

```python
from mixdex_catalog import CatalogSnapshot

catalog = CatalogSnapshot.build(
    ids=[101, 102, 103],
    numeric=[[2000, 90], [2010, 110], [2020, 100]],
    categorical=[[['drama']], [['drama', 'thriller']], [['comedy']]],
    numeric_names=['year', 'duration'],
    categorical_names=['genres'],
    categorical_set_columns=[0],
    weights=[1, 1, 2],
)
print(catalog.search_by_id(101, k=2)['items'])  # (record_id, distance) pairs.
```

C++ consumers use the installed CMake package:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX)
find_package(mixdex CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE mixdex::mixdex)
```

```cpp
#include <mixdex/index/index.hpp>
#include <mixdex/types.hpp>

int main() {
    mixdex::Dataset data({}, {0, 1}, 2, 0, 1,
                         {mixdex::FeatureType::Categorical}, {}, {"kind"});
    mixdex::Index index(data);
    const auto neighbors = index.search().knn(0, 1);
    return neighbors.size() == 1 && neighbors[0].first == 1 ? 0 : 1;
}
```

See [build and installation](docs/api.md#build) for C++-only builds, package
installation, tests and platform-specific settings.

## Contracts and limitations

- Native search results use Dataset row positions; catalog results use external
  signed 64-bit IDs. Internal indices are signed 32-bit values, not a tested
  dataset-capacity promise.
- Keep the Dataset alive while its native index is in use. Rebuild after changing
  weights, dictionaries or representations. Catalog geometry changes through a
  new snapshot; `remove/restore` changes only logical activity.
- Exact HDBSCAN requires BallTree. Approximate mode uses a restricted graph and
  can return different results. Search speed depends on the data and workload.
- A saved catalog contains source data and fitted transforms. Loading rebuilds
  the index and checks preprocessing compatibility; it does not restore a native
  binary image.

CI targets macOS, Windows and Linux. Check the
[CI runs](https://github.com/AlSky361/Mixdex/actions/workflows/ci.yml) for the result
of each revision.

Detailed reading:

- [API, examples and contracts](docs/api.md)
- [Distance, lower bounds and quantiles](docs/distance-bounds.md)
- [Scientific references and implementation scope](docs/references.md)

## License

[BSD 3-Clause](LICENSE). Third-party notices are preserved separately in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
