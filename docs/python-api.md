# Python quick start

Install from the repository root with `python -m pip install .` using a C++20
compiler. Python 3.12 is the tested configuration. This installs three modules:

| Module | Use |
|---|---|
| `mixdex` | Native Dataset, distances, search, graphs and clustering |
| `mixdex_catalog` | Raw records, stable IDs, fitted preprocessing, filters and snapshots |
| `mixdex_preprocessing` | Standalone fitted imputation |

For a catalog or recommendation application, start with [CATALOG.md](../CATALOG.md).
The catalog returns candidates; application ranking and business rules remain outside
this library. The examples below demonstrate the lower-level native API.

## Prepare mixed data

Numeric and categorical buffers are separate, both in row-major order. Category
codes compare by equality: their numeric difference has no meaning. Codes must be
nonnegative; numeric values must be finite. The native Python Dataset constructor
rejects duplicate rows and constant features. For cleanup or imputation of raw
records, use CatalogSnapshot instead.

```python
import numpy as np
import mixdex as mx

raw = mx.Dataset(
    [0., 0., .2, .1, .4, .3, 4., 4., 4.2, 4.1, 4.4, 4.3],
    [0, 0, 0, 1, 1, 1],
    6, 2, 1,
    [mx.FeatureType.Numeric, mx.FeatureType.Numeric, mx.FeatureType.Categorical],
    [mx.NumericType.Continuous, mx.NumericType.Continuous],
    ["x", "y", "kind"],
)
scaler = mx.Scaler.fit(raw)
data = scaler.transform(raw)
index = mx.Index(data)
neighbors = mx.knn(index, 0, k=2)
assert len(neighbors) == 2 and all(row != 0 for row, _ in neighbors)
```

`Dataset` does not fit numeric scales automatically. Fit once and reuse the same
Scaler for stored and external rows. Keep the Dataset alive while using its native
index; rebuild the index after changing weights, dictionaries or representations.
Set weights before fitting/indexing. Weights must be finite and nonnegative with
positive total weight; zero weights remove features and can merge duplicate rows.
See [distance-bounds.md](distance-bounds.md) for the distance and bound assumptions.

Native results identify zero-based Dataset row positions, not external catalog IDs.
Neighbor results are `(row, distance)` pairs sorted by distance, then row index.
Indexed queries exclude their own row. External queries have no row to exclude;
an identical stored row can be returned at distance zero.

## Query new rows

External batches are two-dimensional arrays: numeric `float64`, categorical `int32`.
Their columns must match the retained Dataset schema. Use `None` for an absent
block. These low-level calls require complete, encoded and scaled values; they do
not apply a catalog transform or fill missing values.

```python
query_num = scaler.transform_rows(np.array([[.1, .05]], dtype=np.float64))
query_cat = np.array([[0]], dtype=np.int32)
batches = mx.knn_query(index, query_num, query_cat, k=2)
assert len(batches) == 1 and len(batches[0]) == 2
```

## Clustering and outliers

`min_pts`/`m_pts` count the query point itself. The `k` in indexed kNN and LOF
counts other rows. Noise is labeled `-1`; other cluster labels are nonnegative.
An epsilon threshold uses the final scaled, weighted distance.

```python
labels = mx.dbscan(index, eps=.1, min_pts=2)
lof = mx.LOF(index, k=2)
result = mx.hdbscan(
    data, m_pts=2, min_cluster_size=2,
    mode=mx.HDBSCANMode.Exact, method=mx.SearchMethod.BallTree,
)
assert len(labels) == len(lof.scores) == len(result.labels) == 6
assert len(result.outlier_scores) == 6
assert len(result.cluster_stabilities) == len(set(result.labels) - {-1})
```

Exact HDBSCAN currently requires BallTree. Approximate mode uses a restricted kNN
graph and can differ from Exact; automatic k selection can fail if no connected
graph is found before its search limit. It is not an accuracy-equivalent shortcut.
`labels`, `outlier_scores` and `cluster_stabilities` are result attributes.

## Other entry points

| Operation | Python call / result |
|---|---|
| Pairwise distance | `mx.distance(data, i, j)` |
| Radius search | `mx.radius_query_points(index, query_num, query_cat, r)` |
| Core distances | `mx.core_distances(index, m_pts)` |
| Mutual-reachability MST | `mx.dt_boruvka(data, core)` → edges with `u`, `v`, `w` |
| kNN classification | `mx.knn_classify(index, labels, k)`; `-1` labels are ignored |
| Prediction for new rows | `mx.knn_predict(index, labels, query_num, query_cat, k)` |
| kNN outlier score | `mx.knn_outlier_scores(index, k)` |
| New-row LOF scores | `lof.score(query_num, query_cat)` |

Most search-based functions accept either Index or Dataset. Dataset overloads build
an index for that call; reuse Index for repeated calls. HDBSCAN accepts Dataset.
Detailed overload signatures and defaults are available in Python:

```python
help(mx.Dataset)
help(mx.knn)
help(mx.hdbscan)
```

Invalid native arguments generally raise `mx.InvalidArgument`, a ValueError
subclass. Catalog validation raises ValueError. Range and allocation failures can
raise other exceptions; successful small examples do not establish memory capacity.
For set-valued records, use CatalogSnapshot or attach canonical set dictionaries
before indexing as shown in the repository README.
