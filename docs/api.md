# API and contracts

Start with the [README](../README.md) for installation, the input model and a
minimal example. This reference covers Python and C++ contracts; it does not
require reading the internal implementation.

- [Native data, queries and algorithms](#native-data-and-algorithms)
- [Catalog snapshots, filters and persistence](#catalog)
- [Categorical encoding](#categorical-encoding)
- [Fitted imputation](#imputation)
- [Quantiles and scaling](#quantiles-and-scaling)
- [Numeric types and errors](#numeric-types-and-errors)
- [C++ data views](#data-views)
- [Build, installation and tests](#build)

<a id="native-data-and-algorithms"></a>

## Native data and algorithms

Install from the repository root with `python -m pip install .` using a C++20
compiler and Python 3.10 or newer. This installs four modules:

| Module | Use |
|---|---|
| `mixdex` | Native Dataset, distances, search, graphs and clustering |
| `mixdex_catalog` | Raw records, stable IDs, fitted preprocessing, filters and snapshots |
| `mixdex_preprocessing` | Standalone fitted imputation |
| `mixdex_encoding` | Native scalar/set encoders and category normalization |

For raw records and stable IDs, start with [catalog snapshots](#catalog).
The catalog returns candidates; application ranking and business rules remain outside
this library. The examples below demonstrate the lower-level native API.

### Prepare mixed data

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

For standalone quantiles, `mx.quantiles` requires sorted, finite values;
`mx.quantiles_select` accepts an unsorted input and works on a copy.
Definitions, validation and complexity: [quantiles and scaling](#quantiles-and-scaling).

Native results identify zero-based Dataset row positions, not external catalog IDs.
Neighbor results are `(row, distance)` pairs sorted by distance, then row index.
Indexed queries exclude their own row. External queries have no row to exclude;
an identical stored row can be returned at distance zero.

Categorical label encoders are available in both C++ and Python; see [categorical encoding](api.md#categorical-encoding).

### Query new rows

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

### Clustering and outliers

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

### Other entry points

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
before indexing as shown below.

### Native set dictionaries

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
of string labels in that column; see [catalog snapshots](#catalog). External set queries may contain
new combinations and members. Unknown data and known empty sets are different.

<a id="catalog"></a>

## Catalog snapshots

`mixdex_catalog` wraps the native index with stable IDs, immutable geometry,
query filters, logical activity and JSON persistence. It retrieves candidates;
personalization, availability rules and scheduling belong to the application.

```python
from mixdex_catalog import CatalogSnapshot, SnapshotStore

snapshot = CatalogSnapshot.build(
    ids=[1001, 1002, 1003],
    numeric=[[2000, 90], [2010, 105], [2020, 120]],
    categorical=[["comedy", "US"], ["drama", "FR"], ["drama", "US"]],
    numeric_names=["year", "duration"],
    categorical_names=["genre", "country"],
    weights=[1, 1, 2, 1],
    version="catalog-1",
)
snapshot.save("catalog.json")
store = SnapshotStore(CatalogSnapshot.load("catalog.json"))
print(store.search_by_id(1001, k=2))
store.remove([1002])
store.restore([1002])
```

### Data contract

- External IDs are unique signed 64-bit Python integers. Internal row indices
  remain signed 32-bit values; this is a representation limit, not tested capacity.
- Numeric columns default to continuous; `numeric_types` accepts Continuous/Discrete
  enum values or names, one per original numeric column. Scalar categories are strings. Declare set-valued
  categorical column indices with `categorical_set_columns=[...]`; each value is
  a list, tuple, set or frozenset of strings. Values are canonicalized, with empty
  sets distinct from `None` (missing). Columns use independent dictionaries.
- Missing significant values exclude a record from the index, not from the source
  catalog. `report()` records source-to-row mappings, duplicate groups, exclusions,
  dropped columns and fitted transforms. Zero-weight and constant columns are
  removed. Negative/nonfinite weights or a nonpositive total or nonfinite reciprocal total are rejected.
- Equal representations share a point while retaining separate record IDs. Signed
  zeros compare equal. Scaling can create additional exact duplicate points.
- With no informative features, usable records share distance zero. Check
  `stats()["informative_features"]` before interpreting results as meaningful.

### Queries

Opt-in `build(..., impute=True)` fits fills on the original catalog before cleanup:
type-specific numeric medians, scalar modes, and frequent-element sets with a
per-column discrete median size. Known empty sets count as size zero; missing
sets do not count. Fully missing positive-weight columns raise. Zero-weight
columns remain ignored. See [fitted imputation](api.md#imputation).
Missing query values use the same fitted fills, including `search_by_id`.
Fills appear in `report()['transform']['imputation']`; remove/restore never refits.

```python
sets = CatalogSnapshot.build(
    [1, 2, 3], categorical=[[['drama']], [['drama', 'thriller']], [[]]],
    categorical_names=['genres'], categorical_set_columns=[0],
)
print(sets.search(categorical=[{'thriller', 'mystery'}], k=3))
```

Queries compare actual sets even when the combination or individual labels were
not in the snapshot. No dictionary mutation or brute-force fallback is needed.
Sets match by their members, not by equality of their encoded combination.

`search(numeric=(), categorical=(), k=10, exclude_ids=(), allowed_ids=None)` uses
raw values in the original schema. Unknown category strings differ from stored
labels; without imputation, missing significant query values raise an exception. The service decides
how to handle an unusable query.

`search_by_id(id, k=10, **filters)` uses a source record and excludes its own ID.
Other IDs with the same representation remain eligible. Source records excluded
from the index return an empty result with a reason.

Results contain `items` as `(record_id, distance)` pairs, `version`, `snapshot_id`,
`activity_id` and `incomplete`. Ordering is by distance, then external ID, including
boundary ties. Filters apply during candidate selection and to each grouped ID.
Fewer than k eligible records produces a shorter list with `incomplete=True`.

`allowed_ids=None` allows all active IDs; an empty collection allows none.
Unknown query-filter IDs are ignored. `remove/restore` validate all IDs before
mutation; unknown IDs reject the call. Repeating a valid operation is harmless.

### Reusable filters

```python
allowed = snapshot.prepare_filter([1001, 1003])
result = snapshot.search_by_id(1001, k=10, allowed_ids=allowed)
```

A prepared filter belongs to one exact snapshot instance. Its fixed ID collection
maps to internal rows and caches one activity mask. Preparation plus the first
mask costs O(V+A) time/memory for V points and A allowed IDs. Repeated queries reuse
the mask until activity changes. Exclusions inspect affected groups. A new snapshot
requires a new prepared filter; the application controls how many filters it retains.

### Updates and persistence

Coordinates remain unchanged until rebuilding. `remove/restore` publish a new
activity state, preserving deleted centers as geometric representatives. A query
holds one consistent activity state. Use mutations through `SnapshotStore` when
replacing snapshots: `replace` transfers current activity for shared IDs so a
background build cannot undo intervening removals or restorations.

An in-flight reader can finish against an old snapshot. If immediate restrictions
must apply at response time, the service checks availability again before display.
This API does not update an existing MST or HDBSCAN hierarchy after removal.

`rebuild(version=None, compact=False)` returns an independent snapshot of the
source data. Compaction permanently omits inactive IDs. Use `build` with a new
catalog to introduce new or changed record records.

`save` writes a `mixdex.catalog.v1` envelope for legacy scalar-only schemas, or
`mixdex.catalog.v2` when only set columns are declared. Explicit numeric types
or imputation use `mixdex.catalog.v3`, including fitted preprocessing parameters.
All formats contain source data,
transforms, activity and a SHA-256 checksum. A same-directory temporary file is
flushed and replaced atomically. This is not an authentication mechanism or a
power-loss durability guarantee. No pickle or binary pointers are stored.

`load` checks format, schema and checksum, rebuilds the tree and verifies that the
reproduced transform matches the saved one. It does not restore native binary
state. Save/load require temporary space and memory for reconstruction; measure
peak usage with both snapshots alive. Checksums do not make untrusted files safe
against excessive resource consumption.

Category dictionaries are fitted by the native encoders exposed through
`mixdex_encoding`; [categorical encoding](api.md#categorical-encoding) describes their two-level set
representation and query-only unknown codes. Snapshot formats remain v1/v2/v3.

<a id="categorical-encoding"></a>

## Categorical encoding

`mixdex/encoding.hpp` provides standalone C++ `LabelEncoder` and `SetEncoder`.
Python exposes the same native objects through `mixdex` and `mixdex_encoding`.
No Python runtime is required by the C++ implementation.

Each encoder fits one categorical column. It owns a deterministic sorted
vocabulary and assigns dense nonnegative `Code` values starting at zero.
Numeric differences between codes have no meaning. Fitting returns a new object;
encoding and query preparation do not add labels or change existing codes.

### Scalar labels

```python
from mixdex_encoding import LabelEncoder

encoder = LabelEncoder.fit(['series', 'film', 'film'])
assert encoder.classes == ['film', 'series']
assert encoder.transform(['series', 'film']) == [1, 0]
assert encoder.decode(1) == 'series'
```

`encode` and `transform` require known labels and reject unknown values.
`encode_query` returns a known code or the sentinel `len(classes)` for an unseen
label. This sentinel is only for comparison with the fitted stored vocabulary.
Different unknown labels share it: it must not be used to compare two unknown
raw labels or to build a new stored dataset. Use `encode` for stored values.

Python labels must be strings; bytes, numbers, booleans and None are rejected.
An empty string is a valid label. C++ uses `std::string` label identity; Python
Unicode labels are represented as UTF-8. Lexicographic ordering of valid UTF-8
labels preserves the previous catalog's Unicode ordering. Ordering is deterministic,
not locale-sensitive or semantic. Encoders are independent for different columns.

### Set-valued labels

`SetEncoder` maintains two levels:

- `classes`: distinct canonical whole sets, addressed by stored column codes;
- `member_classes`: distinct member labels, addressed inside each set.

`dictionary` is the list of sorted unique member-code vectors in whole-set-code
order. Pass it directly to `Dataset::set_category_sets`.

```python
from mixdex_encoding import SetEncoder

encoder = SetEncoder.fit([[], ['a'], ['b', 'a', 'a']])
assert encoder.classes == [[], ['a'], ['a', 'b']]
assert encoder.member_classes == ['a', 'b']
assert encoder.dictionary == [[], [0], [0, 1]]
assert encoder.encode(['b', 'a']) == 2
query = encoder.encode_query(['a', 'new'])
assert query.code == 3
assert query.members == [0, 2]
```

Stored sets are sorted and deduplicated. A known empty set differs from a missing
value. A novel query set carries both a whole-set code and its actual members:
known labels keep their codes; distinct unseen labels receive distinct codes
outside the member vocabulary. These unknown member codes are local to one query,
so they cannot be compared across independently encoded external queries.

This preserves partial overlap: `{a,new}` remains at Jaccard distance 1/2 from
`{a}`. The whole-set sentinel alone cannot represent that overlap. Query member
vectors are sorted, unique and nonnegative, as required by native search.

### Native use

```cpp
#include <mixdex/encoding.hpp>
#include <mixdex/types.hpp>

auto encoder = mixdex::SetEncoder::fit({{}, {"a"}, {"a", "b"}});
mixdex::Dataset data({}, {0, 1, 2}, 3, 0, 1,
                     {mixdex::FeatureType::Categorical}, {}, {"tags"});
data.set_category_sets(0, encoder.dictionary());
auto query = encoder.encode_query({"a", "new"});
// query.members is the actual set for RowView::cat_sets.
```

Unknown stored labels throw `std::invalid_argument`. Invalid decode codes throw
`std::out_of_range`. Class counts and query-only codes are checked against the
signed 32-bit index range. Empty fitted vocabularies are permitted, but have no
known values; decoding or stored encoding consequently rejects every code/label.
Python properties return detached lists. C++ accessors borrow encoder-owned data.
Read-only calls can share an encoder; assignment/destruction must not race readers.
The native Python wrapper converts inputs while holding the GIL and releases it
for fitting, batches and set queries; those regions access no Python objects.

### Integration and cost

`mixdex_encoding.category_value` validates raw Python categories and delegates
set canonicalization to C++. `mixdex_preprocessing` keeps an import alias for
compatibility, but owns only numeric preparation and imputation. CatalogSnapshot
selects active/nonconstant columns, fits encoders, groups representations, scales
numbers and constructs the index. Recommendation ranking remains outside Mixdex.

Scalar fitting deduplicates before sorting: expected O(n + u log u) dictionary
work for n labels and u distinct classes, assuming bounded label cost. Lookup
uses unordered maps with expected constant bucket work; hashing/comparing strings
also depends on their lengths. Worst-case hash behavior is not guaranteed constant.
Persistent dictionary storage follows unique classes, not repeated fitting rows.
Set work also depends on member counts and byte lengths; sorted unique inputs
have a linear canonicalization path. Set-map keys use byte-length prefixes to
avoid ambiguities from concatenation, empty labels or embedded zero bytes.
No universal latency improvement is asserted for the Python/native boundary.

Catalog snapshot formats v1/v2/v3 retain their existing category arrays, member
ordering and source/transform layout. Loading still rebuilds and checks the
transform. Standalone applications can serialize `classes` and reconstruct with
`fit`; they must preserve which column and vocabulary an encoded value belongs to.

The dense per-column integer convention is also described by
[scikit-learn OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).
Mixdex implements its own encoders; set dictionaries and query-local unknown
members follow the contracts above. GIL handling follows the
[pybind11 guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil).

<a id="imputation"></a>

## Fitted imputation

Imputation is a separate Python preprocessing step, not a DatasetPolicy.
C++ Dataset keeps Strict/Clean. The reusable `mixdex_preprocessing.Imputer`
accepts raw row matrices; `CatalogSnapshot.build(..., impute=True)` fits and
applies it before encoding, row merging, feature removal and scaling.
The default `impute=False` preserves existing exclusion behavior.

### Fitting rules

Each source object contributes once to each observed column, including objects
missing a different column. Fit before deduplication: repeated representations
still describe separate source objects. Feature weights control relevance, not
observation multiplicity. Zero-weight columns are ignored and later removed.

- Numeric: the 0.5 quantile of observed values. Continuous uses MedianUnbiased
  (Hyndman–Fan type 8); Discrete uses InvertedCDF (type 1). These are the same
  methods used by native scaling. Discrete fills are observed values; this type
  does not require integer-valued data.
- Scalar categorical: the most frequent observed string. Ties choose the
  lexicographically smallest label, independently of input order.
- Set categorical: let k be the InvertedCDF 0.5 quantile of known set sizes.
  Count how many objects contain each element; repeats within one object count
  once. Select k elements by descending frequency, then ascending label. Store
  the resulting set in canonical label order. Each set column fits its own k.

Missing sets do not affect sizes or counts. Known empty sets do: their size is
zero, and they remain empty during transform. With known sizes 0 and 2, k=0
(the lower median); a missing set is consequently filled with the empty set.
This is a selected heuristic, not a proven Jaccard-optimal representative or a
claim of improved recommendation quality.

None denotes missing. For consistency with the existing catalog contract,
numeric NaN and infinity are also normalized to missing. Fully missing active
columns raise: there is no observed median, mode or size distribution. An empty
catalog therefore cannot fit an imputer. No synthetic fallback is inserted.

### Reuse and persistence

`Imputer.fit` creates a fitted object. `transform` returns detached numeric and
categorical row matrices; it does not encode labels, remove columns or refit.
`transform_row` uses the same fills for one external query. `parameters()` returns
detached JSON-compatible fill values and observation counts.

```python
from mixdex_preprocessing import Imputer

imputer = Imputer.fit(
    numeric=[[1, 1], [9, 9], [None, None]],
    numeric_types=['Continuous', 'Discrete'],
    categorical=[[['a']], [['a', 'b', 'c']], [None]],
    categorical_set_columns=[0],
)
numeric, categorical = imputer.transform_row([None, None], [None])
# numeric == [5.0, 1.0]; categorical == [('a',)]
```

Catalog snapshots store original missing values, fitted fills, types and counts
in readable JSON v3. Loading rebuilds from the original source and checks the
reproduced transform against the saved parameters. v1/v2 snapshots remain
readable; they retain their formats when no new preprocessing options are used.
Logical remove/restore never refits. Explicit rebuild may refit; compaction fits
on the remaining source objects, so it can change fills and distances. Compacting
away every observed value of an active column raises like a fresh build.

Fills and scales are fixed within one snapshot. Distances and bounds apply to
the resulting complete representations, which can coincide for distinct source
objects. No coordinates are excluded separately for individual pairs. The
existing conservative handling of Jaccard lower bounds remains unchanged.

### Cost and evidence

Numeric medians reuse native `quantiles_select` with expected linear selection
work per column. Modes use frequency tables. Set fitting counts canonicalized
elements and selects top-k with a heap: O(U log k) selection for U distinct
elements (O(U) for k<=1), plus input canonicalization and output ordering. This
adds preprocessing work; unchanged average search performance is not asserted.
Queries only substitute fitted values; they do not scan the source catalog.

Numeric methods: [NumPy quantile documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html),
which cites Hyndman and Fan, *Sample Quantiles in Statistical Packages* (1996).
Median/mode imputation: [scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
Fit on training data only and reuse at evaluation/query time:
[scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html).
The frequency-set rule above is the author's chosen design, not attributed to
these sources.

<a id="quantiles-and-scaling"></a>

## Quantiles and scaling

`mixdex/quantile.hpp` implements Hyndman–Fan type 1 (`InvertedCDF`) for
`NumericType::Discrete` and type 8 (`MedianUnbiased`) for
`NumericType::Continuous`. These are sample quantile conventions, not a test
of whether the input happens to contain integers.

See [quantile definitions](distance-bounds.md#quantiles) for the formulas.

### C++ calls

- `quantiles(x_sorted, ps, method)` requires sorted, finite input. It checks
  the method, length, probabilities and accessed values, without scanning
  the whole input. Checking only accessed values does not validate the
  complete array. For q probabilities, time and result storage are O(q).
- `quantiles_select(x_work, ps, method)` accepts unsorted, finite input and
  reorders the working array. All values and probabilities are validated
  before reordering; invalid arguments leave the array unchanged.
- Both require input length to fit `int`. Nonempty probabilities require
  nonempty input; empty probabilities return an empty result. The selection
  path still validates finite input in this case. Invalid methods and
  numeric types throw `std::invalid_argument`, including Release builds.

Selection sorts and deduplicates the required order-statistic indices, then
uses `std::nth_element` on progressively shorter suffixes. With r distinct
indices (r <= min(2q, n)), expected work is O(n + q log q + r*n), and extra
storage is O(q). For a fixed set of probabilities, such as the seven used by
scaling, expected work is O(n). No worst-case linear bound is claimed.

### Python and preprocessing

`mixdex.quantiles` validates one-dimensional, sorted, finite input in O(n)
before calling C++; total time is O(n + q). `mixdex.quantiles_select` copies
the input and delegates validation to C++. Neither modifies the Python input.

Standalone scaling rejects nonfinite values through the same native
selection path. Imputation and scaling share these quantile conventions;
`method_for` rejects an invalid numeric type before fitting the scale.

Catalog snapshot loading retains its exact preprocessing compatibility
check. Old fixtures remain unchanged. A snapshot whose preprocessing relied
on a corrected numerical defect may be rejected and must be rebuilt from
its source data; no silent migration or relaxed comparison is applied.

### Scaling

`fit_scale_params` and `scale_feature` fit numeric medians and widths.
`Scaler::fit` fits all numeric columns; reuse its parameters for stored and
external rows. Width selection tries IQR, then the interdecile range, then
max - min, accepting the first finite width greater than
`kScaleRelEps * max(abs(min), abs(max))`. Empty, nonfinite, constant and
numerically degenerate inputs are rejected. `ScaleLevel` records which width
was selected. Scaling does not refit on each query.

If no finite usable width exists because the range overflows, fitting throws
`std::overflow_error`. Transforming first tries `(value - median) / scale`;
if only the subtraction overflows, it tries `value / scale - median / scale`.
An unrepresentable transformed value also throws `std::overflow_error`.
The current Python runtime-error translator exposes these failures as
`mixdex.RuntimeError`, a subclass of `RuntimeError`.
`scale` and `Scaler::transform` preserve an empty Dataset's schema and metadata;
the latter still checks that the numeric feature count matches the fitted scaler.

Distance kernels recover intermediate overflow with a binary-scaled accumulator.
They retain the original calculation when its result is finite. If the final
normalized result is unrepresentable, the `noexcept` kernels return positive
infinity. This is overflow recovery, not a correctly rounded arithmetic or
underflow guarantee. See [range checks](distance-bounds.md#floating-point-range).

<a id="numeric-types-and-errors"></a>

## Numeric types and errors

Library-authored comments, error messages and warnings use English.

Mixdex uses `std::int32_t` for `Idx` and `Code`, `std::uint64_t` for
`Count`, and `std::size_t` for buffer sizes. Supported targets require a
32-bit `int` because several C++ and Python interfaces expose `int` directly.
Changing signed statistics to unsigned statistics requires recompiling C++ clients.

- A required row index must satisfy `0 <= i < n`.
- `kInvalidIdx == -1` means absence only in fields that explicitly allow it,
  such as missing children in a tree. It is not an accepted row index.
- Prepared category codes are nonnegative. Dataset construction reserves the
  largest code because category counts need `max_code + 1`.
- Negative category inputs retain their existing missing-value behavior:
  `Strict` rejects them; `Clean` removes affected rows. Imputation is a separate
  preprocessing step. Numerical non-finite values retain the Dataset policy.
- Noise labels, unlabeled training entries and automatic parameters have their
  own documented `-1` conventions. They are not row indices.

Both Dataset constructors accept zero rows with a valid feature schema in
`Strict` and `Clean`. Metadata is retained, coordinate buffers are empty, and
categorical code counts are zero. No cleanup warnings are emitted for an
initially empty dataset. Column views are empty and do not offset an empty
buffer. There are no valid row indices. A malformed empty schema is still
rejected; this storage contract does not make an empty dataset valid input to
every algorithm or to fitting a Scaler.

Only `DatasetPolicy::Strict` and `DatasetPolicy::Clean` are accepted. Other
underlying enum values throw `std::invalid_argument` before cleanup or warnings.

Low-level supporting containers have explicit preconditions:

- `FixedMaxHeap` retains at most its capacity, ordered by (distance, index).
  `max_distance` is a pruning cutoff: positive infinity until full, and also
  when empty. `farthest` returns the actual farthest retained distance, or zero
  when empty. Insertions require a nonnegative distance (positive infinity is
  allowed, NaN is not) and a nonnegative index; Debug builds assert this.
  Construction can throw allocation/length exceptions. If `reset` fails to
  allocate, retained entries and logical capacity are unchanged; an internal
  buffer may already have reserved more space. Successful reset clears entries
  while reusing storage; geometric buffer growth avoids allocating at every
  one-element capacity increase. Top-k selection takes O(k) initial heap construction
  and O(log k) per accepted replacement, with O(k) storage.
- `UF(n)` accepts zero and rejects negative n with `std::invalid_argument`
  before constructing its buffers. Element indices must be in [0,n); Debug
  builds assert this, while Release keeps the unchecked `noexcept` fast path.
  `find_compress`, `same`, `unite` and `flatten` modify parent links.
  Concurrent `find_root` calls require that no thread modifies the structure.
  Union by size and path splitting use O(n) storage; the amortized operation
  bound is described in the [method references](references.md#supporting-structures).

Dataset row and feature counts must fit the signed index range. Buffer products,
edge counts and cluster IDs have separate limits; a valid row count alone does
not establish that a derived structure fits. Checked narrowing throws
`std::length_error`, invalid arguments throw `std::invalid_argument`, and counter
overflow throws `std::overflow_error`. Python maps these to its corresponding
exceptions. Integers outside the range of a bound C++ argument are rejected by
the binding before the algorithm runs.

`Scaler::transform_rows` checks `m * n_num` with `checked_size_product` before
validating the buffer length or reading values. An unrepresentable product
throws `std::length_error`; a representable mismatched length throws
`std::invalid_argument`. Zero rows accept an empty numeric buffer.

The integer rules follow the C++ working draft:
[`std::in_range`](https://eel.is/c++draft/utility.intcmp) and
[unsigned arithmetic](https://eel.is/c++draft/basic.fundamental).

`parallel_for` accepts zero work and rejects negative counts. `chunk_begin`
requires a nonnegative total, a positive partition count, and a partition
boundary from zero through the partition count, inclusive.

Statistics never intentionally wrap. Task-local DT-Boruvka counters mark
exhaustion and report it after the task region; counter reductions are checked.
The optional atomic kNN visit counter checks before publishing an increment.
These checks preserve asymptotic complexity, but profiling has runtime overhead.

`f_equal` treats equal signed infinities as equal, all other pairs containing an
infinity as unequal, and every pair containing NaN as unequal. Finite comparisons
retain the relative/absolute tolerance rule. This is not a transitive equivalence
relation and must not replace exact equality for deduplication or sorting.

<a id="data-views"></a>

## C++ data views

Dataset stores separate row-major numeric and categorical buffers. Rows use
`std::span<const T>`; columns use `StridedSpan<T>`. A view borrows its source and
expires when that storage is destroyed or replaced. It does not freeze the data.

`StridedSpan` addresses element `i` as `pointer[i * stride]`, with stride measured
in elements. Construction and `operator[]` are unchecked: every addressed element
must belong to the source buffer, and offset products must be representable.
`data()` returns only the first element's address; it does not imply contiguity.
`is_contiguous()` is true for at most one element or a unit stride.

| Operation | Contract and cost |
| :--- | :--- |
| `at(i)` | Throws `std::out_of_range` unless `i < size()`; O(1) |
| `subspan(offset, count)` | Checks `offset <= size()` and `count <= size() - offset`; O(1) |
| `to_vector()` | Owns a detached copy; O(n) time and memory for numeric elements |
| `copy_to(T*, step)` | Unchecked destination capacity; O(n) assignments |
| `copy_to(std::span<T>, step)` | Checks destination extent and positive step for nonempty copies before writing; O(n) |

A valid empty subspan retains the original base pointer and stride. It never
computes a physical end address beyond the backing array. Its `data()` must not
be interpreted as the original slice boundary. Empty copies do no work and
accept empty destinations. Source and destination must not overlap.

The destination span covers the entire writable backing range, including gaps:

```cpp
#include <span>
#include <mixdex/utils/strided_span.hpp>

const int rows[]{1, 2, 3, 4, 5, 6};
mixdex::StridedSpan<int> column(rows + 1, 3, 2); // 2, 5.
int output[3]{};
column.copy_to(std::span<int>(output), 2); // 2, 0, 5.
```

`StridedIterator` is a read-only forward iterator. It supports independent
multipass traversal, postfix increment, `std::distance` and ranges algorithms.
It stores a logical position, so `end()` does not construct a strided end pointer.
It is not a random-access or contiguous iterator; `std::distance` takes O(n).
Equality includes the source pointer, stride and position.

Pointer copying remains noexcept for nonthrowing element assignments; assignments
of user-defined elements can propagate exceptions. Checked operations can throw.
For built-in numeric types, view creation, indexing and iterator increments are
O(1), with O(1) view storage and no allocation.

### Sources

- [C++ pointer arithmetic](https://eel.is/c++draft/expr.add).
- [Forward iterator requirements](https://eel.is/c++draft/iterator.concept.forward).
- [Copy algorithm preconditions](https://eel.is/c++draft/alg.copy).

<a id="build"></a>

## Build, installation and tests

### Python bindings and tests

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

On macOS, `build.sh` explicitly selects the SDK from the active Xcode or Command
Line Tools directory. It respects `DEVELOPER_DIR` and an explicit SDK path in
`SDKROOT`, and prints the SDK used. This avoids mixing a newer standalone SDK
with an older linker. The system-wide developer selection is not changed.
To choose an SDK manually, run `SDKROOT=/path/to/MacOSX.sdk ./build.sh`.
Direct CMake builds can use `-DCMAKE_OSX_SYSROOT=/path/to/MacOSX.sdk`.

OpenMP is optional. For macOS install libomp and enable `USE_OPENMP`; for Linux
use a compiler with the matching OpenMP runtime. MSVC task-based builds require
an OpenMP implementation supporting tasks; start with the sequential build.
For heavily oversubscribed OpenMP tests, use OMP_WAIT_POLICY=PASSIVE and
KMP_BLOCKTIME=0 to avoid expensive spinning. Native CPU flags are off by default. Python packaging uses scikit-build-core; wheels are specific to the Python ABI,
OS and architecture used to build them.

### C++ installation

```sh
cmake --install build --config Release --prefix /your/prefix --component CppDevelopment
```

Consumers use `find_package(mixdex CONFIG REQUIRED)` and link `mixdex::mixdex`.
The exported target carries C++20 and thread/OpenMP dependencies. A relocatable
install is tested with the separate project under `tests/install`.

