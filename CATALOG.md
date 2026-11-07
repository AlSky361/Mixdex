# Catalog snapshots

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

## Data contract

- External IDs are unique signed 64-bit Python integers. Internal row indices
  remain signed 32-bit values; this is a representation limit, not tested capacity.
- Numeric columns default to continuous; `numeric_types` accepts Continuous/Discrete
  enum values or names, one per original numeric column. Scalar categories are strings. Declare set-valued
  categorical column indices with `categorical_set_columns=[...]`; each value is
  a list, tuple, set or frozenset of strings. Values are canonicalized, with empty
  sets distinct from `None` (missing). Columns use independent dictionaries.
- Missing significant values exclude a film from the index, not from the source
  catalog. `report()` records source-to-row mappings, duplicate groups, exclusions,
  dropped columns and fitted transforms. Zero-weight and constant columns are
  removed. Negative/nonfinite weights or a nonpositive total or nonfinite reciprocal total are rejected.
- Equal representations share a point while retaining separate film IDs. Signed
  zeros compare equal. Scaling can create additional exact duplicate points.
- With no informative features, usable films share distance zero. Check
  `stats()["informative_features"]` before interpreting results as meaningful.

## Queries

Opt-in `build(..., impute=True)` fits fills on the original catalog before cleanup:
type-specific numeric medians, scalar modes, and frequent-element sets with a
per-column discrete median size. Known empty sets count as size zero; missing
sets do not count. Fully missing positive-weight columns raise. Zero-weight
columns remain ignored. See [imputation.md](docs/imputation.md).
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

`search_by_id(id, k=10, **filters)` uses a source film and excludes its own ID.
Other IDs with the same representation remain eligible. Source films excluded
from the index return an empty result with a reason.

Results contain `items` as `(film_id, distance)` pairs, `version`, `snapshot_id`,
`activity_id` and `incomplete`. Ordering is by distance, then external ID, including
boundary ties. Filters apply during candidate selection and to each grouped ID.
Fewer than k eligible films produces a shorter list with `incomplete=True`.

`allowed_ids=None` allows all active IDs; an empty collection allows none.
Unknown query-filter IDs are ignored. `remove/restore` validate all IDs before
mutation; unknown IDs reject the call. Repeating a valid operation is harmless.

## Reusable filters

```python
allowed = snapshot.prepare_filter([1001, 1003])
result = snapshot.search_by_id(1001, k=10, allowed_ids=allowed)
```

A prepared filter belongs to one exact snapshot instance. Its fixed ID collection
maps to internal rows and caches one activity mask. Preparation plus the first
mask costs O(V+A) time/memory for V points and A allowed IDs. Repeated queries reuse
the mask until activity changes. Exclusions inspect affected groups. A new snapshot
requires a new prepared filter; the application controls how many filters it retains.

## Updates and persistence

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
catalog to introduce new or changed film records.

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
`mixdex_encoding`; [encoding.md](docs/encoding.md) describes their two-level set
representation and query-only unknown codes. Snapshot formats remain v1/v2/v3.
