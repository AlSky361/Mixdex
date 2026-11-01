# Fitted imputation

Imputation is a separate Python preprocessing step, not a DatasetPolicy.
C++ Dataset keeps Strict/Clean. The reusable `mixdex_preprocessing.Imputer`
accepts raw row matrices; `CatalogSnapshot.build(..., impute=True)` fits and
applies it before encoding, row merging, feature removal and scaling.
The default `impute=False` preserves existing exclusion behavior.

## Fitting rules

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

## Reuse and persistence

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

## Cost and evidence

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
