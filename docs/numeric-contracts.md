# Numeric types and valid values

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

Dataset row and feature counts must fit the signed index range. Buffer products,
edge counts and cluster IDs have separate limits; a valid row count alone does
not establish that a derived structure fits. Checked narrowing throws
`std::length_error`, invalid arguments throw `std::invalid_argument`, and counter
overflow throws `std::overflow_error`. Python maps these to its corresponding
exceptions. Integers outside the range of a bound C++ argument are rejected by
the binding before the algorithm runs.

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
