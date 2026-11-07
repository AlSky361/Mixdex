# Implementation layout

The public C++ API remains in `include/mixdex`. Implementation-only helpers live
under `src`; they are not additional public APIs.

## Data views

Rows borrow contiguous storage; columns use a read-only strided view. Iterator,
checked access, copying and lifetime contracts are described in
[data-views.md](data-views.md). Alternative scaling traversals remain outside the
library until their performance and behavior are accepted.

## Encoding

`include/mixdex/encoding.hpp` and `src/data/encoding.cpp` own scalar and set label
vocabularies. `python/bindings/bind_encoding.cpp` exposes them; the Python
`mixdex_encoding` facade normalizes raw category types. Preprocessing imports
that normalization, and CatalogSnapshot uses the native encoders. See
[encoding.md](encoding.md) for unknown-label scope and compatibility.

## Dataset

| File | Responsibility |
| :--- | :--- |
| `src/types.cpp` | Construction, metadata access and completion of initialization |
| `src/data/dataset_validation.cpp` | Structural/value checks and preparation order |
| `src/data/dataset_rows.cpp` | Exact duplicate detection and row compaction |
| `src/data/dataset_columns.cpp` | Constant detection and ordered column projection |
| `src/data/dataset_weights.cpp` | Weight validation and transactional weight changes |
| `src/data/categorical.cpp` | Set dictionaries, category views and Jaccard comparisons |
| `src/index/node_categories.cpp` | Per-node category summaries |

`set_weights` validates the weights, stages changes in a copy, projects zero-weight
columns, removes newly identical rows, and commits the result. Its sum validator
only checks and totals weights. Projection does not decide which columns should
be dropped and does not remove duplicate rows.

`project_columns` applies one ordered selection to numeric/categorical row-major
buffers, feature names/types, weights, counts and set dictionaries. Constant
cleanup and zero-weight removal use this same operation. Dictionary keys are
remapped with their categorical columns. Internal selectors copy either metadata,
row-major coordinates or dictionaries; they do not mix selection policy with copying.

The projection cost follows the amount of data and dictionaries copied, plus
selection bookkeeping. Removing the former column-major intermediate buffers does
not worsen asymptotic complexity. Exact duplicate detection keeps its original
hash grouping followed by coordinate comparisons; hashes alone do not establish
equality. No new universal speedup or average-case bound is claimed.

## BallTree

`src/index/ball_tree.cpp` contains construction and splitting.
`src/index/ball_tree_bounds.cpp` contains ordinary/mutual-reachability lower bounds
and the summaries needed by the latter. The floating-point expressions, operation
order and rounding allowances are unchanged by this separation.

## Dual-tree Boruvka

- `src/mst/dt_boruvka.cpp`: one round loop shared by both public entry points,
  statistics conversion and optional diagnostic output.
- `src/mst/dt_boruvka_traversal.cpp`: pruning, leaf comparisons, child-pair ordering
  and task dispatch. One insertion-sort template handles three or four pairs.
- `src/mst/dt_boruvka_state.cpp`: state initialization, counter reduction and
  component merging after tasks finish.
- `src/mst/dt_boruvka_internal.hpp`: private state shared by those translation
  units; frequently used candidate-update operations remain inline.

`visit_children` handles scheduling and waiting. Recursive visitation handles one
node pair: reject it, process its leaves, or select its child pairs. Bound-based
ordering is used in the same cases as before. Each round still freezes union-find
while tasks search and merges only after they join. See
[concurrency invariants](boruvka-concurrency.md).

An orchestration function may call several stages: its single responsibility is
coordinating their order. Validation, copying, reporting and traversal mechanics
belong to their respective helpers. File length alone is not a reason to create
another abstraction. The Python catalog remains a separate public facade in this
C++ refactoring pass.
