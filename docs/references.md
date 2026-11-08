# Scientific references and implementation scope

These references identify established methods used by, or relevant to, Mixdex.
They do not claim that Mixdex invented the named algorithms, that its source was
copied from these publications, or that every implementation detail is identical.
Third-party code notices are separate: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Density clustering

- Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996).
  *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
  with Noise*. KDD, 226–231.
  [Publisher paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf).
  Reference for `src/algorithms/dbscan/dbscan.cpp`. Mixdex also provides a
  core-only DBSCAN* variant; it is not the ordinary border-assignment rule.
- Ricardo J. G. B. Campello, Davoud Moulavi, Jörg Sander (2013).
  *Density-Based Clustering Based on Hierarchical Density Estimates*.
  PAKDD, 160–172. [DOI](https://doi.org/10.1007/978-3-642-37456-2_14).
- Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, Jörg Sander (2015).
  *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier
  Detection*. ACM TKDD 10(1), Article 5. [DOI](https://doi.org/10.1145/2733381).
  References for HDBSCAN*, cluster stability, cluster extraction and GLOSH in
  `src/algorithms/hdbscan/` and `src/density/`. Mixdex has explicit handling of
  equal edges, fixed-k core neighborhoods, floating-point tolerances and
  degenerate data. Approximate mode builds a restricted graph and is not the
  complete-graph exact algorithm.
- Leland McInnes, John Healy (2017).
  *Accelerated Hierarchical Density Clustering*.
  [Author preprint, v2](https://arxiv.org/abs/1705.07321v2);
  [published version](https://doi.org/10.1109/ICDMW.2017.12).
  Algorithmic reference for accelerated HDBSCAN* and its dual-tree MST context.
- Leland McInnes, John Healy, Steve Astels (2017).
  *hdbscan: Hierarchical density based clustering*. JOSS 2(11), 205.
  [Software paper](https://doi.org/10.21105/joss.00205),
  [software](https://github.com/scikit-learn-contrib/hdbscan).
  This credits the related software package separately from the original
  HDBSCAN* method. The software's BSD license and the paper's publication
  license are distinct.

## Spanning trees and dual-tree search

- Otakar Borůvka (1926). *O jistém problému minimálním*.
  [Digitized original](https://dml.cz/handle/10338.dmlcz/500114).
  Historical reference for the component-based MST algorithm in `src/mst/boruvka.cpp`.
- William B. March, Parikshit Ram, Alexander G. Gray (2010).
  *Fast Euclidean Minimum Spanning Tree: Algorithm, Analysis, and Applications*.
  KDD, 603–612. [Author manuscript](https://mlpack.org/papers/emst.pdf).
  Reference for DualTreeBoruvka in `src/mst/dt_boruvka.cpp` and
  `src/mst/dt_boruvka_traversal.cpp`.
  The paper's Euclidean complexity analysis is not automatically an analysis of
  Mixdex's mixed-distance or mutual-reachability implementation. Component-local
  locking and deterministic tie handling are documented separately in
  [parallel Boruvka invariants](distance-bounds.md#parallel-boruvka).
- Ryan Curtin, William March, Parikshit Ram, David Anderson, Alexander Gray,
  Charles Isbell (2013). *Tree-Independent Dual-Tree Algorithms*.
  PMLR 28(3), 1435–1443.
  [Publisher record](https://proceedings.mlr.press/v28/curtin13.html).
  Background for separating tree, traversal, base case and pruning rule.
- Stephen M. Omohundro (1989). *Five Balltree Construction Algorithms*.
  [Author manuscript](https://steveomohundro.com/wp-content/uploads/2009/03/omohundro89_five_balltree_construction_algorithms.pdf).
  Background for ball-tree indexing in `src/index/ball_tree.cpp`; this does not
  attribute Mixdex's mixed-feature bounds or all construction details to the paper.

## Outliers and classification

- Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander (2000).
  *LOF: Identifying Density-Based Local Outliers*.
  [Publisher record](https://sigmodrecord.org/2000/06/07/lof-identifying-density-based-local-outliers/).
  Reference for `src/algorithms/outlier/lof.cpp`. Mixdex takes exactly k
  neighbors with deterministic ties and adds `kLofLrdRegularizer` to the average
  reachability distance; this is not a claim of literal identity to every
  neighborhood convention in the paper.
- Sridhar Ramaswamy, Rajeev Rastogi, Kyuseok Shim (2000).
  *Efficient Algorithms for Mining Outliers from Large Data Sets*.
  [Publisher record](https://sigmodrecord.org/2000/06/08/efficient-algorithms-for-mining-outliers-from-large-data-sets/).
  Reference for scoring by distance to the kth neighbor in
  `src/algorithms/outlier/knn_outlier.cpp`, not a claim to implement the paper's
  complete partition-based mining algorithm.
- Thomas M. Cover, Peter E. Hart (1967). *Nearest Neighbor Pattern Classification*.
  [Author manuscript](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf).
  Background for `src/algorithms/knn/knn_classifier.cpp`. Weighted voting,
  zero-distance handling and label tie-breaking follow the documented Mixdex API;
  no statistical risk guarantee is asserted for the mixed-distance classifier.

## Distance and preprocessing

- Richard W. Hamming (1950). *Error Detecting and Error Correcting Codes*.
  Bell System Technical Journal 29(2), 147–160.
  [Original paper](https://zoo.cs.yale.edu/classes/cs323/doc/Hamming.pdf).
  Historical reference for counting coordinate mismatches; Mixdex uses a
  weighted indicator for each scalar categorical coordinate, not an error-correcting codec.
- J. C. Gower (1971). *A General Coefficient of Similarity and Some of Its Properties*.
  Biometrics 27(4), 857–871. [DOI](https://doi.org/10.2307/2528823).
  Historical mixed-data reference for the earlier API name. The current fixed-scale
  L1/Hamming/Jaccard composite is specified independently; it does not implement
  the pair-dependent missing-value comparison rule of the original coefficient.
- Sven Kosub (2016). *A note on the triangle inequality for the Jaccard distance*.
  [Author preprint, v1](https://arxiv.org/abs/1612.02696v1).
  Reference for the finite-set Jaccard component discussed in
  [distance-bounds.md](distance-bounds.md). Mixdex treats two known empty sets
  as distance zero. Numeric L1 and categorical Hamming are separate components.
  The fixed-weight composite and its implementation bounds are explained in
  that document; unknown codes and floating-point guards are project conventions.
- Rob J. Hyndman, Yanan Fan (1996). *Sample Quantiles in Statistical Packages*.
  The American Statistician 50(4), 361–365.
  [DOI](https://doi.org/10.1080/00031305.1996.10473566),
  [definitions in R](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html).
  `src/quantile.cpp`, `src/scale.cpp` and `python/mixdex_preprocessing.py` use
  type 1 for discrete and type 8 for continuous numeric features.
  Contracts, numerical handling and selection complexity: [quantiles and scaling](api.md#quantiles-and-scaling).
- Nicholas J. Higham (2002). *Accuracy and Stability of Numerical Algorithms*,
  second edition. [Author record](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/).
  Reference for unit roundoff and gamma-style error bounds in
  `include/mixdex/utils/constants.hpp`. Operation allowances, comparison
  tolerances, scaling fallback rules and the LOF regularizer are Mixdex choices,
  not constants attributed to Higham.

Median/mode imputation is also described by
[scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
The frequent-element set fill and fitted snapshot policy are project choices;
see [fitted imputation](api.md#imputation). Basic heaps, union-find and nearest-neighbor
search are standard supporting structures; no invention claim is made for them.
The former `gower_*` name is historical; the current composite is defined in
README.md and is not a claim to reproduce all missing-data conventions of Gower's
original coefficient.

## Row hashing

- David Stafford (2011). *Better Bit Mixing - Improving on MurmurHash3's 64-bit
  Finalizer*. [Author's parameter table](https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html).
  The Mix13 row specifies shifts 30, 27, 31 and multipliers
  `0xbf58476d1ce4e5b9`, `0x94d049bb133111eb`.
- Sebastiano Vigna (2015). *SplitMix64 reference implementation*.
  [Author's source](https://prng.di.unimi.it/splitmix.c).
  `mix64` in `src/data/dataset_rows.cpp` uses the same fixed increment and
  output transformation as one step of that implementation. It is stateless
  here: Mixdex combines coordinate bits into row hashes rather than maintaining
  a pseudorandom generator. The source notice is preserved in
  [LICENSES/splitmix64.txt](../LICENSES/splitmix64.txt).
- Guy L. Steele Jr., Doug Lea, Christine H. Flood (2014).
  *Fast Splittable Pseudorandom Number Generators*. OOPSLA, 453–472.
  [Author copy](https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf),
  [DOI](https://doi.org/10.1145/2660193.2660195).
  Background for SplitMix, the golden-ratio increment and Stafford Mix13.

The increment `0x9e3779b97f4a7c15` is the odd integer closest to
2^64 / phi, where phi = (1 + sqrt(5)) / 2. The two multipliers are odd and,
together with the shifts, are the empirically evaluated Mix13 parameters.
They are retained as a named algorithm's fixed constants, not retuned here.

The unsigned arithmetic wraps modulo 2^64 intentionally. These constants are
algorithm parameters, not dataset-dependent tolerances. Row hashes only group
candidates: exact coordinate comparisons decide duplicates. A bijective mixer
on single words does not make the variable-length row hash collision-free.

## Software and reproducibility

[NumPy](https://github.com/numpy/numpy),
[pybind11](https://github.com/pybind/pybind11), and
[scikit-learn](https://github.com/scikit-learn/scikit-learn) are acknowledged as
runtime/build/test dependencies, respectively. See requirements-test.txt and
pyproject.toml for roles and version constraints. Tests against another package
are finite comparisons, not proofs of identical behavior for all inputs.
