# Distance, bounds and quantiles

Scope: exact arithmetic unless stated otherwise.

## Composite distance and safe bounds

Partition active indices into N (numeric), C (scalar categorical), S (finite sets).
All weights are positive and fixed; W is their sum. Numeric coordinates already
use fixed positive scales. Define

    D(x,y) = [sum_N w_j |x_j-y_j| + sum_C w_j 1[x_j!=y_j]
              + sum_S w_j d_J(x_j,y_j)] / W.

d_J(A,B)=|A symmetric_difference B|/|A union B|, with d_J(empty,empty)=0.
The similarity coefficient is J=1-d_J, so J(empty,empty)=1. Do not use
the same J symbol for both similarity and distance.
The distance d_J is a metric on finite sets; see
[Kosub, 2016](https://arxiv.org/abs/1612.02696v1) and the
[method references](references.md#distance-and-preprocessing).
Each component obeys the triangle inequality. Multiplication by positive weights,
addition and division by fixed W preserve it. Zero distance identifies equal
representations; separate record IDs need not be distinct points.

For any node v, the existing numerical bbox and numerical-ball contributions
remain valid with the total W. Scalar category contribution is w_j when the
query category is absent from the node, otherwise zero. For two nodes it is w_j
when their category sets are disjoint. Set-feature contribution is initially zero.
Since d_J >= 0, dropping it from a lower estimate is admissible: for every stored
point (or node pair) the estimated numerator is at most the true numerator.
Take the maximum of bounds on the same numeric contribution, then add only
nonoverlapping categorical contributions; do not count a component twice.

Do NOT apply the scalar absent-category bound to set codes: different sets can
have Jaccard distance less than one. In particular {a} and {a,b} have distance 1/2.
The implementation skips set columns in scalar-category bounds.

The legacy full mutual-reachability ball rounding guard uses L1/Hamming-specific
operation allowances (6 for point/node, 8 for node/node). These are project
allowances in a gamma-style rounding model, not constants supplied by Higham.
A complete operation-by-operation certification for every floating-point input
is not established by this exact-arithmetic derivation.
Jaccard introduces extra divisions; until a new floating-point error bound is
established, that full-ball bound is disabled for datasets with set dictionaries.
The numerical bbox/ball, scalar category bounds and minimum core distances remain.
This restriction avoids claiming unproved numerical safety. No extra set radius
or tuning parameter is introduced. Set comparison costs depend on set lengths;
constant per-feature cost and unchanged average search time are not asserted.

Potential later improvement: a Jaccard ball gives max(0,d_J(q,c)-r), directly by
triangle inequality. It requires its own radii, rounding guard and measurements;
it is not enabled here.

Dictionary sets must be distinct after sorting/deduplication. Otherwise different
codes could represent the same set and bypass Dataset duplicate handling.
Unknown codes are isolated points at distance one from every known set and other
unknown codes (except themselves). Triangle inequality holds since known d_J is
at most one. Unknown codes are not an encoding of a new partially overlapping set.

Audit obligations: disjoint index partition, fixed denominator, empty-set case,
canonical dictionaries, projection/remapping, scaling, stale index rejection,
point/node and node/node finite checks. Tests do not replace the derivation.

Notation: set inclusion allowing equality is written as A subseteq B. Probabilistic
notes use M[Z] and D[Z] for expectation and variance and F_Z(t)=P{Z<t}.
The symbol D(x,y) above denotes a distance, not the variance operator.

<a id="floating-point-range"></a>

## Floating-point range

The ordinary distance expression is retained when it produces a finite result.
Otherwise a private accumulator stores a nonnegative binary fraction and an
integer exponent. Products and additions align those exponents; normalization
materializes the final result only after multiplying by the stored inverse W.
If subtraction of finite coordinates overflows, its operands have opposite
signs, so their absolute values can be accumulated separately. The fallback
takes O(d) work and O(1) storage, in addition to set-comparison work. It can
still return positive infinity when the final result exceeds the double range.
It does not promise correct rounding or recover ordinary-path underflow.

The range guard in `src/distance_range.hpp` prevents mixing this recovered
arithmetic with a geometric bound derived from the original operation order.
The following argument assumes binary64, round-to-nearest, gradual underflow,
finite complete coordinates, validated weights, and no fast-math reassociation.
It certifies absence of overflow in those distance expressions; it does not
complete the legacy rounding-allowance proof described above.

1. Let r be the stored rounded inverse of the finite sequential weight sum W.
   Positive summands imply each weight is at most W. With b=nextafter(r,0),
   b <= 1/W. If frexp(b) has exponent e, each weight is at most the power
   P_w=2^(1-e). Without explicit weights use P_w=1.
2. Let M be the maximum of one and the absolute rounded differences between
   the query and both bbox endpoints, or the opposing endpoints of two boxes.
   If frexp(M) has exponent a, a conservative bound for both cross distances
   and internal node differences is 2^(a+1). The latter also covers radii,
   because centers are stored points. Reject a nonfinite endpoint difference
   or an unrepresentable coordinate-difference envelope.
3. Every weighted component is bounded by P=2^(a+1)*P_w. Numeric coordinates
   use the envelope above; categorical distances are at most one. A positive
   prefix of t components is at most t*P by monotone rounding, since that
   dyadic upper bound is representable. The dimension count has at most
   31 significant bits, so d*P is exact whenever finite. Checking that d*P
   and (d*P)*r are finite excludes product, sum and normalization overflow.
4. A cheap sufficient case handles M <= 2^256 and
   2^-256 <= r <= 2^256. Then coordinate differences are bounded by 2^258,
   weights by 2^257, d < 2^31, and normalized prefixes by 2^802, safely
   below binary64 overflow. These powers are range budgets, not model parameters.
5. If the guard fails, point/node geometry returns zero; node/node
   mutual reachability retains only the minimum-core-distance bound.
   Pure categorical data need no guard: every component is at most one,
   and their sequential weighted sum is bounded by the validated W.

The numeric early cutoff uses the same dyadic-prefix argument for its remaining
categorical additions; otherwise it completes the distance before deciding.
These checks add constant work per bound without changing its O(d) cost.
Conservative fallback can reduce pruning and increase search work on extreme
data; unchanged average search time is not asserted.

Primitive contracts: Microsoft CRT documentation for
[`frexp`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/frexp?view=msvc-170)
and [`ldexp`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/ldexp?view=msvc-170).

<a id="quantiles"></a>

## Quantile definitions

`mixdex/quantile.hpp` implements Hyndman–Fan type 1 (`InvertedCDF`) for
`NumericType::Discrete` and type 8 (`MedianUnbiased`) for
`NumericType::Continuous`. These are sample quantile conventions, not a test
of whether the input happens to contain integers.

For sorted values x[1], ..., x[n] and a probability p in [0, 1], let
j = floor(n*p + m) and g = n*p + m - j. Indices outside [1, n] select the
nearest endpoint. The result is (1 - gamma)*x[j] + gamma*x[j + 1], where:

| Method | m | gamma |
| --- | --- | --- |
| Type 1 | 0 | 0 when g == 0, otherwise 1 |
| Type 8 | (p + 1)/3 | g |

p = 0 and p = 1 return the minimum and maximum exactly. Type 1 selects an
observed value; type 8 interpolates. No application comparison tolerance is
used to widen a rank boundary. As with other double implementations, n*p
and interpolation are subject to floating-point rounding.

Repeated endpoints and zero/one interpolation weights return the endpoint
directly, preserving constant subnormal values. Other interpolation keeps the
weighted-sum operation order used by existing snapshots. If this produces a
nonfinite value or leaves the endpoint interval, `std::lerp` provides the
finite, bounded fallback. This does not promise correctly rounded quantiles
for every possible floating-point input.

- Hyndman, R. J.; Fan, Y. (1996). *Sample Quantiles in Statistical Packages*.
  The American Statistician 50(4), 361–365.
  [Author record](https://robjhyndman.com/publications/quantiles/),
  [definitions and method numbering in R](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html).
- C++ working draft: [`std::lerp` guarantees](https://eel.is/c++draft/c.math.lerp),
  [`std::nth_element` and its complexity](https://eel.is/c++draft/alg.nth.element).
- [NumPy quantile documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
  supplies an independent reference in the Python tests.

<a id="parallel-boruvka"></a>

## Parallel Boruvka invariants

Each Boruvka round reads a fixed union-find
partition. Merge/compression and counter reduction happen only after all tasks
have joined. No task updates union-find during candidate search.

Each component has an independent best edge, ordered by
(weight, min(endpoint), max(endpoint)) in deterministic mode. Its mutex serializes
all reads/writes of that edge during search. Relaxed atomic weights are pruning
hints; the definitive edge comparison occurs under the mutex.

Within a round the best weight never increases. An older larger weight can only
retain unnecessary candidates. Deterministic mode admits equal weights so the
endpoint tie-break is not skipped. Round reset precedes the next parallel region;
all workers must see the new round state through the region synchronization.

The two endpoint components are updated separately. No operation assumes an
atomic pair of updates: an intermediate state simply retains more work for the
component not yet updated. Locks are never nested, so these updates cannot form
a lock-order cycle. Distinct components do not share one global critical
section. Mutex storage is O(number of points), alongside the existing candidate
and atomic-weight arrays.

This argument depends on immutable UF during search and the task-join boundary.
It does not establish safety if a future change merges components inside a task.
Finite tests check edge/weight determinism across thread counts and full HDBSCAN
results; their agreement is not a substitute for those synchronization invariants.

API validation and preprocessing contracts: [api.md](api.md#quantiles-and-scaling).
