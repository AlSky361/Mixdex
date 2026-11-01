# Composite distance and safe bounds

Self-reviewed derivation, 2026-11-01. Scope: exact arithmetic unless stated otherwise.

Partition active indices into N (numeric), C (scalar categorical), S (finite sets).
All weights are positive and fixed; W is their sum. Numeric coordinates already
use fixed positive scales. Define

    D(x,y) = [sum_N w_j |x_j-y_j| + sum_C w_j 1[x_j!=y_j]
              + sum_S w_j J(x_j,y_j)] / W.

J(A,B)=|A symmetric_difference B|/|A union B|, with J(empty,empty)=0.
Jaccard is a metric on finite sets; see
[Kosub, 2016](https://arxiv.org/abs/1612.02696v1) and the
[method references](references.md#distance-and-preprocessing).
Each component obeys the triangle inequality. Multiplication by positive weights,
addition and division by fixed W preserve it. Zero distance identifies equal
representations; separate record IDs need not be distinct points.

For any node v, the existing numerical bbox and numerical-ball contributions
remain valid with the NEW total W. Scalar category contribution is w_j when the
query category is absent from the node, otherwise zero. For two nodes it is w_j
when their category sets are disjoint. Set-feature contribution is initially zero.
Since J >= 0, dropping it from a lower estimate is admissible: for every stored
point (or node pair) the estimated numerator is at most the true numerator.
Take the maximum of bounds on the same numeric contribution, then add only
nonoverlapping categorical contributions; do not count a component twice.

Do NOT apply the scalar absent-category bound to set codes: different sets can
have Jaccard distance less than one. In particular {a} and {a,b} have distance 1/2.
The implementation skips set columns in scalar-category bounds.

The old full mutual-reachability ball rounding guard was derived for L1/Hamming.
Jaccard introduces extra divisions; until a new floating-point error bound is
established, that full-ball bound is disabled for datasets with set dictionaries.
The numerical bbox/ball, scalar category bounds and minimum core distances remain.
This restriction avoids claiming unproved numerical safety. No extra set radius
or tuning parameter is introduced. Set comparison costs depend on set lengths;
constant per-feature cost and unchanged average search time are not asserted.

Potential later improvement: a Jaccard ball gives max(0,J(q,c)-r), directly by
triangle inequality. It requires its own radii, rounding guard and measurements;
it is not enabled here.

Dictionary sets must be distinct after sorting/deduplication. Otherwise different
codes could represent the same set and bypass Dataset duplicate handling.
Unknown codes are isolated points at distance one from every known set and other
unknown codes (except themselves). Triangle inequality holds since known J is
at most one. Unknown codes are not an encoding of a new partially overlapping set.

Audit obligations: disjoint index partition, fixed denominator, empty-set case,
canonical dictionaries, projection/remapping, scaling, stale index rejection,
point/node and node/node finite checks. Tests do not replace the derivation.
