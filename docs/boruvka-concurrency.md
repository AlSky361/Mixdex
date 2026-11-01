# Component-local MST updates

Self-reviewed argument, 2026-11-01. Each Boruvka round reads a fixed union-find
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
a lock-order cycle. Distinct components no longer share one global critical
section. Mutex storage is O(number of points), alongside the existing candidate
and atomic-weight arrays.

This argument depends on immutable UF during search and the task-join boundary.
It does not establish safety if a future change merges components inside a task.
Finite tests check edge/weight determinism across thread counts and full HDBSCAN
results; their agreement is not a substitute for those synchronization invariants.

Recorded macOS stress-test durations are workload-specific: the old global-lock
MST test took 268 seconds in one run; the component-lock version took under a
second in the later run. This is not a general speedup claim or a controlled
paired benchmark. No ThreadSanitizer pass is claimed unless recorded separately.
