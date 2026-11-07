# Component-local MST updates

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
a lock-order cycle. Distinct components no longer share one global critical
section. Mutex storage is O(number of points), alongside the existing candidate
and atomic-weight arrays.

This argument depends on immutable UF during search and the task-join boundary.
It does not establish safety if a future change merges components inside a task.
Finite tests check edge/weight determinism across thread counts and full HDBSCAN
results; their agreement is not a substitute for those synchronization invariants.

## ThreadSanitizer setup

The Linux diagnostic job uses Clang 19 and a checksum-verified LLVM OpenMP 19.1.7
runtime built by `.github/scripts/build_tsan_openmp.sh`. Runtime atomics are
instrumented so TSan can observe synchronization when task buffers are recycled.
Generic loads/stores in the runtime are not instrumented; Mixdex and its tests
retain full memory instrumentation. Archer supplies the OpenMP task/barrier
annotations. This checks Mixdex's concurrency, not all runtime internals.

The runtime-only flag `-mllvm -tsan-instrument-memory-accesses=false` leaves atomic
instrumentation enabled, as defined in
[LLVM's TSan pass](https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/llvm/lib/Transforms/Instrumentation/ThreadSanitizer.cpp).
The task allocator uses pooled memory rather than one system allocation per task;
see [LLVM's tasking implementation](https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/openmp/runtime/src/kmp_tasking.cpp).
An uninstrumented runtime can hide synchronization from the detector, producing
false positives; see the
[Clang TSan guidance](https://clang.llvm.org/docs/ThreadSanitizer.html#non-instrumented-code).

`tests/sanitizers/openmp_runtime_probe.cpp` calibrates the configuration before
CTest: repeated valid tasks must succeed, and `--race` must produce a data-race
report and exit code 66. This intentional failure is accepted only for that
control run. Reports from library tests still fail the job. No project functions
are suppressed or excluded from memory instrumentation.

This runtime is built in the runner's temporary directory and is not shipped in
the library. Changes to the diagnostic compiler/runtime require both controls
and the complete C++ suite to pass again. Finite runs do not prove race freedom
for every input or schedule.
