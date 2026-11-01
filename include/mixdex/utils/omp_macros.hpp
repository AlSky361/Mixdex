#pragma once

// Wrap compiler-specific OpenMP directives; run the same blocks sequentially without OpenMP.
// MSVC uses __pragma; GCC/Clang use _Pragma.

#ifdef _OPENMP
#include <omp.h>

#define MIXDEX_STRINGIFY2(...) #__VA_ARGS__
#define MIXDEX_STRINGIFY(...) MIXDEX_STRINGIFY2(__VA_ARGS__)

#if defined(_MSC_VER) && !defined(__clang__)
#define MIXDEX_PRAGMA(...) __pragma(__VA_ARGS__)
#else
#define MIXDEX_PRAGMA(...) _Pragma(MIXDEX_STRINGIFY(__VA_ARGS__))
#endif

#define MIXDEX_OMP_STATIC MIXDEX_PRAGMA(omp parallel for schedule(static))
#define MIXDEX_OMP_DYNAMIC MIXDEX_PRAGMA(omp parallel for schedule(dynamic))
#define MIXDEX_OMP_GUIDED MIXDEX_PRAGMA(omp parallel for schedule(guided))

#define MIXDEX_OMP_CRITICAL MIXDEX_PRAGMA(omp critical)

// One worker creates tasks; the team executes them and joins before leaving the block.
#define MIXDEX_OMP_PARALLEL_TASKS_N(nthreads, block)                                               \
    MIXDEX_PRAGMA(omp parallel num_threads(nthreads))                                              \
    MIXDEX_PRAGMA(omp single nowait) block

#define MIXDEX_OMP_TASK_SHARED(shared_vars) MIXDEX_PRAGMA(omp task shared(shared_vars))
#define MIXDEX_OMP_TASK_IF_SHARED(cond, shared_vars)                                               \
    MIXDEX_PRAGMA(omp task if (cond) shared(shared_vars))
#define MIXDEX_OMP_TASKWAIT MIXDEX_PRAGMA(omp taskwait)

#define MIXDEX_OMP_MAX_THREADS() omp_get_max_threads()
#define MIXDEX_OMP_THREAD_NUM() omp_get_thread_num()

#else

#define MIXDEX_OMP_STATIC
#define MIXDEX_OMP_DYNAMIC
#define MIXDEX_OMP_GUIDED
#define MIXDEX_OMP_CRITICAL

#define MIXDEX_OMP_PARALLEL_TASKS_N(nthreads, block) block
#define MIXDEX_OMP_TASK_SHARED(shared_vars)
#define MIXDEX_OMP_TASK_IF_SHARED(cond, shared_vars)
#define MIXDEX_OMP_TASKWAIT

#define MIXDEX_OMP_MAX_THREADS() 1
#define MIXDEX_OMP_THREAD_NUM() 0

#endif
