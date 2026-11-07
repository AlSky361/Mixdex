// Check OpenMP task-buffer reuse and that TSan still detects an intentional race.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>

namespace {
void visit(int node, int depth, int limit, std::vector<int>& output);

template <int N>
void children(int (&nodes)[N], int depth, int limit, std::vector<int>& output) {
    const int next = depth + 1;
    for (int i = 0; i < N; ++i) {
#pragma omp task if (depth < limit) shared(output)
        visit(nodes[i], next, limit, output);
    }
#pragma omp taskwait
}

void visit(int node, int depth, int limit, std::vector<int>& output) {
    if (depth == 7) {
        output[node] = node + 1;
    } else if (depth % 2) {
        int nodes[3] = {4 * node, 4 * node + 1, 4 * node + 2};
        children(nodes, depth, limit, output);
    } else {
        int nodes[4] = {4 * node, 4 * node + 1, 4 * node + 2, 4 * node + 3};
        children(nodes, depth, limit, output);
    }
}

int check_task_reuse() {
    std::vector<int> expected(1 << 14);
    visit(0, 0, 0, expected);
    for (int repeat = 0; repeat < 100; ++repeat) {
        std::vector<int> output(expected.size());
#pragma omp parallel num_threads(4)
#pragma omp single
        visit(0, 0, 3, output);
        if (output != expected)
            return 1;
    }
    std::puts("OpenMP task reuse: OK");
    return 0;
}

int check_race_detection() {
    volatile int value = 0;
    std::atomic<int> arrived{0};
    omp_set_dynamic(0);
#pragma omp parallel num_threads(2) shared(value, arrived)
    {
        // Each worker runs an undeferred task; neither write precedes the other.
#pragma omp task if (0) shared(value, arrived)
        {
            arrived.fetch_add(1, std::memory_order_release);
            while (arrived.load(std::memory_order_acquire) != 2) {
            }
            value = omp_get_thread_num();  // Deliberate race: TSan must reject this mode.
        }
    }
    return value < 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 1)
        return check_task_reuse();
    if (argc == 2 && std::strcmp(argv[1], "--race") == 0)
        return check_race_detection();
    return 2;
}
