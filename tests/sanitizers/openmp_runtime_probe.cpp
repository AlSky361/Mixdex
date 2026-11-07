// Check OpenMP task-buffer reuse and that TSan still detects an intentional race.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <thread>
#include <vector>

namespace {
volatile int race_value = 0;
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

void race_write(int worker, volatile int& value, std::atomic<int>& arrived) {
    // Coordinate entry without introducing a happens-before edge between writers.
    arrived.fetch_add(1, std::memory_order_relaxed);
    while (arrived.load(std::memory_order_relaxed) != 2) {
    }
    // Exercise many conflicting accesses per worker.
    for (int i = 0; i < 10000; ++i) value = worker;
}

int check_race_detection() {
    volatile int& value = race_value;
    std::atomic<int> arrived{0};
    omp_set_dynamic(0);
#pragma omp parallel num_threads(2) shared(value, arrived)
    {
        const int worker = omp_get_thread_num();
#pragma omp task if (0) firstprivate(worker) shared(value, arrived)
        race_write(worker, value, arrived);
    }
    return value < 0;
}

int check_thread_race() {
    volatile int& value = race_value;
    std::atomic<int> arrived{0};
    std::thread first([&] {
        race_write(0, value, arrived);
    });
    std::thread second([&] {
        race_write(1, value, arrived);
    });
    first.join();
    second.join();
    return value < 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 1)
        return check_task_reuse();
    if (argc == 2 && std::strcmp(argv[1], "--race") == 0)
        return check_race_detection();
    if (argc == 2 && std::strcmp(argv[1], "--thread-race") == 0)
        return check_thread_race();
    return 2;
}
