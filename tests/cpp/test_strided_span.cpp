#include <cstdlib>
#include <vector>

#include "mixdex/utils/strided_span.hpp"
int main() {
    const int rows[]{1, 2, 3, 4, 5, 6};
    mixdex::StridedSpan<int> col(rows + 1, 3, 2);
    std::vector<int> found;
    for (int value : col) found.push_back(value);
    if (found != std::vector<int>{2, 5})
        return EXIT_FAILURE;
    mixdex::StridedSpan<int> empty;
    if (empty.begin() != empty.end())
        return EXIT_FAILURE;
    empty.copy_to(nullptr);
    return EXIT_SUCCESS;
}
