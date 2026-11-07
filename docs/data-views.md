# Read-only data views

Dataset stores separate row-major numeric and categorical buffers. Rows use
`std::span<const T>`; columns use `StridedSpan<T>`. A view borrows its source and
expires when that storage is destroyed or replaced. It does not freeze the data.

`StridedSpan` addresses element `i` as `pointer[i * stride]`, with stride measured
in elements. Construction and `operator[]` are unchecked: every addressed element
must belong to the source buffer, and offset products must be representable.
`data()` returns only the first element's address; it does not imply contiguity.
`is_contiguous()` is true for at most one element or a unit stride.

| Operation | Contract and cost |
| :--- | :--- |
| `at(i)` | Throws `std::out_of_range` unless `i < size()`; O(1) |
| `subspan(offset, count)` | Checks `offset <= size()` and `count <= size() - offset`; O(1) |
| `to_vector()` | Owns a detached copy; O(n) time and memory for numeric elements |
| `copy_to(T*, step)` | Unchecked destination capacity; O(n) assignments |
| `copy_to(std::span<T>, step)` | Checks destination extent and positive step for nonempty copies before writing; O(n) |

A valid empty subspan retains the original base pointer and stride. It never
computes a physical end address beyond the backing array. Its `data()` must not
be interpreted as the original slice boundary. Empty copies do no work and
accept empty destinations. Source and destination must not overlap.

The destination span covers the entire writable backing range, including gaps:

```cpp
const int rows[]{1, 2, 3, 4, 5, 6};
mixdex::StridedSpan<int> column(rows + 1, 3, 2); // 2, 5.
int output[3]{};
column.copy_to(std::span<int>(output), 2); // 2, 0, 5.
```

`StridedIterator` is a read-only forward iterator. It supports independent
multipass traversal, postfix increment, `std::distance` and ranges algorithms.
It stores a logical position, so `end()` does not construct a strided end pointer.
It is not a random-access or contiguous iterator; `std::distance` takes O(n).
Equality includes the source pointer, stride and position.

Pointer copying remains noexcept for nonthrowing element assignments; assignments
of user-defined elements can propagate exceptions. Checked operations can throw.
For built-in numeric types, view creation, indexing and iterator increments are
O(1), with O(1) view storage and no allocation.

## Sources

- [C++ pointer arithmetic](https://eel.is/c++draft/expr.add).
- [Forward iterator requirements](https://eel.is/c++draft/iterator.concept.forward).
- [Copy algorithm preconditions](https://eel.is/c++draft/alg.copy).

The layout experiment is outside the library, under
`Эксперименты/Производительность/Доступ к данным` in the research project. Its
scaling changes are not part of the current implementation.
