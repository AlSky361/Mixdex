# Categorical encoding

`mixdex/encoding.hpp` provides standalone C++ `LabelEncoder` and `SetEncoder`.
Python exposes the same native objects through `mixdex` and `mixdex_encoding`.
No Python runtime is required by the C++ implementation.

Each encoder fits one categorical column. It owns a deterministic sorted
vocabulary and assigns dense nonnegative `Code` values starting at zero.
Numeric differences between codes have no meaning. Fitting returns a new object;
encoding and query preparation do not add labels or change existing codes.

## Scalar labels

```python
from mixdex_encoding import LabelEncoder

encoder = LabelEncoder.fit(['series', 'film', 'film'])
assert encoder.classes == ['film', 'series']
assert encoder.transform(['series', 'film']) == [1, 0]
assert encoder.decode(1) == 'series'
```

`encode` and `transform` require known labels and reject unknown values.
`encode_query` returns a known code or the sentinel `len(classes)` for an unseen
label. This sentinel is only for comparison with the fitted stored vocabulary.
Different unknown labels share it: it must not be used to compare two unknown
raw labels or to build a new stored dataset. Use `encode` for stored values.

Python labels must be strings; bytes, numbers, booleans and None are rejected.
An empty string is a valid label. C++ uses `std::string` label identity; Python
Unicode labels are represented as UTF-8. Lexicographic ordering of valid UTF-8
labels preserves the previous catalog's Unicode ordering. Ordering is deterministic,
not locale-sensitive or semantic. Encoders are independent for different columns.

## Set-valued labels

`SetEncoder` maintains two levels:

- `classes`: distinct canonical whole sets, addressed by stored column codes;
- `member_classes`: distinct member labels, addressed inside each set.

`dictionary` is the list of sorted unique member-code vectors in whole-set-code
order. Pass it directly to `Dataset::set_category_sets`.

```python
from mixdex_encoding import SetEncoder

encoder = SetEncoder.fit([[], ['a'], ['b', 'a', 'a']])
assert encoder.classes == [[], ['a'], ['a', 'b']]
assert encoder.member_classes == ['a', 'b']
assert encoder.dictionary == [[], [0], [0, 1]]
assert encoder.encode(['b', 'a']) == 2
query = encoder.encode_query(['a', 'new'])
assert query.code == 3
assert query.members == [0, 2]
```

Stored sets are sorted and deduplicated. A known empty set differs from a missing
value. A novel query set carries both a whole-set code and its actual members:
known labels keep their codes; distinct unseen labels receive distinct codes
outside the member vocabulary. These unknown member codes are local to one query,
so they cannot be compared across independently encoded external queries.

This preserves partial overlap: `{a,new}` remains at Jaccard distance 1/2 from
`{a}`. The whole-set sentinel alone cannot represent that overlap. Query member
vectors are sorted, unique and nonnegative, as required by native search.

## Native use

```cpp
#include <mixdex/encoding.hpp>
#include <mixdex/types.hpp>

auto encoder = mixdex::SetEncoder::fit({{}, {"a"}, {"a", "b"}});
mixdex::Dataset data({}, {0, 1, 2}, 3, 0, 1,
                     {mixdex::FeatureType::Categorical}, {}, {"tags"});
data.set_category_sets(0, encoder.dictionary());
auto query = encoder.encode_query({"a", "new"});
// query.members is the actual set for RowView::cat_sets.
```

Unknown stored labels throw `std::invalid_argument`. Invalid decode codes throw
`std::out_of_range`. Class counts and query-only codes are checked against the
signed 32-bit index range. Empty fitted vocabularies are permitted, but have no
known values; decoding or stored encoding consequently rejects every code/label.
Python properties return detached lists. C++ accessors borrow encoder-owned data.
Read-only calls can share an encoder; assignment/destruction must not race readers.
The native Python wrapper converts inputs while holding the GIL and releases it
for fitting, batches and set queries; those regions access no Python objects.

## Integration and cost

`mixdex_encoding.category_value` validates raw Python categories and delegates
set canonicalization to C++. `mixdex_preprocessing` keeps an import alias for
compatibility, but owns only numeric preparation and imputation. CatalogSnapshot
selects active/nonconstant columns, fits encoders, groups representations, scales
numbers and constructs the index. Recommendation ranking remains outside Mixdex.

Scalar fitting deduplicates before sorting: expected O(n + u log u) dictionary
work for n labels and u distinct classes, assuming bounded label cost. Lookup
uses unordered maps with expected constant bucket work; hashing/comparing strings
also depends on their lengths. Worst-case hash behavior is not guaranteed constant.
Persistent dictionary storage follows unique classes, not repeated fitting rows.
Set work also depends on member counts and byte lengths; sorted unique inputs
have a linear canonicalization path. Set-map keys use byte-length prefixes to
avoid ambiguities from concatenation, empty labels or embedded zero bytes.
No universal latency improvement is asserted for the Python/native boundary.

Catalog snapshot formats v1/v2/v3 retain their existing category arrays, member
ordering and source/transform layout. Loading still rebuilds and checks the
transform. Standalone applications can serialize `classes` and reconstruct with
`fit`; they must preserve which column and vocabulary an encoded value belongs to.

The dense per-column integer convention is also described by
[scikit-learn OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).
Mixdex implements its own encoders; set dictionaries and query-local unknown
members follow the contracts above. GIL handling follows the
[pybind11 guidance](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil).
