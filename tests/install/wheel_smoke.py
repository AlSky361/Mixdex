"""Exercise installed native and catalog modules without the source build path."""
import tempfile
from pathlib import Path
from importlib.metadata import distribution, version

import mixdex
from mixdex_catalog import CatalogSnapshot
from mixdex_preprocessing import Imputer
from mixdex_encoding import LabelEncoder, SetEncoder

scalar = LabelEncoder.fit(['b', 'a'])
assert isinstance(scalar, mixdex.LabelEncoder) and scalar.encode('a') == 0
sets = SetEncoder.fit([['a'], ['a', 'b']])
assert sets.encode_query(['a', 'new']).members == [0, 2]

assert version('mixdex') == '0.1.0'
dist = distribution('mixdex')
assert dist.metadata['License-Expression'] == 'BSD-3-Clause'
license_paths = {str(p): p for p in dist.files if '.dist-info/licenses/' in str(p)}
for suffix, notice in (
    ('/LICENSE', 'Aleksandr'),
    ('/LICENSES/pybind11.txt', 'Wenzel Jakob'),
    ('/THIRD_PARTY_NOTICES.md', 'pybind11'),
):
    matches = [p for name, p in license_paths.items() if name.endswith(suffix)]
    assert len(matches) == 1, f'missing or duplicate license file: {suffix}'
    assert notice in dist.locate_file(matches[0]).read_text()
snapshot = CatalogSnapshot.build(
    [1, 2, 3], categorical=[[['a']], [['a', 'b']], [[]]], categorical_set_columns=[0],
)
assert snapshot.search(categorical=[['b', 'new']], k=3)['items'] == [(2, 2/3), (1, 1.), (3, 1.)]
snapshot.remove([2])
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'catalog.json'
    snapshot.save(path)
    loaded = CatalogSnapshot.load(path)
    assert loaded.stats() == snapshot.stats()
    assert loaded.search(categorical=[['a']]) == snapshot.search(categorical=[['a']])
print('Installed wheel passed:', mixdex.__file__)

# A finite positive sum can still have an infinite reciprocal.
d = mixdex.Dataset([0., 1.], [], 2, 1, 0, [mixdex.FeatureType.Numeric],
                   [mixdex.NumericType.Continuous], ['x'])
try:
    d.set_weights([5e-324])
except ValueError:
    pass
else:
    raise AssertionError('unrepresentable inverse weight sum was accepted')
assert not d.has_weights()

imputer = Imputer.fit([[1, 1], [9, 9]], numeric_types=['Continuous', 'Discrete'])
assert imputer.transform_row([None, None])[0] == [5., 1.]
imputed = CatalogSnapshot.build(
    [1, 2, 3], [[1, 1], [9, 9], [None, None]],
    categorical=[[['a']], [['a', 'b', 'c']], [None]], categorical_set_columns=[0],
    numeric_types=['Continuous', 'Discrete'], impute=True,
)
assert imputed.search([None, None], [None], k=1)['items'] == [(3, 0.)]
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'imputed.json'
    imputed.save(path)
    loaded = CatalogSnapshot.load(path)
    assert loaded.report() == imputed.report()
    assert loaded.search([None, None], [None]) == imputed.search([None, None], [None])
