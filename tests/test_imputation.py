"""Fitted fills, query consistency and legacy snapshot compatibility."""
from concurrent.futures import ThreadPoolExecutor
import json
import random

import numpy as np
import pytest
import mixdex
from mixdex_preprocessing import Imputer
from mixdex_catalog import CatalogSnapshot


def test_medians_use_numeric_type_and_do_not_refit():
    rows = [[1, 1], [9, 9], [None, float('nan')]]
    imputer = Imputer.fit(rows, numeric_types=['Continuous', mixdex.NumericType.Discrete])
    assert imputer.parameters()['numeric_fill'] == [5., 1.]
    assert imputer.transform([[None, None], [1000, 2000]]) == ([[5., 1.], [1000., 2000.]], [[], []])
    assert rows[2][0] is None
    assert imputer.parameters()['numeric_observed'] == [2, 2]
    assert imputer.transform_row([None, None]) == ([5., 1.], [])


def test_randomized_quantiles_match_numpy_and_preserve_input():
    rng = random.Random(72)
    for n in (1, 2, 3, 10, 21, 200):
        values = [rng.randrange(-100, 100) for _ in range(n)]
        before = list(values)
        for name, method in [('inverted_cdf', mixdex.QuantileMethod.InvertedCDF),
                             ('median_unbiased', mixdex.QuantileMethod.MedianUnbiased)]:
            ps = [0, .1, .25, .5, .75, .9, 1]
            assert mixdex.quantiles_select(values, ps, method) == pytest.approx(np.quantile(values, ps, method=name))
            assert values == before
    with pytest.raises(ValueError):
        mixdex.quantiles_select([float('nan')], [.5], method)
    with pytest.raises(ValueError):
        mixdex.quantiles_select([1], [float('nan')], method)
    with pytest.raises(ValueError):
        mixdex.quantiles_select([], [.5], method)


def test_mode_uses_original_objects_and_stable_label_ties():
    imputer = Imputer.fit(categorical=[['b'], ['a'], ['b'], [None]])
    assert imputer.transform(categorical=[[None], ['new']]) == ([[], []], [['b'], ['new']])
    tied = Imputer.fit(categorical=[['b'], ['a']])
    assert tied.parameters()['categorical_fill'] == ['a']
    report = imputer.parameters()
    report['categorical_fill'][0] = 'changed'
    assert imputer.transform_row(categorical=[None])[1] == ['b']


def test_sets_use_discrete_size_median_document_frequency_and_empty_sets():
    # Known sizes: 1, 3, 3, 3 -> k=3; repeated labels in one object count once.
    rows = [[['b', 'b']], [['b', 'c', 'd']], [['a', 'c', 'd']], [['a', 'c', 'e']], [None]]
    imputer = Imputer.fit(categorical=rows, categorical_set_columns=[0])
    assert imputer.parameters()['categorical_fill'] == [['a', 'b', 'c']]
    assert imputer.parameters()['categorical_observed'] == [4]
    assert imputer.transform_row(categorical=[None])[1] == [('a', 'b', 'c')]
    assert imputer.transform_row(categorical=[[]])[1] == [()]
    assert imputer.transform_row(categorical=[['z', 'z']])[1] == [('z',)]
    lower = Imputer.fit(categorical=[[[]], [['a', 'b']], [None]], categorical_set_columns=[0])
    assert lower.parameters()['categorical_fill'] == [[]]
    all_empty = Imputer.fit(categorical=[[[]], [[]], [None]], categorical_set_columns=[0])
    assert all_empty.parameters()['categorical_fill'] == [[]]


@pytest.mark.parametrize('num,cat,columns', [([[None]], None, []), (None, [[None]], []), (None, [[None]], [0])])
def test_fully_missing_active_columns_raise(num, cat, columns):
    with pytest.raises(ValueError, match='no observed'):
        Imputer.fit(num, cat, categorical_set_columns=columns)


def test_ignored_columns_and_invalid_schema():
    imputer = Imputer.fit([[1], [2]], [[None], [None]], weights=[1, 0])
    assert imputer.transform_row([None], [None]) == ([1.5], [None])
    for kwargs in ({'numeric_types': []}, {'numeric_types': ['unknown']},
                   {'weights': [True]}, {'weights': [0]}, {'categorical_set_columns': [0]}):
        with pytest.raises(ValueError):
            Imputer.fit([[1]], **kwargs)
    with pytest.raises(ValueError):
        imputer.transform([[None]], [[]])


def make_snapshot():
    return CatalogSnapshot.build(
        [1, 2, 3, 4], numeric=[[1, 1], [9, 9], [5, 1], [None, None]],
        numeric_types=['Continuous', 'Discrete'],
        categorical=[['b', ['a']], ['a', ['a', 'b', 'c']], ['b', ['b', 'c']], [None, None]],
        categorical_set_columns=[1], impute=True, leaf_size=1,
    )


def test_snapshot_queries_frozen_fills_groups_filters_and_parallelism():
    snapshot = make_snapshot()
    report = snapshot.report()
    assert not report['excluded']
    assert report['transform']['imputation']['numeric_fill'] == [5., 1.]
    assert report['transform']['imputation']['categorical_fill'] == ['b', ['a', 'b']]
    assert snapshot.search_by_id(4) == snapshot.search([5, 1], ['b', ['a', 'b']], exclude_ids=[4])
    expected = snapshot.search([None, None], [None, None])
    assert expected == snapshot.search([5, 1], ['b', ['a', 'b']])
    snapshot.remove([1])
    assert snapshot.report()['transform'] == report['transform']
    allowed = snapshot.prepare_filter([2, 4])
    def query(_):
        return snapshot.search([None, None], [None, None], allowed_ids=allowed)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(query, range(20)))
    assert all(r == results[0] for r in results)
    assert results[0]['items'][0] == (4, 0.)


def test_snapshot_roundtrip_rebuild_and_filled_duplicates(tmp_path):
    snapshot = make_snapshot()
    path = tmp_path / 'imputed.json'
    snapshot.remove([2])
    snapshot.save(path)
    payload = json.loads(path.read_text())['payload']
    assert payload['format'] == 'mixdex.catalog.v3'
    assert payload['source']['numeric'][-1] == [None, None]
    loaded = CatalogSnapshot.load(path)
    assert loaded.report() == snapshot.report()
    assert loaded.stats() == snapshot.stats()
    assert loaded.search([None, None], [None, None]) == snapshot.search([None, None], [None, None])
    assert snapshot.rebuild().stats() == snapshot.stats()
    assert snapshot.rebuild(compact=True).stats()['films'] == 3
    merged = CatalogSnapshot.build([1, 2, 3], numeric=[[1], [None], [1]], impute=True)
    assert merged.report()['groups'] == [[1, 2, 3]]
    assert merged.stats()['informative_features'] == 0
    assert merged.search([None])['items'] == [(1, 0.), (2, 0.), (3, 0.)]


def test_numeric_types_without_imputation_and_zero_weight_missing(tmp_path):
    s = CatalogSnapshot.build([1, 2], numeric=[[1], [9]], numeric_types=['Discrete'])
    assert s._data.num_type(0) == mixdex.NumericType.Discrete
    p = tmp_path / 'types.json'
    s.save(p)
    assert CatalogSnapshot.load(p).report() == s.report()
    s = CatalogSnapshot.build([1, 2], numeric=[[1], [2]], categorical=[[None], [None]],
                              weights=[1, 0], impute=True)
    assert s.stats()['indexed_films'] == 2
    assert s.report()['transform']['imputation']['categorical_fill'] == [None]


def test_snapshot_fits_before_duplicate_merging_and_rejects_changed_parameters(tmp_path):
    s = CatalogSnapshot.build(range(6), categorical=[['b'], ['b'], ['b'], ['a'], ['a'], [None]], impute=True)
    assert s.report()['groups'] == [[0, 1, 2, 5], [3, 4]]
    assert s.report()['transform']['imputation']['categorical_fill'] == ['b']
    path = tmp_path / 'params.json'
    s.save(path)
    envelope = json.loads(path.read_text())
    envelope['payload']['transform']['imputation']['categorical_fill'] = ['a']
    from mixdex_catalog import _json
    import hashlib
    envelope['sha256'] = hashlib.sha256(_json(envelope['payload']).encode()).hexdigest()
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='preprocessing'):
        CatalogSnapshot.load(path)


def test_snapshot_distance_matches_independent_completed_data_oracle():
    s = make_snapshot()
    num = [[1., 1.], [9., 9.], [5., 1.], [5., 1.]]
    cat = [['b', {'a'}], ['a', {'a', 'b', 'c'}], ['b', {'b', 'c'}], ['b', {'a', 'b'}]]
    transform = s.report()['transform']
    queries = [([None, None], [None, None]), ([7., 9.], ['new', ['a', 'z']])]
    for raw_n, raw_c in queries:
        qn = [5. if raw_n[0] is None else raw_n[0], 1. if raw_n[1] is None else raw_n[1]]
        qc = ['b' if raw_c[0] is None else raw_c[0], {'a', 'b'} if raw_c[1] is None else set(raw_c[1])]
        expected = []
        for id_, a, b in zip(range(1, 5), num, cat):
            scaled_a = [(v-m)/w for v, m, w in zip(a, transform['medians'], transform['scales'])]
            scaled_q = [(v-m)/w for v, m, w in zip(qn, transform['medians'], transform['scales'])]
            numeric = sum(abs(x-y) for x, y in zip(scaled_a, scaled_q))
            union = qc[1] | b[1]
            jac = len(qc[1] ^ b[1])/len(union) if union else 0.
            expected.append((id_, (numeric + (qc[0] != b[0]) + jac)/4))
        expected.sort(key=lambda item: (item[1], item[0]))
        assert s.search(raw_n, raw_c)['items'] == expected
