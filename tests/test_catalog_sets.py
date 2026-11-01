"""Set-valued snapshots, including combinations absent from the training dictionary."""
import concurrent.futures
import json
import random

import pytest
from mixdex_catalog import CatalogSnapshot


def make():
    return CatalogSnapshot.build(
        [50, 10, 30, 20, 40, 60],
        categorical=[[['drama', 'thriller'], 'film'], [['thriller', 'drama'], 'film'],
                     [['drama'], 'film'], [[], 'film'], [['comedy'], 'series'], [None, 'film']],
        categorical_names=['genres', 'kind'], categorical_set_columns=[0], leaf_size=1,
    )


def test_canonical_groups_unknown_members_and_combinations():
    s = make()
    assert s.report()['groups'][0] == [10, 50]
    assert s.report()['excluded'] == {60: 'missing_features'}
    result = s.search(categorical=[{'drama', 'new'}, 'film'], k=10)['items']
    # Two retained features, equal weights; novel labels must preserve overlap.
    assert result == [(30, .25), (10, 1/3), (50, 1/3), (20, .5), (40, 1.)]
    assert s.search(categorical=[[], 'film'], k=1)['items'] == [(20, 0.)]
    assert s.search_by_id(50, k=1)['items'] == [(10, 0.)]
    with pytest.raises(ValueError):
        s.search(categorical=[None, 'film'])


def test_persistence_rebuild_compaction_and_source_ownership(tmp_path):
    data = [[['b', 'a', 'a']], [['a']], [[]]]
    s = CatalogSnapshot.build([1, 2, 3], categorical=data, categorical_set_columns=[0])
    data[0][0].clear()
    s.remove([2])
    path = tmp_path / 'sets.json'
    s.save(path)
    assert json.loads(path.read_text())['payload']['format'] == 'mixdex.catalog.v2'
    t = CatalogSnapshot.load(path)
    assert s.report() == t.report()
    assert s.stats() == t.stats()
    for q in ([], ['a'], ['unseen'], ['a', 'b', 'unseen']):
        assert s.search(categorical=[q]) == t.search(categorical=[q])
    assert s.rebuild().stats() == s.stats()
    compact = s.rebuild(compact=True)
    assert compact.stats()['films'] == 2
    with pytest.raises(KeyError):
        compact.restore([2])


def test_randomized_oracle_filters_ties_and_parallel_queries():
    rng = random.Random(101)
    ids = rng.sample(range(10000), 150)
    rows = [[sorted(rng.sample(list('abcde'), rng.randrange(6))), rng.choice('XY')] for _ in ids]
    snapshot = CatalogSnapshot.build(ids, categorical=rows, categorical_set_columns=[0], leaf_size=3,
                                     weights=[3, 1])
    inactive = set(ids[::9])
    snapshot.remove(inactive)
    allowed = set(ids[::2])
    prepared = snapshot.prepare_filter(allowed)
    queries = [[sorted(rng.sample(list('abcdef'), rng.randrange(7))), rng.choice('XYZ')] for _ in range(40)]

    def check(q):
        expected = []
        for id_, row in zip(ids, rows):
            if id_ in inactive or id_ not in allowed or id_ in ids[::13]:
                continue
            a, b = set(q[0]), set(row[0])
            jac = len(a ^ b) / len(a | b) if a | b else 0.
            expected.append((id_, (3*jac + (q[1] != row[1]))/4))
        expected.sort(key=lambda x: (x[1], x[0]))
        result = snapshot.search(categorical=q, k=12, allowed_ids=prepared, exclude_ids=ids[::13])
        assert result['items'] == expected[:12]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(check, queries))


def test_set_schema_validation_and_legacy_format(tmp_path):
    for columns in ([True], [-1], [1], [0, 0]):
        with pytest.raises(ValueError):
            CatalogSnapshot.build([1], categorical=[[[]]], categorical_set_columns=columns)
    for value in ('not a set', [1], {'a': 1}):
        with pytest.raises(ValueError):
            CatalogSnapshot.build([1], categorical=[[value]], categorical_set_columns=[0])
    s = CatalogSnapshot.build([1, 2], categorical=[['a'], ['b']])
    path = tmp_path / 'legacy.json'
    s.save(path)
    assert json.loads(path.read_text())['payload']['format'] == 'mixdex.catalog.v1'
    assert CatalogSnapshot.load(path).stats() == s.stats()


def test_multiple_set_columns_projection_and_empty_snapshot(tmp_path):
    s = CatalogSnapshot.build([1, 2, 3], categorical=[[['a'], ['x']], [['b'], []], [None, ['y']]],
                              categorical_set_columns=[0, 1], weights=[0, 2])
    assert s.search(categorical=[None, ['x', 'z']])['items'] == [(1, .5), (2, 1.), (3, 1.)]
    empty = CatalogSnapshot.build([], categorical_names=['genres'], categorical_set_columns=[0])
    path = tmp_path / 'empty.json'
    empty.save(path)
    assert CatalogSnapshot.load(path).stats() == empty.stats()
