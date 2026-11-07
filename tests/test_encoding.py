"""Native encoding, label identity, query-only unknowns and catalog integration."""
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path

import pytest
import mixdex
from mixdex_encoding import LabelEncoder, SetEncoder, category_value
from mixdex_preprocessing import category_value as legacy_category_value
from mixdex_catalog import CatalogSnapshot


def test_scalar_codes_are_dense_stable_and_detached():
    labels = ['😀', 'é', 'a', '', 'α', 'a', '\0']
    expected = sorted(set(labels))
    encoder = LabelEncoder.fit(labels)
    assert isinstance(encoder, mixdex.LabelEncoder)
    assert encoder.classes == expected
    assert encoder.transform(labels) == [expected.index(x) for x in labels]
    assert [encoder.decode(i) for i in range(len(expected))] == expected
    reverse = LabelEncoder.fit(list(reversed(labels)))
    assert reverse.classes == expected
    changed = encoder.classes
    changed.clear()
    assert encoder.classes == expected
    for value in ('new', 'other'):
        assert encoder.encode_query(value) == len(expected)
        with pytest.raises(ValueError): encoder.encode(value)
    for code in (-1, len(expected)):
        with pytest.raises(IndexError): encoder.decode(code)


@pytest.mark.parametrize('value', [None, 1, True])
def test_scalar_raw_types_are_not_silently_converted(value):
    with pytest.raises((TypeError, ValueError)): LabelEncoder.fit(['a', value])
    if value is None:
        assert category_value(value, False) is None
    else:
        with pytest.raises(ValueError): category_value(value, False)


def test_set_classes_distinguish_empty_labels_delimiters_and_empty_sets():
    values = [[], [''], ['ab', 'c'], ['a', 'bc'], ['1:a'], ['1', 'a'], ['a\0b'], ['b', 'a', 'a']]
    expected = sorted({tuple(sorted(set(x))) for x in values})
    encoder = SetEncoder.fit(values + [values[-1]])
    assert encoder.classes == [list(x) for x in expected]
    assert encoder.transform(values) == [expected.index(tuple(sorted(set(x)))) for x in values]
    assert encoder.encode(['a', 'b', 'b']) == encoder.encode(['b', 'a'])
    assert [encoder.decode(i) for i in range(len(expected))] == [list(x) for x in expected]
    dictionary = encoder.dictionary
    dictionary[0].append(999)
    assert encoder.dictionary != dictionary
    assert category_value([], True) == ()
    assert category_value(None, True) is None
    assert category_value(frozenset(['b', 'a']), True) == ('a', 'b')
    assert legacy_category_value is category_value


def test_novel_set_query_preserves_known_overlap_and_distinct_unknowns():
    encoder = SetEncoder.fit([[], ['a'], ['a', 'b']])
    query = encoder.encode_query(['z', 'a', 'new', 'z'])
    assert query.code == 3 and query.members == [0, 2, 3]
    assert encoder.encode_query(['b', 'a']).code == 2
    assert encoder.encode_query([]).members == []
    with pytest.raises(ValueError): encoder.encode(['a', 'z'])
    with pytest.raises(IndexError): encoder.decode(-1)
    snapshot = CatalogSnapshot.build([10, 20, 30], categorical=[[['a']], [['a', 'b']], [[]]],
                                    categorical_set_columns=[0])
    assert snapshot.search(categorical=[['a', 'z']])['items'] == [(10, .5), (20, 2/3), (30, 1.)]


def test_canonicalization_uses_the_native_implementation():
    assert mixdex.canonicalize_category_set(['b', 'a', 'a']) == ['a', 'b']
    for value in ('ab', [1], {'a': 1}):
        with pytest.raises(ValueError): category_value(value, True)
    assert LabelEncoder.fit([]).encode_query('x') == 0
    query = SetEncoder.fit([]).encode_query([])
    assert query.code == 0 and query.members == []


def test_set_encoding_preserves_jaccard_for_all_subsets():
    labels = ['a', 'é', '😀']
    values = [list(x) for n in range(4) for x in itertools.combinations(labels, n)]
    encoder = SetEncoder.fit(values)
    codes = encoder.transform(values)
    data = mixdex.Dataset([], codes, len(values), 0, 1,
                          [mixdex.FeatureType.Categorical], [], ['set'])
    data.set_category_sets(0, encoder.dictionary)
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            a, b = set(a), set(b)
            expected = len(a ^ b) / len(a | b) if a | b else 0.
            assert mixdex.distance(data, i, j) == expected


def test_queries_do_not_refit_and_are_safe_to_share():
    encoder = SetEncoder.fit([['a'], ['a', 'b'], []])
    before = encoder.classes, encoder.member_classes, encoder.dictionary
    queries = [['a', str(i)] for i in range(40)]
    def encode(value):
        q = encoder.encode_query(value)
        return q.code, q.members
    expected = list(map(encode, queries))
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(encode, queries)) == expected
    assert (encoder.classes, encoder.member_classes, encoder.dictionary) == before


@pytest.mark.parametrize('value', [b'a', 1, None, True])
def test_native_entry_points_reject_non_string_labels(value):
    scalar = LabelEncoder.fit(['a'])
    sets = SetEncoder.fit([['a']])
    for action in (lambda: mixdex.LabelEncoder.fit([value]),
                   lambda: scalar.encode(value),
                   lambda: scalar.encode_query(value),
                   lambda: scalar.transform([value]),
                   lambda: mixdex.SetEncoder.fit([[value]]),
                   lambda: sets.encode([value]),
                   lambda: sets.encode_query([value])):
        with pytest.raises((TypeError, ValueError)): action()


@pytest.mark.parametrize('value', ['ab', b'ab'])
def test_a_scalar_is_not_a_collection_of_labels(value):
    for action in (lambda: LabelEncoder.fit(value), lambda: SetEncoder.fit([value]),
                   lambda: SetEncoder.fit(value), lambda: mixdex.canonicalize_category_set(value)):
        with pytest.raises((TypeError, ValueError)): action()


@pytest.mark.parametrize('value', [None, 3, True])
def test_set_rows_must_be_collections(value):
    with pytest.raises((TypeError, ValueError)): SetEncoder.fit([value])


@pytest.mark.parametrize('version', ['v1', 'v2', 'v3'])
def test_snapshots_written_before_encoding_extraction_are_unchanged(version, tmp_path):
    fixture = Path(__file__).parent / 'fixtures' / f'catalog_{version}.json'
    payload = json.loads(fixture.read_text())['payload']
    assert payload['format'] == f'mixdex.catalog.{version}'
    restored = CatalogSnapshot.load(fixture)
    rebuilt = CatalogSnapshot.build(**payload['source'])
    assert restored.report() == rebuilt.report()
    target = tmp_path / 'snapshot.json'
    restored.save(target)
    assert target.read_bytes() == fixture.read_bytes()
