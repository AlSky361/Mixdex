"""Compare set metrics and safe bounds with independent exhaustive references."""
import itertools
import numpy as np
import pytest
import mixdex
from helpers import make


def dataset():
    sets = [[], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    d = make(np.arange(7, dtype=float)[:, None], np.arange(7, dtype=np.int32)[:, None])
    d.set_category_sets(0, sets)
    return d, [set(x) for x in sets]


def test_distance_and_tree():
    d, sets = dataset()
    for weights in ([1., 1.], [2., 5.], [0., 1.]):
        d, sets = dataset()
        d.set_weights(weights)
        tree = mixdex.BallTree(d, 2)
        for i, j in itertools.product(range(7), repeat=2):
            union = sets[i] | sets[j]
            jac = len(sets[i] ^ sets[j]) / len(union) if union else 0.
            expected = (weights[0]*abs(i-j) + weights[1]*jac)/sum(weights)
            assert mixdex.distance(d, i, j) == pytest.approx(expected)
        for i in range(7):
            for ni, node in enumerate(tree.nodes):
                actual = min(mixdex.distance(d, i, j) for j in tree.indices[node.low:node.high])
                assert tree.lower_bound(i, ni) <= actual
            bt = mixdex.knn(d, i, 6, mixdex.SearchMethod.BallTree, 2)
            bf = mixdex.knn(d, i, 6, mixdex.SearchMethod.BruteForce, 2)
            assert bt == bf


def test_dictionary_validation_and_stale_index():
    d, _ = dataset()
    index = mixdex.Index(d)
    d.set_category_sets(0, [[i + 10] for i in range(7)])
    with pytest.raises(Exception):
        mixdex.knn(index, 0, 2)
    for bad in ([[0]] * 7, [[-1]] + [[i] for i in range(6)], [[]]):
        with pytest.raises(ValueError):
            d.set_category_sets(0, bad)


def test_scaling_preserves_sets():
    d, sets = dataset()
    scaled = mixdex.scale(d)
    scaled.set_weights([0., 1.])
    for i, j in itertools.product(range(7), repeat=2):
        union = sets[i] | sets[j]
        expected = len(sets[i] ^ sets[j]) / len(union) if union else 0.
        assert mixdex.distance(scaled, i, j) == pytest.approx(expected)


def test_dual_node_bounds():
    d, _ = dataset()
    tree = mixdex.BallTree(d, 2)
    core = [sorted(mixdex.distance(d, i, j) for j in range(7) if i != j)[1] for i in range(7)]
    tree.set_core_dist(core)
    for a, na in enumerate(tree.nodes):
        for b, nb in enumerate(tree.nodes):
            actual = min(max(core[i], core[j], mixdex.distance(d, i, j))
                         for i in tree.indices[na.low:na.high] for j in tree.indices[nb.low:nb.high])
            assert tree.lower_bound_mreach_node(a, b) <= actual


def test_mixed_multiple_set_columns_and_projection():
    rng = np.random.default_rng(321)
    sets = [set(i for i in range(3) if code & (1 << i)) for code in range(8)]
    numeric = rng.normal(size=(64, 2))
    cat = np.column_stack([rng.integers(0, 3, 64), rng.integers(0, 8, 64), rng.integers(0, 8, 64)]).astype(np.int32)
    d = make(numeric, cat)
    dictionary = [sorted(x) for x in sets]
    d.set_category_sets(1, dictionary)
    d.set_category_sets(2, dictionary)
    d.set_weights([2., 1., 3., 0., 4.])
    tree = mixdex.BallTree(d, 4)
    for i, j in itertools.product(range(64), repeat=2):
        a, b = sets[cat[i, 2]], sets[cat[j, 2]]
        jac = len(a ^ b)/len(a | b) if a | b else 0.
        expected = (2*abs(numeric[i, 0]-numeric[j, 0]) + abs(numeric[i, 1]-numeric[j, 1])
                    + 3*int(cat[i, 0] != cat[j, 0]) + 4*jac)/10
        assert mixdex.distance(d, i, j) == pytest.approx(expected)
    for i in range(8):
        assert mixdex.knn(d, i, 12, mixdex.SearchMethod.BallTree, 4) == mixdex.knn(d, i, 12, mixdex.SearchMethod.BruteForce, 4)
        for ni, node in enumerate(tree.nodes):
            assert tree.lower_bound(i, ni) <= min(mixdex.distance(d, i, j) for j in tree.indices[node.low:node.high])


@pytest.mark.parametrize("name,index", [("row_num", -1), ("row_num", 7), ("col_num", -1), ("col_num", 1), ("feat_type", 2), ("feat_name", -1), ("num_type", 1), ("weight", -1)])
def test_python_accessors_validate_indices(name, index):
    d, _ = dataset()
    with pytest.raises(ValueError):
        getattr(d, name)(index)


@pytest.mark.parametrize("method", [mixdex.SearchMethod.BallTree, mixdex.SearchMethod.BruteForce])
def test_external_actual_sets_and_filtered_ties(method):
    d, sets = dataset()
    index = mixdex.Index(d, method, 2)
    mask = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
    expected = []
    query = {0, 4}
    for i, values in enumerate(sets):
        if not mask[i] or i == 4:
            continue
        jac = len(query ^ values)/len(query | values)
        expected.append((i, (abs(2.5-i) + jac)/2))
    expected.sort(key=lambda x: (x[1], x[0]))
    threshold = expected[1][1]
    assert mixdex.knn_filtered(index, [2.5], [7], 2, mask, [4],
                              categorical_sets=[[0, 4]], include_ties=True) == [x for x in expected if x[1] <= threshold]
    for bad in ([[4, 0]], [[0, 0]], [[-1]], [[0], [1]]):
        with pytest.raises(ValueError):
            mixdex.knn_filtered(index, [2.5], [7], 2, mask, [], categorical_sets=bad)


def test_schema_order_and_set_dictionary_required():
    assert mixdex.NumFeature().type == mixdex.NumericType.Continuous
    with pytest.raises(ValueError):
        mixdex.Dataset([1., 2.], [0, 1], 2, 1, 1,
                       [mixdex.FeatureType.Categorical, mixdex.FeatureType.Numeric],
                       [mixdex.NumericType.Continuous], ['wrong', 'order'])
    with pytest.raises(ValueError):
        mixdex.Dataset([], [0, 1], 2, 0, 1, [mixdex.FeatureType.CategoricalSet], [], ['missing_dictionary'])


def test_balltree_owns_python_dataset_lifetime_and_validates_core():
    import gc
    import weakref
    d, _ = dataset()
    reference = weakref.ref(d)
    tree = mixdex.BallTree(d, 2)
    del d
    gc.collect()
    assert reference() is not None
    before = tree.knn(0, 3)
    for invalid in ([0.], [-1.]*7, [float('nan')]*7, [float('inf')]*7):
        with pytest.raises(ValueError):
            tree.set_core_dist(invalid)
        assert tree.knn(0, 3) == before
    del tree
    gc.collect()
    assert reference() is None
