"""Check kNN classification and radius-query contracts."""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import mixdex
from helpers import METHODS, _distance_matrix, _mixed_ds, _numeric_ds


def _split(n, frac, seed):
    rng = np.random.default_rng(seed)
    test = rng.random(n) < frac
    return test


@pytest.fixture(scope="module")
def numeric_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.8, (120, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    y = np.repeat([0, 1, 2], 120)
    return _numeric_ds(X), y


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [1, 5, 15])
@pytest.mark.parametrize("weighted", [False, True])
def test_matches_sklearn_numeric(numeric_data, method, k, weighted):
    ds, y = numeric_data
    G = _distance_matrix(ds)
    test = _split(len(y), 0.3, 1)
    labels = np.where(test, mixdex.UNLABELED, y)
    pred = np.array(mixdex.knn_classify(ds, labels.tolist(), k, weighted, method))

    clf = KNeighborsClassifier(
        n_neighbors=k, metric="precomputed", weights="distance" if weighted else "uniform"
    ).fit(G[np.ix_(~test, ~test)], y[~test])
    ref = clf.predict(G[np.ix_(test, ~test)])

    assert (pred[~test] == y[~test]).all()  # Preserve existing labels.
    assert (pred[test] == ref).mean() >= 0.98


def test_mixed_data_reasonable_accuracy():
    ds = _mixed_ds()
    n = ds.n_rows()
    y = np.repeat([0, 1, 2], n // 3)
    test = _split(n, 0.3, 2)
    labels = np.where(test, mixdex.UNLABELED, y)
    pred = np.array(mixdex.knn_classify(ds, labels.tolist(), 5))
    assert (pred[test] == y[test]).mean() > 0.9


def test_balltree_equals_bruteforce(numeric_data):
    ds, y = numeric_data
    test = _split(len(y), 0.4, 3)
    labels = np.where(test, mixdex.UNLABELED, y).tolist()
    a = mixdex.knn_classify(ds, labels, 7, False, mixdex.SearchMethod.BallTree)
    b = mixdex.knn_classify(ds, labels, 7, False, mixdex.SearchMethod.BruteForce)
    assert list(a) == list(b)


def test_few_labeled_uses_all(numeric_data):
    ds, y = numeric_data
    labels = [mixdex.UNLABELED] * len(y)
    labels[0], labels[130] = 0, 1
    pred = mixdex.knn_classify(ds, labels, 10)
    assert set(pred) <= {0, 1}


@pytest.mark.parametrize("bad", ["k0", "len", "neg", "nolabel"])
def test_invalid(numeric_data, bad):
    ds, y = numeric_data
    labels = y.tolist()
    k = 3
    if bad == "k0":
        k = 0
    elif bad == "len":
        labels = labels[:-1]
    elif bad == "neg":
        labels[0] = -5
    elif bad == "nolabel":
        labels = [mixdex.UNLABELED] * len(y)
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_classify(ds, labels, k)


def test_radius_query_excludes_self(numeric_data):
    ds, _ = numeric_data
    tree = mixdex.BallTree(ds, 40)
    for i in (0, 17, 200):
        nb = tree.radius_query(i, 0.2)
        assert i not in [j for j, _ in nb]
        assert i not in [j for j, _ in tree.knn(i, 10)]
