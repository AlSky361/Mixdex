"""Check external queries and knn_predict."""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import mixdex
from helpers import BF, BT, METHODS, distance_np, make, unique_lattice_rows

@pytest.fixture(scope="module")
def mixed():
    rng = np.random.default_rng(0)
    n = 400
    Xn = np.vstack([rng.normal(c, 0.5, (n // 4, 3)) for c in ([0] * 3, [3] * 3, [0, 3, 0], [3, 0, 3])])
    Xc = rng.integers(0, 4, (n, 2)).astype(np.int32)
    Qn = rng.normal(1.5, 1.5, (60, 3))
    Qc = rng.integers(0, 6, (60, 2)).astype(np.int32)  # Codes 4 and 5 are absent from training data.
    return Xn, Xc, Qn, Qc, make(Xn, Xc)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [1, 5, 20])
def test_knn_query_matches_numpy(mixed, method, k):
    Xn, Xc, Qn, Qc, ds = mixed
    D = distance_np(Qn, Qc, Xn, Xc)
    res = mixdex.knn_query(ds, Qn, Qc, k, method)
    assert len(res) == len(Qn)
    for i, nb in enumerate(res):
        got = np.array([d for _, d in nb])
        want = np.sort(D[i])[:k]
        assert np.allclose(got, want)  # Neighbor IDs may differ on exact distance ties.
        assert np.allclose(D[i][[j for j, _ in nb]], got)


@pytest.mark.parametrize("method", METHODS)
def test_radius_query_points_matches_numpy(mixed, method):
    Xn, Xc, Qn, Qc, ds = mixed
    D = distance_np(Qn, Qc, Xn, Xc)
    r = 0.25
    res = mixdex.radius_query_points(ds, Qn, Qc, r, method)
    for i, nb in enumerate(res):
        assert sorted(j for j, _ in nb) == sorted(np.nonzero(D[i] <= r)[0].tolist())


def test_balltree_object_methods(mixed):
    Xn, Xc, Qn, Qc, ds = mixed
    tree = mixdex.BallTree(ds, 20)
    a = tree.knn_query(Qn, Qc, 7)
    b = mixdex.knn_query(ds, Qn, Qc, 7, mixdex.SearchMethod.BruteForce)
    for x, y in zip(a, b):
        assert np.allclose([d for _, d in x], [d for _, d in y])


def test_row_of_dataset_as_external_point(mixed):
    Xn, Xc, _, _, ds = mixed
    tree = mixdex.BallTree(ds, 20)
    for i in (0, 123, 399):
        ext = tree.knn_query(Xn[[i]], Xc[[i]], 6)[0]
        assert ext[0][1] == 0.0  # External queries include a matching training point at distance zero.
        inner = tree.knn(i, 5)
        assert np.allclose([d for _, d in ext[1:]], [d for _, d in inner])


def test_numeric_only_and_categorical_only():
    rng = np.random.default_rng(1)
    Xn = rng.normal(size=(150, 2))
    ds = make(Xn, np.zeros((150, 0), dtype=np.int32))
    Q = rng.normal(size=(10, 2))
    res = mixdex.knn_query(ds, Q, None, 3)
    D = distance_np(Q, np.zeros((10, 0)), Xn, np.zeros((150, 0)))
    for i, nb in enumerate(res):
        assert np.allclose([d for _, d in nb], np.sort(D[i])[:3])

    _, Xc = unique_lattice_rows(rng, 150, 0, 4)
    ds = make(np.zeros((150, 0)), Xc)
    Qc = rng.integers(0, 3, (10, 4)).astype(np.int32)
    res = mixdex.knn_query(ds, None, Qc, 3)
    D = distance_np(np.zeros((10, 0)), Qc, np.zeros((150, 0)), Xc)
    for i, nb in enumerate(res):
        assert np.allclose([d for _, d in nb], np.sort(D[i])[:3])


def test_schema_errors(mixed):
    Xn, Xc, Qn, Qc, ds = mixed
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_query(ds, Qn[:, :2], Qc, 3)  # Wrong numeric column count.
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_query(ds, Qn, None, 3)  # Missing categorical block.
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_query(ds, Qn[:10], Qc[:11], 3)  # Mismatched row counts.
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_query(ds, Qn.ravel(), Qc, 3)  # Input is not two-dimensional.


def test_external_query_not_supported_in_mreach_mode(mixed):
    Xn, Xc, Qn, Qc, ds = mixed
    tree = mixdex.BallTree(ds, 20)
    tree.set_core_dist([0.1] * ds.n_rows())
    with pytest.raises(mixdex.InvalidArgument):
        tree.knn_query(Qn, Qc, 3)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [1, 5, 15])
@pytest.mark.parametrize("weighted", [False, True])
def test_knn_predict_matches_sklearn(method, k, weighted):
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [3, 0], [0, 3]], dtype=float)
    y = np.repeat([0, 1, 2], 120)
    Xn = centers[y] + rng.normal(0, 0.8, (360, 2))
    Xc = np.where(rng.random((360, 1)) < 0.8, y[:, None], rng.integers(0, 3, (360, 1))).astype(np.int32)
    test = rng.random(360) < 0.3
    ds_train = make(Xn[~test], Xc[~test])

    pred = np.array(mixdex.knn_predict(ds_train, y[~test].tolist(), Xn[test], Xc[test], k, weighted, method))

    G = distance_np(Xn[test], Xc[test], Xn[~test], Xc[~test])
    clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed",
                               weights="distance" if weighted else "uniform").fit(
        distance_np(Xn[~test], Xc[~test], Xn[~test], Xc[~test]), y[~test])
    ref = clf.predict(G)
    assert (pred == ref).mean() >= 0.98


def test_knn_predict_ignores_unlabeled_rows(mixed):
    Xn, Xc, Qn, Qc, ds = mixed
    labels = [mixdex.UNLABELED] * ds.n_rows()
    labels[3], labels[250] = 0, 1
    pred = mixdex.knn_predict(ds, labels, Qn, Qc, 10)
    assert set(pred) <= {0, 1}
