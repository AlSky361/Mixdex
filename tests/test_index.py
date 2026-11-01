"""Reuse Index across algorithms and compare with Dataset overloads."""
import gc

import numpy as np
import pytest

import mixdex
from helpers import METHODS, make


@pytest.fixture()
def mixed():
    rng = np.random.default_rng(3)
    n = 300
    Xn = np.vstack([rng.normal(c, 0.4, (n // 3, 3)) for c in ([0] * 3, [3] * 3, [0, 3, 0])])
    Xc = rng.integers(0, 4, (n, 2)).astype(np.int32)
    Qn = rng.normal(1.5, 1.5, (40, 3))
    Qc = rng.integers(0, 4, (40, 2)).astype(np.int32)
    labels = [i % 3 if i % 4 else mixdex.UNLABELED for i in range(n)]
    return Xn, Xc, Qn, Qc, labels, make(Xn, Xc)


@pytest.mark.parametrize("method", METHODS)
def test_properties(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method, 25)
    assert idx.n_rows == ds.n_rows()
    assert idx.method == method
    assert idx.leaf_size == 25


def test_int_max_leaf_size_does_not_overflow(mixed):
    ds=mixed[-1]
    index=mixdex.Index(ds, mixdex.SearchMethod.BallTree, 2**31-1)
    assert mixdex.knn_all(index, 3) == mixdex.knn_all(ds, 3, mixdex.SearchMethod.BruteForce)


def test_int_max_category_code_is_rejected():
    c=mixdex.CategoricalFeature()
    c.encoded=[0, 2**31-1]
    with pytest.raises(ValueError, match="max_code"):
        mixdex.Dataset([], [c], ['category'])


@pytest.mark.parametrize("method", METHODS)
def test_dbscan_same(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method)
    for eps in (0.1, 0.2, 0.3):  # Reuse one index across calls.
        assert mixdex.dbscan(idx, eps, 5) == mixdex.dbscan(ds, eps, 5, method)


@pytest.mark.parametrize("method", METHODS)
def test_knn_same(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method)
    assert mixdex.knn(idx, 7, 6) == mixdex.knn(ds, 7, 6, method)
    assert mixdex.knn_all(idx, 4) == mixdex.knn_all(ds, 4, method)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weighted", [False, True])
def test_classify_predict_same(mixed, method, weighted):
    Xn, Xc, Qn, Qc, labels, ds = mixed
    idx = mixdex.Index(ds, method)
    assert mixdex.knn_classify(idx, labels, 5, weighted) == mixdex.knn_classify(ds, labels, 5, weighted, method)
    assert mixdex.knn_predict(idx, labels, Qn, Qc, 5, weighted) == mixdex.knn_predict(
        ds, labels, Qn, Qc, 5, weighted, method)


@pytest.mark.parametrize("method", METHODS)
def test_external_and_outliers_same(mixed, method):
    Xn, Xc, Qn, Qc, labels, ds = mixed
    idx = mixdex.Index(ds, method)
    assert mixdex.knn_query(idx, Qn, Qc, 4) == mixdex.knn_query(ds, Qn, Qc, 4, method)
    assert mixdex.radius_query_points(idx, Qn, Qc, 0.2) == mixdex.radius_query_points(ds, Qn, Qc, 0.2, method)
    assert mixdex.knn_outlier_scores(idx, 5) == mixdex.knn_outlier_scores(ds, 5, method)
    assert mixdex.knn_outlier_scores_points(idx, Qn, Qc, 5) == mixdex.knn_outlier_scores_points(ds, Qn, Qc, 5, method)


@pytest.mark.parametrize("method", METHODS)
def test_lof_on_index_same(mixed, method):
    Xn, Xc, Qn, Qc, labels, ds = mixed
    idx = mixdex.Index(ds, method)
    a = mixdex.LOF(idx, 8)
    b = mixdex.LOF(ds, 8, method)
    assert a.scores == b.scores
    assert a.score(Qn, Qc) == b.score(Qn, Qc)
    # Reuse the index with another LOF neighborhood size.
    assert mixdex.LOF(idx, 12).scores == mixdex.LOF(ds, 12, method).scores


def test_index_keeps_dataset_alive(mixed):
    Xn, Xc, Qn, Qc, labels, _ = mixed
    idx = mixdex.Index(make(Xn, Xc))  # Keep a temporary Dataset alive through its index.
    gc.collect()
    assert len(mixdex.dbscan(idx, 0.2, 5)) == len(Xn)
    lof = mixdex.LOF(idx, 5)
    del idx
    gc.collect()
    assert len(lof.scores) == len(Xn)


@pytest.mark.parametrize("method", METHODS)
def test_stale_weights_detected(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method)
    mixdex.dbscan(idx, 0.2, 5)
    ds.set_weights([1.0, 2.0, 1.0, 1.0, 3.0])
    with pytest.raises(Exception, match="rebuild the index"):
        mixdex.dbscan(idx, 0.2, 5)
    # A rebuilt index must work after weight changes.
    assert len(mixdex.dbscan(mixdex.Index(ds, method), 0.2, 5)) == ds.n_rows()


@pytest.mark.parametrize("method", METHODS)
def test_zero_weights_require_rebuilding_index(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method)
    ds.set_weights([1.0, 0.0, 1.0, 1.0, 0.0])
    assert ds.n_num() == 2 and ds.n_cat() == 1
    with pytest.raises(Exception, match="rebuild the index"):
        mixdex.knn_all(idx, 4)
    fresh = mixdex.Index(ds, method)
    assert mixdex.knn_all(fresh, 4) == mixdex.knn_all(ds, 4, method)


@pytest.mark.parametrize("method", METHODS)
def test_core_graph_findk_same(mixed, method):
    ds = mixed[-1]
    idx = mixdex.Index(ds, method)
    assert mixdex.core_distances(idx, 6) == mixdex.core_distances(ds, 6, method)
    assert mixdex.find_k_connected(idx, 5, 30) == mixdex.find_k_connected(ds, 5, 30, method)
    g_i, g_d = mixdex.knn_graph(idx, 4), mixdex.knn_graph(ds, 4, method)
    key = lambda e: (e.u, e.v, e.w)
    assert sorted(map(key, g_i.get_edges())) == sorted(map(key, g_d.get_edges()))


def test_dataset_overload_signature_unchanged(mixed):
    """Keep positional and keyword method/leaf_size support in Dataset overloads."""
    ds = mixed[-1]
    a = mixdex.dbscan(ds, 0.2, 5, mixdex.SearchMethod.BruteForce, 20)
    b = mixdex.dbscan(ds, eps=0.2, min_pts=5, method=mixdex.SearchMethod.BruteForce, leaf_size=20)
    c = mixdex.dbscan(index=mixdex.Index(ds), eps=0.2, min_pts=5)
    assert a == b == c


def _reference_k_connected(ds, method, m_pts, k_max):
    """Return the smallest connected k in [m_pts, k_max], or -1."""
    for k in range(m_pts, k_max + 1):
        if mixdex.knn_graph(ds, k, method).is_connected():
            return k
    return -1


@pytest.mark.parametrize("n,n_clusters,separation,k_max", [
    (300, 4, 0.5, 30),   # The first iteration already connects at m_pts.
    (120, 4, 4.0, 60),   # Double toward cluster size, then cap at k_max.
    (300, 4, 4.0, 20),   # Clusters of 75 need k_star > k_max; return -1.
])
@pytest.mark.parametrize("method", METHODS)
def test_find_k_connected_matches_definition(n, n_clusters, separation, k_max, method):
    p = mixdex.SyntheticParams()
    p.n, p.d_num, p.n_clusters, p.separation = n, 4, n_clusters, separation
    p.p_cat, p.cardinality, p.seed = 0.3, 8, 3
    ds = mixdex.make_synthetic(p).dataset
    assert mixdex.find_k_connected(ds, 5, k_max, method) == _reference_k_connected(ds, method, 5, k_max)
