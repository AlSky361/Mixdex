"""Check indexed kNN outlier scores and LOF."""
import numpy as np
import pytest
from sklearn.neighbors import LocalOutlierFactor

import mixdex
from helpers import METHODS, distance_np, make

@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    Xn = np.vstack([rng.normal(c, 0.4, (n // 3, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    Xc = rng.integers(0, 3, (n, 2)).astype(np.int32)
    Xn[:5] += rng.normal(0, 4, (5, 2))  # Include a few outliers.
    Qn = rng.normal(1.5, 2.0, (40, 2))
    Qc = rng.integers(0, 5, (40, 2)).astype(np.int32)
    return Xn, Xc, Qn, Qc, make(Xn, Xc)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [1, 5, 20])
def test_knn_scores_internal(data, method, k):
    Xn, Xc, _, _, ds = data
    D = distance_np(Xn, Xc, Xn, Xc)
    np.fill_diagonal(D, np.inf)
    want = np.sort(D, axis=1)[:, k - 1]
    got = np.array(mixdex.knn_outlier_scores(ds, k, method))
    assert np.allclose(got, want)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [1, 5, 20])
def test_knn_scores_points(data, method, k):
    Xn, Xc, Qn, Qc, ds = data
    D = distance_np(Qn, Qc, Xn, Xc)
    want = np.sort(D, axis=1)[:, k - 1]
    got = np.array(mixdex.knn_outlier_scores_points(ds, Qn, Qc, k, method))
    assert np.allclose(got, want)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [3, 10, 30])
def test_lof_training_matches_sklearn(data, method, k):
    Xn, Xc, _, _, ds = data
    D = distance_np(Xn, Xc, Xn, Xc)
    ref = -LocalOutlierFactor(n_neighbors=k, metric="precomputed").fit(D).negative_outlier_factor_
    lof = mixdex.LOF(ds, k, method)
    assert lof.k == k
    assert np.allclose(lof.scores, ref, rtol=1e-6)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [3, 10, 30])
def test_lof_novelty_matches_sklearn(data, method, k):
    Xn, Xc, Qn, Qc, ds = data
    Dtr = distance_np(Xn, Xc, Xn, Xc)
    Dq = distance_np(Qn, Qc, Xn, Xc)
    sk = LocalOutlierFactor(n_neighbors=k, metric="precomputed", novelty=True).fit(Dtr)
    ref = -sk.score_samples(Dq)
    got = mixdex.LOF(ds, k, method).score(Qn, Qc)
    assert np.allclose(got, ref, rtol=1e-6)


def test_lof_finds_planted_outliers(data):
    _, _, _, _, ds = data
    s = np.array(mixdex.LOF(ds, 15).scores)
    assert set(np.argsort(-s)[:5]) >= {0, 1, 2, 3, 4} or s[:5].mean() > 1.5 * np.median(s)


def test_invalid_k(data):
    _, _, Qn, Qc, ds = data
    n = ds.n_rows()
    for bad in (0, n):
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.LOF(ds, bad)
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.knn_outlier_scores(ds, bad)
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_outlier_scores_points(ds, Qn, Qc, n + 1)
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.LOF(ds, 5).score(Qn[:, :1], Qc)  # Invalid schema.
