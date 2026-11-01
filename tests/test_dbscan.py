"""Check indexed DBSCAN."""
import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

import mixdex
from helpers import METHODS, _distance_matrix, _mixed_ds, _numeric_ds

@pytest.fixture(scope="module")
def numeric_ds():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (100, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    return _numeric_ds(X)


@pytest.fixture(scope="module")
def mixed_ds():
    return _mixed_ds()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("eps,min_pts", [(0.03, 4), (0.05, 5), (0.08, 8)])
@pytest.mark.parametrize("ds_name", ["numeric_ds", "mixed_ds"])
def test_dbscan_matches_sklearn(request, ds_name, eps, min_pts, method):
    ds = request.getfixturevalue(ds_name)
    ref = DBSCAN(eps=eps, min_samples=min_pts, metric="precomputed").fit(_distance_matrix(ds)).labels_
    lab = np.array(mixdex.dbscan(ds, eps, min_pts, method))
    # Noise and core structure agree; border assignment may differ with traversal order.
    assert ((lab == -1) == (ref == -1)).all()
    assert adjusted_rand_score(ref, lab) > 0.99


def test_balltree_equals_bruteforce(mixed_ds):
    a = mixdex.dbscan(mixed_ds, 0.15, 8, mixdex.SearchMethod.BallTree)
    b = mixdex.dbscan(mixed_ds, 0.15, 8, mixdex.SearchMethod.BruteForce)
    assert list(a) == list(b)


def test_eps_zero_all_noise_or_duplicates(numeric_ds):
    lab = mixdex.dbscan(numeric_ds, 0.0, 2)
    assert all(l == -1 for l in lab)


def test_min_pts_one_no_noise(numeric_ds):
    lab = mixdex.dbscan(numeric_ds, 0.01, 1)
    assert -1 not in set(lab)


@pytest.mark.parametrize("eps,min_pts", [(-0.1, 5), (float("nan"), 5), (0.1, 0)])
def test_invalid_args(numeric_ds, eps, min_pts):
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.dbscan(numeric_ds, eps, min_pts)
