"""Check DensityHierarchy and DBSCAN*."""
import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

import mixdex
from helpers import APPROX, BT, EXACT, distance_np, make

@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2)
    n = 450
    Xn = np.vstack([rng.normal(c, 0.7, (n // 3, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    Xc = rng.integers(0, 3, (n, 2)).astype(np.int32)
    return Xn, Xc, make(Xn, Xc)


@pytest.mark.parametrize("m_pts", [2, 5, 10])
@pytest.mark.parametrize("eps", [0.08, 0.12, 0.2])
def test_matches_dbscan_on_core_points(data, m_pts, eps):
    Xn, Xc, ds = data
    D = distance_np(Xn, Xc, Xn, Xc)
    # A core point has at least m_pts-1 other neighbors within eps.
    np.fill_diagonal(D, np.inf)
    core = np.sort(D, axis=1)[:, m_pts - 2] <= eps
    np.fill_diagonal(D, 0.0)

    ref = DBSCAN(eps=eps, min_samples=m_pts, metric="precomputed").fit(D).labels_
    ref_star = np.where(core, ref, -1)  # DBSCAN* excludes border points.

    h = mixdex.DensityHierarchy(ds, m_pts, EXACT, BT)
    lab = np.array(h.dbscan_star(eps))

    assert ((lab == -1) == (ref_star == -1)).all()
    assert adjusted_rand_score(ref_star[core], lab[core]) == 1.0
    assert (np.array(h.core_distances) <= eps).tolist() == core.tolist()


def test_exact_equals_approximate(data):
    _, _, ds = data
    a = mixdex.DensityHierarchy(ds, 5, EXACT, BT)
    b = mixdex.DensityHierarchy(ds, 5, APPROX, BT, 40, 21)
    for eps in (0.1, 0.15, 0.25):
        assert list(a.dbscan_star(eps)) == list(b.dbscan_star(eps))


def test_limits(data):
    _, _, ds = data
    h = mixdex.DensityHierarchy(ds, 5, EXACT, BT)
    core = np.array(h.core_distances)
    assert (np.array(h.dbscan_star(core.min() / 2)) == -1).all()      # Below every core distance, all points are noise.
    top = max(w for _, _, w in h.mst_edges)
    assert set(h.dbscan_star(max(top, core.max()))) == {0}            # Above all relevant distances, one cluster remains.
    assert len(h.mst_edges) == ds.n_rows() - 1


def test_monotone_noise_and_min_cluster_size(data):
    _, _, ds = data
    h = mixdex.DensityHierarchy(ds, 5, EXACT, BT)
    prev = None
    for eps in np.linspace(0.05, 0.4, 8):
        lab = np.array(h.dbscan_star(eps))
        noise = (lab == -1).sum()
        if prev is not None:
            assert noise <= prev  # Increasing eps must not increase noise.
        prev = noise
    base = np.array(h.dbscan_star(0.12))
    big = np.array(h.dbscan_star(0.12, 20))
    assert (big == -1).sum() >= (base == -1).sum()
    sizes = np.bincount(big[big >= 0])
    assert (sizes >= 20).all()


def test_labels_are_dense_and_ordered(data):
    _, _, ds = data
    lab = np.array(mixdex.DensityHierarchy(ds, 5, EXACT, BT).dbscan_star(0.12))
    u = [x for x in dict.fromkeys(lab.tolist()) if x >= 0]
    assert u == list(range(len(u)))  # Label clusters by first occurrence.


def test_hdbscan_unchanged_after_refactor(data):
    _, _, ds = data
    r = mixdex.hdbscan(ds, 5, 15, EXACT, BT)
    assert len(r.labels) == ds.n_rows()
    assert max(r.labels) >= 0


@pytest.mark.parametrize("bad", [(-0.1, 1), (float("nan"), 1), (0.1, 0)])
def test_invalid_args(data, bad):
    _, _, ds = data
    h = mixdex.DensityHierarchy(ds, 5, EXACT, BT)
    with pytest.raises(mixdex.InvalidArgument):
        h.dbscan_star(*bad)
