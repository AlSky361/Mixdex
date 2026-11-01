"""Bounded Composite search must match BruteForce, including weights and ties."""
import numpy as np
import pytest

import mixdex
from helpers import make, unique_lattice_rows

BT, BF = mixdex.SearchMethod.BallTree, mixdex.SearchMethod.BruteForce


def _dataset(seed, n=400, d_num=4, d_cat=3, ties=False, weights=None):
    rng = np.random.default_rng(seed)
    # Use distinct lattice rows to retain many exact distance ties without duplicate rows.
    if ties or d_num == 0:
        Xn, Xc = unique_lattice_rows(rng, n, d_num, d_cat)
    else:
        Xn = np.vstack([rng.normal(c, 0.5, (n // 4, d_num)) for c in (0, 2, 4, 6)])
        Xc = rng.integers(0, 3, (len(Xn), d_cat)).astype(np.int32)
    ds = make(Xn, Xc)
    if weights is not None:
        ds.set_weights(weights)
        # Zero weights remove columns and any duplicates created by projection.
        Xn = np.asarray(ds.num_data()).reshape(ds.n_rows(), ds.n_num())
        Xc = np.asarray(ds.cat_data(), dtype=np.int32).reshape(ds.n_rows(), ds.n_cat())
    return ds, Xn, Xc


CASES = [
    dict(seed=1),
    dict(seed=2, ties=True),
    dict(seed=3, weights=[1.0, 2.0, 0.5, 3.0, 1.0, 0.0, 2.0]),
    dict(seed=4, ties=True, weights=[0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5]),
    dict(seed=5, d_num=1, d_cat=1),
    dict(seed=6, d_num=6, d_cat=0),
    dict(seed=7, d_num=0, d_cat=5),
]


def _q(Xn, Xc, i):
    return (Xn[i:i + 1] if Xn.shape[1] else None, Xc[i:i + 1] if Xc.shape[1] else None)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("leaf", [1, 7, 40])
def test_knn_and_radius_identical(case, leaf):
    ds, Xn, Xc = _dataset(**case)
    bt, bf = mixdex.Index(ds, BT, leaf), mixdex.Index(ds, BF)
    for k in (1, 5, 20):
        a, b = mixdex.knn_all(bt, k), mixdex.knn_all(bf, k)
        # Distances match exactly; IDs may differ only in the documented rounding-tie cases.
        assert [[d for _, d in row] for row in a] == [[d for _, d in row] for row in b]
        if not case.get("ties"):
            assert a == b
    for i in (0, 17, 133):
        for r in (0.05, 0.2, 0.5, 1.0):
            a = mixdex.radius_query_points(bt, *_q(Xn, Xc, i), r)
            b = mixdex.radius_query_points(bf, *_q(Xn, Xc, i), r)
            assert a == b


@pytest.mark.parametrize("leaf", [1, 7, 40])
def test_ties_and_zero_radius_identical(leaf):
    ds, Xn, Xc = _dataset(seed=2, ties=True)
    bt, bf = mixdex.Index(ds, BT, leaf), mixdex.Index(ds, BF)
    assert mixdex.knn_all(bt, 20) == mixdex.knn_all(bf, 20)
    ds, Xn, Xc = _dataset(seed=6, d_cat=0)
    bt, bf = mixdex.Index(ds, BT, leaf), mixdex.Index(ds, BF)
    assert mixdex.radius_query_points(bt, *_q(Xn, Xc, 0), 0.0) == \
        mixdex.radius_query_points(bf, *_q(Xn, Xc, 0), 0.0)


@pytest.mark.parametrize("case", CASES)
def test_external_points_identical(case):
    ds, Xn, Xc = _dataset(**case)
    rng = np.random.default_rng(99)
    Qn = rng.normal(2.0, 2.0, (30, Xn.shape[1])) if Xn.shape[1] else None
    Qc = rng.integers(0, 3, (30, Xc.shape[1])).astype(np.int32) if Xc.shape[1] else None
    bt, bf = mixdex.Index(ds, BT, 10), mixdex.Index(ds, BF)
    assert mixdex.knn_query(bt, Qn, Qc, 7) == mixdex.knn_query(bf, Qn, Qc, 7)
    assert mixdex.radius_query_points(bt, Qn, Qc, 0.3) == mixdex.radius_query_points(bf, Qn, Qc, 0.3)


@pytest.mark.parametrize("case", CASES)
def test_algorithms_identical(case):
    ds, *_ = _dataset(**case)
    bt, bf = mixdex.Index(ds, BT, 10), mixdex.Index(ds, BF)
    assert mixdex.dbscan(bt, 0.25, 4) == mixdex.dbscan(bf, 0.25, 4)
    assert mixdex.core_distances(bt, 6) == mixdex.core_distances(bf, 6)
    assert mixdex.knn_outlier_scores(bt, 5) == mixdex.knn_outlier_scores(bf, 5)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("leaf", [1, 7, 40])
def test_lower_bound_never_exceeds_distance(case, leaf):
    """Check node bounds against every contained point by direct double comparison."""
    ds, Xn, Xc = _dataset(**case)
    t = mixdex.BallTree(ds, leaf)
    idx, nodes = t.indices, t.nodes
    for q in range(0, t.n_rows(), 9):
        for i, nd in enumerate(nodes):
            md = min(mixdex.distance(ds, q, int(p)) for p in idx[nd.low:nd.high])
            assert t.lower_bound(q, i) <= md


@pytest.mark.parametrize("case", CASES)
def test_mreach_node_lower_bound_exact(case):
    """Check dual-tree mreach bounds against every leaf-pair distance without tolerance."""
    ds, *_ = _dataset(**case)
    core = mixdex.core_distances(ds, 5)
    assert mixdex.check_lb_violations(ds, core, 8) == []


def test_mreach_node_lower_bound_exact_random():
    rng = np.random.default_rng(31)
    for trial in range(12):
        n = int(rng.integers(40, 120))
        dn = int(rng.integers(0, 8))
        dc = int(rng.integers(0, 4))
        if dn + dc == 0:
            dn = 2
        scale = 10.0 ** rng.integers(-4, 5)
        if trial % 2 or dn == 0:  # Use ties or categorical-only data.
            Xn, Xc = unique_lattice_rows(rng, n, dn, dc)
            Xn = Xn * scale
        else:
            Xn = rng.normal(0, 1, (n, dn)) * scale
            Xc = rng.integers(0, 3, (n, dc)).astype(np.int32)
        ds = make(Xn, Xc)
        core = mixdex.core_distances(ds, int(rng.integers(2, 9)))
        assert mixdex.check_lb_violations(ds, core, int(rng.choice([1, 4, 10]))) == []


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("min_pts", [1, 2, 5, 12])
@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3, 0.8])
def test_dbscan_grid_matches_bruteforce(case, min_pts, eps):
    """Compare BallTree existence/ID queries with BruteForce defaults."""
    ds, *_ = _dataset(**case)
    bt, bf = mixdex.Index(ds, BT, 10), mixdex.Index(ds, BF)
    assert mixdex.dbscan(bt, eps, min_pts) == mixdex.dbscan(bf, eps, min_pts)


def test_default_leaf_size():
    ds, *_ = _dataset(seed=1)
    assert mixdex.Index(ds).leaf_size == 40
