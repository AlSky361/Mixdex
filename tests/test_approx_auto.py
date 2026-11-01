"""Check automatic Approximate k selection using mutual-reachability prefix lists."""
import pytest

import mixdex

BT = mixdex.SearchMethod.BallTree
APPROX = mixdex.Approximate
M_PTS, MCS = 5, 10


def synth(n, n_clusters, separation):
    p = mixdex.SyntheticParams()
    p.n, p.d_num, p.n_clusters, p.separation = n, 4, n_clusters, separation
    p.p_cat, p.cardinality, p.seed = 0.3, 8, 3
    return mixdex.make_synthetic(p).dataset


def labels(ds, k_opt=-1):
    return list(mixdex.hdbscan(ds, M_PTS, MCS, APPROX, BT, 40, k_opt).labels)


def test_auto_matches_explicit_when_connected_at_m_pts():
    # k_star=m_pts: reuse k_try=10 without a second search.
    ds = synth(300, 4, 0.5)
    assert labels(ds) == labels(ds, 10)


def test_auto_capped_at_sqrt_n():
    # Reuse lists at the sqrt(n)=20 cap.
    ds = synth(400, 25, 4.0)
    assert labels(ds) == labels(ds, 20)


def test_auto_recomputes_when_k_opt_exceeds_k_try():
    # Connected k_try=20 needs one extension to k_opt=24.
    ds = synth(600, 36, 4.0)
    assert labels(ds) == labels(ds, 24)


def test_auto_refuses_when_disconnected_up_to_sqrt_n():
    ds = synth(300, 4, 4.0)  # Separated groups require k_star > sqrt(300).
    with pytest.raises(Exception, match="Exact"):
        labels(ds)


def test_auto_impossible_for_tiny_n():
    ds = synth(10, 2, 1.0)  # sqrt(10) = 3 < m_pts = 5
    with pytest.raises(Exception, match="Exact"):
        labels(ds)


def test_explicit_k_opt_validation_unchanged():
    ds = synth(300, 4, 0.5)
    with pytest.raises(Exception, match="k_opt=3"):
        labels(ds, 3)  # k_opt < m_pts
    with pytest.raises(Exception, match="k_opt=100"):
        labels(ds, 100)  # k_opt > sqrt(n)
