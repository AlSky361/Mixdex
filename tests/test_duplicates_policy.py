"""Reject exactly equal rows; allow nearby rows and treat signed zeros as equal."""
import numpy as np
import pytest
from helpers import make

import mixdex


def _cat0(n):
    return np.zeros((n, 0), dtype=np.int32)


def test_exact_duplicate_rejected():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    with pytest.raises(mixdex.InvalidArgument, match="duplicate.*rows 0 and 2"):
        make(X, _cat0(3))


def test_negative_zero_is_duplicate_of_zero():
    X = np.array([[0.0, 5.0], [1.0, 1.0], [-0.0, 5.0]])
    with pytest.raises(mixdex.InvalidArgument, match="duplicate"):
        make(X, _cat0(3))


@pytest.mark.parametrize("rel", [1e-12, 1e-15, 2.3e-16])
def test_near_duplicates_are_accepted(rel):
    X = np.array([[1000.0, 2000.0], [1000.0 * (1 + rel), 2000.0], [5.0, 6.0]])
    ds = make(X, _cat0(3))
    assert ds.n_rows() == 3


def test_near_duplicates_do_not_break_clustering():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 2))
    X = np.vstack([base, base + 1e-13 * rng.normal(size=base.shape)])   # Each point has a near neighbor.
    ds = mixdex.scale(make(X, _cat0(len(X))))
    d = mixdex.distance(ds, 0, 60)
    assert 0.0 < d < 1e-9
