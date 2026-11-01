"""Shared test helpers."""
import math

import numpy as np

import mixdex

METHODS = [mixdex.SearchMethod.BallTree, mixdex.SearchMethod.BruteForce]


def _numeric_ds(X):
    n, d = X.shape
    return mixdex.Dataset(
        X.ravel().tolist(), [], n, d, 0,
        [mixdex.FeatureType.Numeric] * d,
        [mixdex.NumericType.Continuous] * d,
        [f"f{i}" for i in range(d)],
    )


def _mixed_ds(seed=1, n=450):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.4, (n // 3, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    cat = np.repeat([0, 1, 2], n // 3)
    cat = np.where(rng.random(n) < 0.15, rng.integers(0, 3, n), cat)
    nf = []
    for k in range(2):
        f = mixdex.NumFeature()
        f.array = X[:, k].tolist()
        f.type = mixdex.NumericType.Continuous
        nf.append(f)
    of = mixdex.CategoricalFeature()
    of.encoded = [int(c) for c in cat]
    return mixdex.Dataset(nf, [of], ["a", "b", "c"])


def _distance_matrix(ds):
    n = ds.n_rows()
    return np.array([[mixdex.distance(ds, i, j) for j in range(n)] for i in range(n)])




def unique_lattice_rows(rng, n, dn, dc, slack=1.5, max_tries=20):
    """Generate distinct lattice rows with frequent coordinate and distance ties.
    
    Choose the alphabet from n and dimension so there are enough unique rows
    without making ties artificially rare."""
    d = dn + dc
    if d == 0:
        raise ValueError("unique_lattice_rows: dn + dc must be > 0")
    card = max(2, math.ceil((n * slack) ** (1.0 / d)))
    for attempt in range(max_tries):
        batch_n = int(n * slack) + attempt * n
        Xn = rng.integers(0, card, (batch_n, dn)).astype(float) if dn else np.zeros((batch_n, 0))
        Xc = (
            rng.integers(0, card, (batch_n, dc)).astype(np.int32)
            if dc
            else np.zeros((batch_n, 0), dtype=np.int32)
        )
        combined = np.hstack([Xn, Xc.astype(float)])
        _, first_idx = np.unique(combined, axis=0, return_index=True)
        if len(first_idx) >= n:
            keep = np.sort(first_idx)[:n]
            return Xn[keep], Xc[keep]
        card += 1
    raise RuntimeError(f"unique_lattice_rows: could not generate {n} unique rows for d={d}")


def make(Xn, Xc):
    """Build a Dataset from numeric and categorical matrices."""
    nf, of = [], []
    for k in range(Xn.shape[1]):
        f = mixdex.NumFeature()
        f.array = Xn[:, k].tolist()
        f.type = mixdex.NumericType.Continuous
        nf.append(f)
    for k in range(Xc.shape[1]):
        f = mixdex.CategoricalFeature()
        f.encoded = [int(v) for v in Xc[:, k]]
        of.append(f)
    return mixdex.Dataset(nf, of, [f"f{i}" for i in range(Xn.shape[1] + Xc.shape[1])])


def distance_np(Qn, Qc, Xn, Xc):
    """Compute equal-weight Composite distances from Q rows to X rows."""
    d = Xn.shape[1] + Xc.shape[1]
    out = np.zeros((len(Qn), len(Xn)))
    for k in range(Xn.shape[1]):
        out += np.abs(Qn[:, [k]] - Xn[:, k][None, :])
    for k in range(Xc.shape[1]):
        out += (Qc[:, [k]] != Xc[:, k][None, :])
    return out / d


EXACT = mixdex.MSTMode.Exact
APPROX = mixdex.MSTMode.Approximate
BT = mixdex.SearchMethod.BallTree
BF = mixdex.SearchMethod.BruteForce
