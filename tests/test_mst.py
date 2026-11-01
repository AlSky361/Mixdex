"""Compare dual-tree and full Boruvka mutual-reachability MST weights.

Tied MST edge choices may differ while their weight multisets agree."""
import numpy as np
import pytest

import mixdex
from helpers import make, unique_lattice_rows


def _total(edges):
    return float(sum(e.w for e in edges))


def _datasets():
    rng = np.random.default_rng(5)
    for trial in range(8):
        n = int(rng.integers(30, 140))
        dn = int(rng.integers(0, 6))
        dc = int(rng.integers(0, 3))
        if dn + dc == 0:
            dn = 2
        if trial % 2:  # Unique rows with many coordinate and distance ties.
            Xn, Xc = unique_lattice_rows(rng, n, dn, dc)
        elif dn == 0:
            # Categorical-only rows require explicit uniqueness.
            Xn, Xc = unique_lattice_rows(rng, n, dn, dc)
        else:
            Xn = np.vstack([rng.normal(c, 0.5, (n // 3 + 1, dn)) for c in (0, 2, 5)])[:n]
            Xc = rng.integers(0, 3, (n, dc)).astype(np.int32)
        yield trial, make(Xn, Xc)


@pytest.mark.parametrize("leaf", [1, 5, 20])
def test_dt_boruvka_weight_equals_full_boruvka(leaf):
    for trial, ds in _datasets():
        n = ds.n_rows()
        core = mixdex.core_distances(ds, 4)
        full = mixdex.boruvka_edges(mixdex.full_mreach_edges(ds, core), n)
        dt = mixdex.dt_boruvka(ds, core, leaf)
        assert len(dt) == n - 1 == len(full)
        assert _total(dt) == pytest.approx(_total(full), rel=1e-12, abs=1e-12), trial


def test_dt_boruvka_connects_all_points():
    for _, ds in _datasets():
        n = ds.n_rows()
        core = mixdex.core_distances(ds, 3)
        edges = mixdex.dt_boruvka(ds, core, 8)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in edges:
            parent[find(e.u)] = find(e.v)
        assert len({find(i) for i in range(n)}) == 1


def _connected(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        parent[find(e.u)] = find(e.v)
    return len({find(i) for i in range(n)}) == 1


@pytest.mark.parametrize("leaf", [1, 8])
def test_dt_boruvka_nondeterministic_mode_is_still_a_valid_mst(leaf):
    """Nondeterministic MSTs may vary on ties but must retain n-1 edges, connectivity
    and the deterministic MST weight multiset."""
    for trial, ds in _datasets():
        n = ds.n_rows()
        core = mixdex.core_distances(ds, 4)
        det = mixdex.dt_boruvka(ds, core, leaf)
        for _ in range(3):
            fast = mixdex.dt_boruvka(ds, core, leaf, deterministic=False)
            assert len(fast) == n - 1, trial
            assert _connected(n, fast), trial
            # MST weight multisets agree even if tied edges differ.
            assert sorted(e.w for e in fast) == sorted(e.w for e in det), trial


def _partition(labels):
    """Compare labels up to renaming while preserving noise=-1."""
    seen, out = {}, []
    for v in labels:
        out.append(-1 if v < 0 else seen.setdefault(v, len(seen)))
    return out


def test_hdbscan_result_does_not_depend_on_mst_determinism_flag():
    """Canonical HDBSCAN labels and GLOSH must match across tied MST choices.
    
    Allow rounding differences in summed stabilities."""
    for trial, ds in _datasets():
        ref = mixdex.hdbscan(ds, 4, 5, mixdex.HDBSCANMode.Exact, mixdex.SearchMethod.BallTree)
        for _ in range(3):
            got = mixdex.hdbscan(ds, 4, 5, mixdex.HDBSCANMode.Exact, mixdex.SearchMethod.BallTree,
                                 deterministic_mst=False)
            assert list(got.labels) == list(ref.labels), trial
            # Stability sum order depends on the MST; allow rounding differences.
            assert len(got.cluster_stabilities) == len(ref.cluster_stabilities), trial
            assert np.allclose(list(got.cluster_stabilities), list(ref.cluster_stabilities),
                               rtol=1e-12, atol=0.0), trial
            assert list(got.outlier_scores) == list(ref.outlier_scores), trial


def test_hdbscan_labels_are_canonically_numbered():
    """Canonical labels follow first point occurrence, preserving noise=-1.
    
    The stability array has one entry per cluster."""
    for trial, ds in _datasets():
        r = mixdex.hdbscan(ds, 4, 5, mixdex.HDBSCANMode.Exact, mixdex.SearchMethod.BallTree)
        labels = list(r.labels)
        assert labels == _partition(labels), trial
        n_clusters = len({v for v in labels if v >= 0})
        assert len(list(r.cluster_stabilities)) == n_clusters, trial
