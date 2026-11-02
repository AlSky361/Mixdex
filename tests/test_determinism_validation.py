"""Check deterministic index answers and input validation."""
import subprocess
import sys

import numpy as np
import pytest
from helpers import BF, BT, METHODS, distance_np, make, unique_lattice_rows

import mixdex


@pytest.mark.parametrize("i,j", [(-1, 0), (0, -1), (3, 0), (0, 3)])
def test_pairwise_distances_reject_invalid_rows(i, j):
    data = make(np.array([[0.], [1.], [2.]]), np.zeros((3, 0), dtype=np.int32))
    with pytest.raises(ValueError, match="index out of range"):
        mixdex.distance(data, i, j)
    with pytest.raises(ValueError, match="index out of range"):
        mixdex.mreach_distance([0., 1., 2.], data, i, j)


@pytest.mark.parametrize("core", [[], [0., 1.], [0., 1., 2., 3.],
                                  [-1., 0., 0.], [np.nan, 0., 0.], [0., np.inf, 0.]])
def test_pairwise_mreach_rejects_invalid_core(core):
    data = make(np.array([[0.], [1.], [2.]]), np.zeros((3, 0), dtype=np.int32))
    with pytest.raises(ValueError):
        mixdex.mreach_distance(core, data, 0, 1)


def test_pairwise_mreach_keeps_the_distance_definition():
    data = make(np.array([[0.], [1.], [2.]]), np.zeros((3, 0), dtype=np.int32))
    core = [0., 3., 0.]
    assert mixdex.mreach_distance(core, data, 0, 1) == max(core[0], core[1], mixdex.distance(data, 0, 1))
    assert mixdex.mreach_distance(core, data, 1, 1) == core[1]


@pytest.mark.parametrize("u,v", [(-1, 0), (0, -1), (3, 0), (0, 3)])
def test_graph_rejects_invalid_vertices_without_adding_edges(u, v):
    graph = mixdex.Graph(3)
    with pytest.raises(ValueError, match="vertex out of range"):
        graph.add_edge(u, v, 1.)
    assert graph.get_edges() == []


def test_graph_rejects_negative_size():
    with pytest.raises(ValueError, match="nonnegative"):
        mixdex.Graph(-1)


@pytest.fixture(scope="module")
def tied():
    """Categorical-only unique lattice rows exercise frequent distance ties."""
    rng = np.random.default_rng(0)
    _, Xc = unique_lattice_rows(rng, 200, 0, 3)
    Q = rng.integers(0, 2, (30, 3)).astype(np.int32)
    return Xc, Q, make(np.zeros((200, 0)), Xc)


@pytest.fixture(scope="module")
def mixed_tied():
    """Mixed unique lattice rows exercise rounded numeric and categorical ties."""
    rng = np.random.default_rng(1)
    Xn, Xc = unique_lattice_rows(rng, 250, 1, 3)
    Xn = Xn * 0.1 - 1.0  # Match the scale of rounded normal samples.
    Qn = np.round(rng.normal(size=(30, 1)), 1)
    Qc = rng.integers(0, 3, (30, 3)).astype(np.int32)
    return Xn, Xc, Qn, Qc, make(Xn, Xc)


# BallTree and BruteForce must agree, including tied neighbor IDs.

@pytest.mark.parametrize("k", [1, 5, 40])
def test_knn_query_identical_indices_on_ties(tied, k):
    Xc, Q, ds = tied
    a = mixdex.knn_query(ds, None, Q, k, BT)
    b = mixdex.knn_query(ds, None, Q, k, BF)
    assert a == b


def test_knn_answer_is_sorted_by_distance_then_index(tied):
    Xc, Q, ds = tied
    for nb in mixdex.knn_query(ds, None, Q, 25, BT):
        keys = [(d, j) for j, d in nb]
        assert keys == sorted(keys)


def test_knn_matches_lexicographic_reference(mixed_tied):
    Xn, Xc, Qn, Qc, ds = mixed_tied
    D = distance_np(Qn, Qc, Xn, Xc)
    for method in METHODS:
        res = mixdex.knn_query(ds, Qn, Qc, 12, method)
        for i, nb in enumerate(res):
            order = sorted(range(len(D[i])), key=lambda j: (D[i][j], j))[:12]
            assert [j for j, _ in nb] == order


@pytest.mark.parametrize("method", METHODS)
def test_indexed_knn_matches_reference_and_excludes_self(mixed_tied, method):
    Xn, Xc, _, _, ds = mixed_tied
    D = distance_np(Xn, Xc, Xn, Xc)
    tree = mixdex.BallTree(ds, 10)
    for i in (0, 77, 249):
        want = [j for j in sorted(range(len(D)), key=lambda j: (D[i][j], j)) if j != i][:8]
        assert [j for j, _ in tree.knn(i, 8)] == want


def test_radius_identical_and_sorted(tied):
    Xc, Q, ds = tied
    a = mixdex.radius_query_points(ds, None, Q, 0.34, BT)
    b = mixdex.radius_query_points(ds, None, Q, 0.34, BF)
    assert a == b
    for nb in a:
        keys = [(d, j) for j, d in nb]
        assert keys == sorted(keys)


def test_downstream_results_identical_across_indexes(tied, mixed_tied):
    Xc, Q, ds = tied
    labels = ([0] * 100) + ([1] * 100)
    for w in (False, True):
        assert list(mixdex.knn_predict(ds, labels, None, Q, 7, w, BT)) == \
               list(mixdex.knn_predict(ds, labels, None, Q, 7, w, BF))
    for k in (3, 10):
        assert list(mixdex.LOF(ds, k, BT).scores) == list(mixdex.LOF(ds, k, BF).scores)
        assert list(mixdex.knn_outlier_scores(ds, k, BT)) == list(mixdex.knn_outlier_scores(ds, k, BF))
    for eps, mp in ((0.34, 3), (0.34, 8), (0.67, 5)):
        assert list(mixdex.dbscan(ds, eps, mp, BT)) == list(mixdex.dbscan(ds, eps, mp, BF))  # Include boundary distances.
    Xn, Xc2, Qn, Qc, ds2 = mixed_tied
    assert list(mixdex.dbscan(ds2, 0.2, 4, BT)) == list(mixdex.dbscan(ds2, 0.2, 4, BF))
    assert list(mixdex.LOF(ds2, 10, BT).scores) == list(mixdex.LOF(ds2, 10, BF).scores)


def test_repeatable_across_runs(tied):
    Xc, Q, ds = tied
    first = mixdex.knn_query(ds, None, Q, 9, BT)
    for _ in range(5):
        assert mixdex.knn_query(ds, None, Q, 9, BT) == first


# Input validation

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [0, -1, -100])
def test_bad_k_raises_not_aborts(tied, method, k):
    Xc, Q, ds = tied
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_query(ds, None, Q, k, method)
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.knn_predict(ds, [0] * 200, None, Q, k, False, method)
    tree = mixdex.BallTree(ds, 10)
    with pytest.raises(mixdex.InvalidArgument):
        tree.knn_batch(k)
    with pytest.raises(mixdex.InvalidArgument):
        tree.knn(0, k)
    with pytest.raises(mixdex.InvalidArgument):
        tree.knn_query(None, Q, k)


def test_k_larger_than_n_returns_everything(tied):
    Xc, Q, ds = tied
    assert all(len(nb) == 200 for nb in mixdex.knn_query(ds, None, Q, 1000, BT))
    assert all(len(nb) == 199 for nb in mixdex.BallTree(ds, 10).knn_batch(1000))  # Exclude the query point itself.


def test_nan_and_inf_in_query_rejected(mixed_tied):
    Xn, Xc, Qn, Qc, ds = mixed_tied
    for bad in (np.nan, np.inf, -np.inf):
        Q = Qn.copy()
        Q[3, 0] = bad
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.knn_query(ds, Q, Qc, 3, BT)
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.radius_query_points(ds, Q, Qc, 0.2, BF)
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.LOF(ds, 5).score(Q, Qc)  # Previously accepted invalid input must now fail.
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.knn_predict(ds, [0] * 250, Q, Qc, 3)
        with pytest.raises(mixdex.InvalidArgument):
            mixdex.knn_outlier_scores_points(ds, Q, Qc, 3)


def test_radius_nan_rejected_negative_empty(mixed_tied):
    Xn, Xc, Qn, Qc, ds = mixed_tied
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.radius_query_points(ds, Qn, Qc, float("nan"))
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.BallTree(ds, 10).radius_query(0, float("nan"))
    assert all(nb == [] for nb in mixdex.radius_query_points(ds, Qn, Qc, -1.0))


def test_empty_batch(mixed_tied):
    Xn, Xc, Qn, Qc, ds = mixed_tied
    assert mixdex.knn_query(ds, Qn[:0], Qc[:0], 3) == []


def test_python_arguments_are_required():
    # Require explicit k, radius and query blocks.
    with pytest.raises(TypeError):
        mixdex.knn_query()
    rng = np.random.default_rng(0)
    ds = make(rng.normal(size=(20, 2)), np.zeros((20, 0), dtype=np.int32))
    with pytest.raises(TypeError):
        mixdex.knn_query(ds, rng.normal(size=(2, 2)), None)  # Missing k.
    with pytest.raises(TypeError):
        mixdex.radius_query_points(ds, rng.normal(size=(2, 2)), None)  # Missing radius.


def test_old_and_new_enum_names_are_the_same():
    assert mixdex.HDBSCANMode is mixdex.MSTMode
    assert mixdex.HDBSCANMode.Exact == mixdex.MSTMode.Exact


# Feature weights

def test_weights_are_respected_by_external_queries():
    rng = np.random.default_rng(4)
    Xn = rng.normal(size=(120, 2))
    Xc = rng.integers(0, 3, (120, 1)).astype(np.int32)
    ds = make(Xn, Xc)
    w = [3.0, 1.0, 2.0]
    ds.set_weights(w)
    Qn = rng.normal(size=(10, 2))
    Qc = rng.integers(0, 3, (10, 1)).astype(np.int32)
    W = sum(w)
    D = (w[0] * np.abs(Qn[:, [0]] - Xn[:, 0][None, :]) +
         w[1] * np.abs(Qn[:, [1]] - Xn[:, 1][None, :]) +
         w[2] * (Qc[:, [0]] != Xc[:, 0][None, :])) / W
    for method in METHODS:
        got = mixdex.knn_query(ds, Qn, Qc, 5, method)
        for i, nb in enumerate(got):
            order = sorted(range(120), key=lambda j: (D[i][j], j))[:5]
            assert [j for j, _ in nb] == order
            assert np.allclose([d for _, d in nb], D[i][order])


@pytest.mark.parametrize("n", [-1, -2])
def test_boruvka_edges_rejects_negative_vertex_count(n):
    with pytest.raises(ValueError, match="nonnegative"):
        mixdex.boruvka_edges([], n)


@pytest.mark.parametrize("u,v", [(-1, 0), (0, -2), (2, 0)])
def test_boruvka_edges_validates_vertices(u, v):
    with pytest.raises(ValueError, match="out of range"):
        mixdex.boruvka_edges([mixdex.Edge(u, v, 1.)], 2)


def test_boruvka_edges_rejects_nan_and_accepts_negative_weights():
    with pytest.raises(ValueError, match="NaN"):
        mixdex.boruvka_edges([mixdex.Edge(0, 1, float("nan"))], 2)
    assert mixdex.boruvka_edges([mixdex.Edge(0, 1, -2.)], 2)[0].w == -2.


@pytest.mark.parametrize("n", [2**31, -(2**31)-1, 2**64])
def test_binding_rejects_unrepresentable_sizes(n):
    with pytest.raises((TypeError, OverflowError)):
        mixdex.Graph(n)
    with pytest.raises((TypeError, OverflowError)):
        mixdex.boruvka_edges([], n)
