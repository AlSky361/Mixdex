"""Check Dataset validation, scaling and core-distance definitions."""
import numpy as np
import pytest
from helpers import APPROX, BF, BT, EXACT, distance_np, make
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

import mixdex


# Dataset validation

def _feats(Xn, Xc):
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
    return nf, of, [f"f{i}" for i in range(Xn.shape[1] + Xc.shape[1])]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_rejected(bad):
    rng = np.random.default_rng(0)
    Xn = rng.normal(size=(30, 3))
    Xn[7, 2] = bad
    with pytest.raises(mixdex.InvalidArgument, match="строка 7, числовой признак 2"):
        mixdex.Dataset(*_feats(Xn, np.zeros((30, 0), dtype=np.int32)))


def test_negative_category_code_rejected():
    rng = np.random.default_rng(0)
    Xc = rng.integers(0, 3, (30, 2)).astype(np.int32)
    Xc[11, 1] = -1
    with pytest.raises(mixdex.InvalidArgument, match="строка 11, категориальный признак 1"):
        mixdex.Dataset(*_feats(np.zeros((30, 0)), Xc))


def test_single_pass_reports_all_violations_at_once():
    rng = np.random.default_rng(0)
    Xn = rng.normal(size=(30, 2))
    Xn[3, 0] = np.nan
    Xn[9, 1] = np.inf
    Xc = rng.integers(0, 3, (30, 1)).astype(np.int32)
    Xc[5, 0] = -2
    with pytest.raises(mixdex.InvalidArgument) as e:
        mixdex.Dataset(*_feats(Xn, Xc))
    msg = str(e.value)
    assert "2 шт." in msg and "строка 3, числовой признак 0" in msg  # Both NaN and infinity are reported.
    assert "1 шт." in msg and "строка 5" in msg                       # The category error is also reported.


def test_flat_constructor_also_validates():
    X = np.array([[0.0, 1.0], [np.nan, 2.0], [3.0, 4.0]])
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.Dataset(X.ravel().tolist(), [], 3, 2, 0,
                       [mixdex.FeatureType.Numeric] * 2, [mixdex.NumericType.Continuous] * 2, ["a", "b"])


def test_valid_dataset_still_builds_and_counts_categories():
    rng = np.random.default_rng(0)
    Xc = np.stack([rng.integers(0, 3, 40), rng.integers(0, 5, 40)], axis=1).astype(np.int32)
    ds = make(rng.normal(size=(40, 2)), Xc)
    assert ds.n_rows() == 40 and ds.n_num() == 2 and ds.n_cat() == 2


# Scaling

@pytest.fixture(scope="module")
def sdata():
    rng = np.random.default_rng(3)
    n = 200
    Xn = np.column_stack([rng.normal(5, 3, n), rng.exponential(2, n)])
    Xc = rng.integers(0, 3, (n, 2)).astype(np.int32)
    return Xn, Xc, make(Xn, Xc)


def test_scale_equals_fit_transform(sdata):
    Xn, Xc, ds = sdata
    a = mixdex.scale(ds)
    b = mixdex.Scaler.fit(ds).transform(ds)
    for i in range(0, 200, 13):
        for j in range(0, 200, 17):
            assert mixdex.distance(a, i, j) == mixdex.distance(b, i, j)


def test_params_match_scale_feature(sdata):
    Xn, _, ds = sdata
    sc = mixdex.Scaler.fit(ds)
    assert sc.n_num == 2
    for k in range(2):
        col = np.ascontiguousarray(Xn[:, k])
        scaled = np.array(mixdex.scale_feature(col, mixdex.NumericType.Continuous))
        assert np.allclose((col - sc.medians[k]) / sc.scales[k], scaled)


def _ladder_data():
    """Nonconstant zero-IQR features must fall back to IDR or full range.
    
    Dataset rejects true constants before scaling; test that policy separately."""
    rng = np.random.default_rng(5)
    n = 200
    x1 = np.zeros(n)
    x1[:10] = rng.normal(size=10) + 5.0
    x2 = np.zeros(n)
    x2[:40] = 1.0
    Xn = np.column_stack([rng.normal(0, 1, n), x1, x2])
    Xc = rng.integers(0, 3, (n, 1)).astype(np.int32)
    return Xn, make(Xn, Xc)


def test_scale_ladder_levels_and_scales():
    Xn, ds = _ladder_data()
    sc = mixdex.Scaler.fit(ds)
    L = mixdex.ScaleLevel
    assert list(sc.levels) == [L.Iqr, L.Range, L.Idr]
    assert sc.scales[1] == pytest.approx(Xn[:, 1].max() - Xn[:, 1].min())   # max - min
    assert sc.scales[2] == pytest.approx(1.0)                               # q90 - q10 = 1 - 0
    out = sc.transform_rows(Xn)
    assert out[:, 1].max() - out[:, 1].min() == pytest.approx(1.0)          # The selected interval has unit width after scaling.


def test_scale_ladder_preserves_independent_columns():
    Xn, ds = _ladder_data()
    sc = mixdex.Scaler.fit(ds)
    q1, q3 = np.percentile(Xn[:, 0], [25, 75], method="median_unbiased")
    assert sc.scales[0] == pytest.approx(q3 - q1)


def test_scale_threshold_is_relative_to_feature_units():
    rng = np.random.default_rng(8)
    n = 200
    tiny = 1e-14 * rng.normal(size=n)                   # Small units with genuine variation.
    stamp = 1.7e9 + rng.integers(0, 5, n).astype(float)  # Timestamp-like values with relative spread 1e-9.
    ds = make(np.column_stack([tiny, stamp]), np.zeros((n, 0), dtype=np.int32))
    sc = mixdex.Scaler.fit(ds)
    assert sc.levels[0] == mixdex.ScaleLevel.Iqr and 1e-15 < sc.scales[0] < 1e-13
    assert sc.scales[1] >= 1.0


def test_dataset_rejects_constant_feature_strict():
    """Dataset rejects constant columns before scaling."""
    rng = np.random.default_rng(5)
    n = 200
    Xn = np.column_stack([rng.normal(0, 1, n), np.full(n, 3.0)])
    Xc = rng.integers(0, 3, (n, 1)).astype(np.int32)
    with pytest.raises(mixdex.InvalidArgument, match="константн"):
        make(Xn, Xc)


def test_fit_scale_params_rejects_constant_column():
    """Scaling must explicitly reject a constant that bypasses Dataset validation."""
    col = np.full(50, 3.0)
    with pytest.raises(mixdex.InvalidArgument, match="констант"):
        mixdex.scale_feature(col, mixdex.NumericType.Continuous)


def test_scale_distinguishes_numerical_degeneracy_from_constant_column():
    # Distinct doubles can have a full range below the relative tolerance.
    col = np.array([1.0, np.nextafter(1.0, 2.0)])
    assert col[0] != col[1]
    ds = make(col.reshape(-1, 1), np.zeros((2, 0), dtype=np.int32))
    with pytest.raises(mixdex.InvalidArgument, match="численно вырожденный масштаб"):
        mixdex.Scaler.fit(ds)
    with pytest.raises(mixdex.InvalidArgument, match="численно вырожденный масштаб"):
        mixdex.scale_feature(col, mixdex.NumericType.Continuous)


def test_scale_feature_accepts_noncontiguous_arrays(sdata):
    Xn, _, _ = sdata
    strided = Xn[:, 1]
    a = np.array(mixdex.scale_feature(strided, mixdex.NumericType.Continuous))
    b = np.array(mixdex.scale_feature(np.ascontiguousarray(strided), mixdex.NumericType.Continuous))
    assert np.allclose(a, b)
    q1 = mixdex.quantiles(strided, [0.25, 0.5], mixdex.QuantileMethod.MedianUnbiased)
    q2 = mixdex.quantiles(np.ascontiguousarray(strided), [0.25, 0.5], mixdex.QuantileMethod.MedianUnbiased)
    assert np.allclose(q1, q2)


def test_transform_rows_matches_transform_of_dataset(sdata):
    Xn, Xc, ds = sdata
    sc = mixdex.Scaler.fit(ds)
    rows = sc.transform_rows(Xn)
    assert rows.shape == Xn.shape
    scaled_ds = sc.transform(ds)
    # External rows must use the same transform as Dataset rows.
    ds_rows = make(rows, Xc)
    for i, j in ((0, 1), (5, 150), (77, 3)):
        assert np.isclose(mixdex.distance(ds_rows, i, j), mixdex.distance(scaled_ds, i, j))


def test_new_rows_land_in_the_index_scale(sdata):
    Xn, Xc, ds = sdata
    sc = mixdex.Scaler.fit(ds)
    scaled_ds = sc.transform(ds)
    # An identical training row transforms identically.
    q = sc.transform_rows(Xn[[10]])
    nb = mixdex.knn_query(scaled_ds, q, Xc[[10]], 1, BT)[0]
    assert nb[0] == (10, 0.0)


def test_scaler_errors(sdata):
    Xn, Xc, ds = sdata
    sc = mixdex.Scaler.fit(ds)
    with pytest.raises(mixdex.InvalidArgument):
        sc.transform_rows(Xn[:, :1])                       # Wrong column count.
    with pytest.raises(mixdex.InvalidArgument):
        sc.transform_rows(np.where(Xn > 100, Xn, np.nan))  # NaN
    with pytest.raises(mixdex.InvalidArgument):
        sc.transform(make(Xn[:, :1], Xc))                  # Different schema.
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.Scaler.fit(mixdex.Dataset())                # No rows.


# Core-distance definition

@pytest.fixture(scope="module")
def cdata():
    rng = np.random.default_rng(2)
    Xn = np.vstack([rng.normal(c, 0.7, (150, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    Xc = rng.integers(0, 3, (450, 2)).astype(np.int32)
    return Xn, Xc, make(Xn, Xc)


@pytest.mark.parametrize("m_pts", [2, 3, 10, 30])
def test_core_matches_numpy(cdata, m_pts):
    Xn, Xc, ds = cdata
    D = distance_np(Xn, Xc, Xn, Xc)
    np.fill_diagonal(D, np.inf)
    want = np.sort(D, axis=1)[:, m_pts - 2]
    h = mixdex.DensityHierarchy(ds, m_pts, EXACT, BT)
    assert np.allclose(h.core_distances, want)


def test_m_pts_one_gives_zero_core(cdata):
    _, _, ds = cdata
    assert (np.array(mixdex.DensityHierarchy(ds, 1, EXACT, BT).core_distances) == 0.0).all()


def test_m_pts_two_is_nearest_neighbour(cdata):
    Xn, Xc, ds = cdata
    D = distance_np(Xn, Xc, Xn, Xc)
    np.fill_diagonal(D, np.inf)
    h = mixdex.DensityHierarchy(ds, 2, EXACT, BT)
    assert np.allclose(h.core_distances, D.min(axis=1))
    assert not np.allclose(h.core_distances, 0.0)  # m_pts=1 and m_pts=2 must differ.


@pytest.mark.parametrize("bad", [0, -3])
def test_m_pts_below_one_rejected(cdata, bad):
    _, _, ds = cdata
    with pytest.raises(mixdex.InvalidArgument):
        mixdex.DensityHierarchy(ds, bad, EXACT, BT)


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
def test_dbscan_star_m_pts_one_is_single_linkage_cut(cdata, eps):
    Xn, Xc, ds = cdata
    D = distance_np(Xn, Xc, Xn, Xc)
    ref = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(D).labels_  # All points are core points.
    lab = np.array(mixdex.DensityHierarchy(ds, 1, EXACT, BT).dbscan_star(eps))
    assert (lab >= 0).all()
    assert adjusted_rand_score(ref, lab) == 1.0


@pytest.mark.parametrize("mode,k_opt", [(EXACT, -1), (APPROX, 21)])
def test_hdbscan_with_m_pts_one_is_sane(cdata, mode, k_opt):
    _, _, ds = cdata
    r = mixdex.hdbscan(ds, 1, 15, mode, BT, 40, k_opt)
    assert len(r.labels) == ds.n_rows()
    assert np.isfinite(np.array(r.outlier_scores)).all()
    assert max(r.labels) >= 1  # Find multiple clusters.


# Streaming kth-distance checks

@pytest.mark.parametrize("method", [BT, BF])
def test_streaming_kth_distance_matches_numpy(cdata, method):
    Xn, Xc, ds = cdata
    D = distance_np(Xn, Xc, Xn, Xc)
    np.fill_diagonal(D, np.inf)
    for k in (1, 7, 25):
        assert np.allclose(mixdex.knn_outlier_scores(ds, k, method), np.sort(D, axis=1)[:, k - 1])
