"""Check recoverable overflow, finite scaling and search consistency."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

import mixdex as mx


LARGEST = np.finfo(float).max


def dataset(numeric, categorical=None):
    num = np.asarray(numeric, dtype=float)
    cat = (np.empty((len(num), 0), dtype=np.int32) if categorical is None
           else np.asarray(categorical, dtype=np.int32))
    p, c = num.shape[1], cat.shape[1]
    return mx.Dataset(num.ravel().tolist(), cat.ravel().tolist(), len(num), p, c,
                      [mx.FeatureType.Numeric] * p + [mx.FeatureType.Categorical] * c,
                      [mx.NumericType.Continuous] * p,
                      [f"n{k}" for k in range(p)] + [f"c{k}" for k in range(c)])


def test_unrepresentable_scale_is_rejected():
    values = [-LARGEST, LARGEST]
    with pytest.raises(RuntimeError, match="numeric scale exceeds"):
        mx.scale_feature(values, mx.NumericType.Continuous)
    with pytest.raises(RuntimeError, match="numeric scale exceeds"):
        mx.Scaler.fit(dataset([[values[0]], [values[1]]]))


def test_overflowing_centering_recovers_a_finite_query():
    scaler = mx.Scaler.fit(dataset([[-1e308], [-9.9e307]]))
    with localcontext() as ctx:
        ctx.prec = 100
        centered = Decimal.from_float(1e308) - Decimal.from_float(scaler.medians[0])
        expected = float(centered / Decimal.from_float(scaler.scales[0]))
    result = scaler.transform_rows(np.array([[1e308]]))[0, 0]
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=2e-15)


def test_unrepresentable_scaled_query_is_rejected():
    scaler = mx.Scaler.fit(dataset([[0.], [1e-20]]))
    with pytest.raises(RuntimeError, match="scaled value exceeds"):
        scaler.transform_rows(np.array([[LARGEST]]))


def test_empty_numeric_and_categorical_schema_survives_scaling():
    empty = mx.Dataset([], [], 0, 1, 1,
                       [mx.FeatureType.Numeric, mx.FeatureType.Categorical],
                       [mx.NumericType.Discrete], ["number", "label"])
    empty.set_weights([2, 3])
    scaler = mx.Scaler.fit(dataset([[0.], [1.]]))
    for result in [mx.scale(empty), scaler.transform(empty)]:
        assert result.n_rows() == 0 and result.n_num() == 1 and result.n_cat() == 1
        assert result.feat_name(0) == "number" and result.feat_name(1) == "label"
        assert result.num_type(0) == mx.NumericType.Discrete
        assert result.weight(0) == 2 and result.weight(1) == 3


@pytest.mark.parametrize("weight", [1., LARGEST / 2, math.ldexp(1., 1020)])
def test_common_weight_rescaling_preserves_one_feature_distance(weight):
    data = dataset([[0.], [3.]])
    data.set_weights([weight])
    assert mx.distance(data, 0, 1) == pytest.approx(3.)
    assert mx.distance(data, 0, 0) == 0.


def test_unweighted_numerator_overflow_recovers_finite_average():
    data = dataset([[0., 0.], [1e308, 1e308]])
    assert mx.distance(data, 0, 1) == pytest.approx(1e308)


def test_difference_overflow_recovers_finite_weighted_distance():
    data = dataset([[-LARGEST], [LARGEST]], [[0], [1]])
    data.set_weights([1, 3])
    assert mx.distance(data, 0, 1) == pytest.approx(LARGEST / 2)


@pytest.mark.parametrize("leaf", [1, 4, 40])
def test_extreme_weights_keep_tree_queries_and_bounds_consistent(leaf):
    data = dataset(np.arange(24, dtype=float)[:, None])
    data.set_weights([math.ldexp(1., 1020)])
    tree = mx.Index(data, mx.SearchMethod.BallTree, leaf)
    brute = mx.Index(data, mx.SearchMethod.BruteForce, leaf)
    for q in [0, 9, 23]:
        assert mx.knn(tree, q, 6) == mx.knn(brute, q, 6)
    queries = np.array([[-3.], [7.25], [33.]])
    assert mx.knn_query(tree, queries, None, 6) == mx.knn_query(brute, queries, None, 6)
    core = mx.core_distances(tree, 2)
    assert mx.check_lb_violations(data, core, leaf) == []
    full = mx.boruvka_edges(mx.full_mreach_edges(data, core), data.n_rows())
    result = mx.dt_boruvka(data, core, leaf)
    assert len(result) == 23
    assert sorted(edge.w for edge in result) == sorted(edge.w for edge in full)
