"""Independent quantile regressions and public input contracts."""
from fractions import Fraction

import numpy as np
import pytest

import mixdex


METHODS = [mixdex.QuantileMethod.InvertedCDF, mixdex.QuantileMethod.MedianUnbiased]


def evaluate_both(values, ps, method):
    values = np.array(values, dtype=float)
    before = values.copy()
    sorted_values = np.sort(values)
    selected = mixdex.quantiles_select(values, ps, method)
    sorted_result = mixdex.quantiles(sorted_values, ps, method)
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(selected, sorted_result)
    return np.array(selected)


def test_discrete_jump_has_no_epsilon_plateau():
    ps = [1 / 3 - 1e-13, 1 / 3, 1 / 3 + 1e-13]
    np.testing.assert_array_equal(
        evaluate_both([20, 0, 10], ps, METHODS[0]), [0, 0, 10]
    )


def test_type8_matches_exact_rational_definition():
    values = [-12, -3, 5, 8, 29]
    ps = [Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2),
          Fraction(3, 4), Fraction(7, 8), Fraction(1)]
    expected = []
    for p in ps:
        h = (len(values) + Fraction(1, 3)) * p + Fraction(1, 3)
        if h <= 1:
            expected.append(values[0])
        elif h >= len(values):
            expected.append(values[-1])
        else:
            j = int(h)
            expected.append(float(values[j - 1] + (h - j) * (values[j] - values[j - 1])))
    np.testing.assert_allclose(
        evaluate_both(values, [float(p) for p in ps], METHODS[1]),
        expected, rtol=2e-15, atol=2e-15,
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [np.nextafter(0., 1.), -np.nextafter(0., 1.),
                                  np.finfo(float).max, -np.finfo(float).max])
def test_constant_extreme_values_are_preserved(method, value):
    result = evaluate_both([value] * 7, [0, .1, .25, .5, .75, .9, 1], method)
    np.testing.assert_array_equal(result, [value] * 7)


def test_finite_extreme_interpolation_and_endpoints():
    largest = np.finfo(float).max
    np.testing.assert_array_equal(
        evaluate_both([-largest, largest], [0, .5, 1], METHODS[1]),
        [-largest, 0., largest],
    )
    tiny = np.nextafter(0., 1.)
    result = evaluate_both([tiny, 2 * tiny], [0, .5, 1], METHODS[1])
    assert result[0] == tiny and result[-1] == 2 * tiny
    assert tiny <= result[1] <= 2 * tiny


@pytest.mark.parametrize("method", METHODS)
def test_matches_numpy_on_finite_ordinary_samples(method):
    name = "inverted_cdf" if method == METHODS[0] else "median_unbiased"
    rng = np.random.default_rng(140)
    ps = [1., 0., .25, .5, .75, .125, .875, .5]
    for n in [1, 2, 3, 7, 32, 101]:
        values = rng.normal(size=n)
        np.testing.assert_allclose(evaluate_both(values, ps, method),
                                   np.quantile(values, ps, method=name),
                                   rtol=5e-14, atol=2e-15)


@pytest.mark.parametrize("function", [mixdex.quantiles, mixdex.quantiles_select])
@pytest.mark.parametrize("ps", [[], [.5]])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_rejected(function, ps, bad):
    with pytest.raises(mixdex.InvalidArgument, match="finite"):
        function([0, bad, 2], ps, METHODS[1])


@pytest.mark.parametrize("ps", [[], [.5]])
def test_python_sorted_entry_rejects_unsorted_values(ps):
    with pytest.raises(mixdex.InvalidArgument, match="sorted"):
        mixdex.quantiles([2, 0, 1], ps, METHODS[1])


@pytest.mark.parametrize("function", [mixdex.quantiles, mixdex.quantiles_select])
@pytest.mark.parametrize("method", METHODS)
def test_empty_input_and_empty_probabilities(function, method):
    assert function([], [], method) == []
    assert function([0, 1], [], method) == []
    with pytest.raises(mixdex.InvalidArgument, match="nonempty"):
        function([], [.5], method)


@pytest.mark.parametrize("function", [mixdex.quantiles, mixdex.quantiles_select])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -.1, 1.1])
def test_invalid_probabilities_rejected(function, bad):
    values = np.array([0., 1., 2.])
    with pytest.raises(mixdex.InvalidArgument):
        function(values, [.5, bad], METHODS[1])
    np.testing.assert_array_equal(values, [0, 1, 2])


@pytest.mark.parametrize("kind", [mixdex.NumericType.Continuous, mixdex.NumericType.Discrete])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_standalone_scaling_rejects_nonfinite_values(kind, bad):
    with pytest.raises(mixdex.InvalidArgument, match="finite"):
        mixdex.scale_feature([0, bad, 2], kind)
