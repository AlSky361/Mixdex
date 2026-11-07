"""Fit once, then fill raw numeric, scalar and set-valued categorical columns."""
from collections import Counter
import heapq
import math
from numbers import Real

import numpy as np
import mixdex
from mixdex_encoding import category_value


def numeric_value(value):
    if value is None:
        return None
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError("numeric features must be numbers or None")
    return float(value) if math.isfinite(value) else None


def resolve_numeric_types(values, count):
    result = [mixdex.NumericType.Continuous] * count if values is None else list(values)
    if len(result) != count:
        raise ValueError("one numeric type per numeric column is required")
    known = {"Continuous": mixdex.NumericType.Continuous, "Discrete": mixdex.NumericType.Discrete}
    for j, value in enumerate(result):
        if isinstance(value, str) and value in known:
            result[j] = known[value]
        elif not isinstance(value, mixdex.NumericType) or value not in known.values():
            raise ValueError("numeric types must be Continuous or Discrete")
    return tuple(result)


def set_columns(values, count):
    result = tuple(values)
    if any(type(j) is not int or not 0 <= j < count for j in result) or len(set(result)) != len(result):
        raise ValueError("set columns must be distinct categorical column indices")
    return tuple(sorted(result))


def feature_weights(values, count):
    result = [1.0] * count if values is None else list(values)
    if len(result) != count or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                                   or not math.isfinite(v) or v < 0 for v in result):
        raise ValueError("weights must be finite and nonnegative")
    total = sum(result)
    if not math.isfinite(total) or total <= 0 or not math.isfinite(1.0 / total):
        raise ValueError("weights must have a finite positive sum and finite reciprocal")
    return tuple(float(v) for v in result)


def _matrices(numeric, categorical, nn=None, nc=None):
    num = None if numeric is None else [list(row) for row in numeric]
    cat = None if categorical is None else [list(row) for row in categorical]
    n = len(num) if num is not None else len(cat) if cat is not None else 0
    num = [[] for _ in range(n)] if num is None else num
    cat = [[] for _ in range(n)] if cat is None else cat
    if nn is None:
        nn = len(num[0]) if num else 0
    if nc is None:
        nc = len(cat[0]) if cat else 0
    if len(num) != len(cat) or any(len(row) != nn for row in num) or any(len(row) != nc for row in cat):
        raise ValueError("feature matrices must match the fitted schema")
    return num, cat, nn, nc


def _median(values, type_):
    method = (mixdex.QuantileMethod.InvertedCDF if type_ == mixdex.NumericType.Discrete
              else mixdex.QuantileMethod.MedianUnbiased)
    return mixdex.quantiles_select(values, [.5], method)[0]


class Imputer:
    """Immutable fitted fills; zero-weight columns are ignored, not guessed.

    Fit on original objects before deduplication. Fully missing active columns
    raise; known empty sets count as observations. Transform never refits.
    """
    def __init__(self):
        raise TypeError("use Imputer.fit")

    @classmethod
    def fit(cls, numeric=None, categorical=None, *, numeric_types=None,
            categorical_set_columns=(), weights=None):
        num, cat, nn, nc = _matrices(numeric, categorical)
        if nn + nc == 0:
            raise ValueError("at least one feature and one fitting row are required")
        types = resolve_numeric_types(numeric_types, nn)
        columns = set_columns(categorical_set_columns, nc)
        w = feature_weights(weights, nn + nc)
        num = [[numeric_value(v) for v in row] for row in num]
        cat = [[category_value(v, j in columns) for j, v in enumerate(row)] for row in cat]
        fills_n, fills_c, observed_n, observed_c = [], [], [], []
        for j, type_ in enumerate(types):
            values = [row[j] for row in num if row[j] is not None]
            observed_n.append(len(values))
            if w[j] and not values:
                raise ValueError(f"numeric column {j} has no observed values")
            fills_n.append(_median(values, type_) if w[j] else None)
        for j in range(nc):
            values = [row[j] for row in cat if row[j] is not None]
            observed_c.append(len(values))
            if not w[nn+j]:
                fills_c.append(None)
            elif not values:
                raise ValueError(f"categorical column {j} has no observed values")
            elif j in columns:
                k = int(_median([len(v) for v in values], mixdex.NumericType.Discrete))
                counts = Counter(label for value in values for label in value)
                top = heapq.nsmallest(k, counts, key=lambda label: (-counts[label], label))
                fills_c.append(tuple(sorted(top)))
            else:
                counts = Counter(values)
                fills_c.append(min(counts, key=lambda label: (-counts[label], label)))
        obj = object.__new__(cls)
        obj._types, obj._set_columns, obj._weights = types, frozenset(columns), w
        obj._numeric, obj._categorical = tuple(fills_n), tuple(fills_c)
        obj._observed_n, obj._observed_c = tuple(observed_n), tuple(observed_c)
        return obj

    def transform(self, numeric=None, categorical=None):
        """Return detached row matrices in the original schema, without encoding."""
        num, cat, _, _ = _matrices(numeric, categorical, len(self._numeric), len(self._categorical))
        for row in num:
            for j, value in enumerate(row):
                value = numeric_value(value)
                row[j] = self._numeric[j] if value is None else value
        for row in cat:
            for j, value in enumerate(row):
                value = category_value(value, j in self._set_columns)
                row[j] = self._categorical[j] if value is None else value
        return num, cat

    def transform_row(self, numeric=(), categorical=()):
        num, cat = self.transform([numeric], [categorical])
        return num[0], cat[0]

    def parameters(self):
        """Return JSON-compatible values; editing them cannot change this object."""
        return dict(numeric_types=[t.name for t in self._types],
                    categorical_set_columns=sorted(self._set_columns),
                    numeric_fill=list(self._numeric),
                    categorical_fill=[list(v) if isinstance(v, tuple) else v for v in self._categorical],
                    numeric_observed=list(self._observed_n), categorical_observed=list(self._observed_c))
