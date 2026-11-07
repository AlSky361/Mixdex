"""Fitted categorical encoders backed by the C++ library."""
import mixdex

LabelEncoder = mixdex.LabelEncoder
SetEncoder = mixdex.SetEncoder
EncodedSetQuery = mixdex.EncodedSetQuery


def category_value(value, is_set):
    """Normalize a raw label or set; None remains distinct from a known empty set."""
    if value is None:
        return None
    if not is_set:
        if not isinstance(value, str):
            raise ValueError("categorical labels must be strings or None")
        return value
    if not isinstance(value, (list, tuple, set, frozenset)) or any(not isinstance(v, str) for v in value):
        raise ValueError("set features must contain string labels, or be None")
    return tuple(mixdex.canonicalize_category_set(value))
