from collections.abc import Iterable

import numpy as np


def circular_variable(x, period):
    """Transform into circular variable by using sine/cosine."""
    c = 2.0 * np.pi / period
    if isinstance(x, Iterable):
        X = np.column_stack([np.sin(c * x), np.cos(c * x)])
    else:
        X = np.array([f(c * x) for f in [np.sin, np.cos]])
    return X


def time_of_day(x):
    """Cos/Sin transform of pd.Timestamp (or collection thereof)."""
    values = x.hour + x.minute / 60.0
    return circular_variable(values, 24)


def time_of_week(x):
    """Cos/Sin transform of pd.Timestamp (or collection thereof)."""
    values = x.dayofweek + x.hour / 24.0 + x.minute / (24.0 * 60.0)
    return circular_variable(values, 7)


def time_of_year(x):
    """Cos/Sin transform of pd.Timestamp (or collection thereof)."""
    values = (
        x.dayofyear
        + x.day / 365.0
        + x.hour / (24.0 * 365.0)
        + x.minute / (60 * 24.0 * 365.0)
    )
    return circular_variable(values, 365)
