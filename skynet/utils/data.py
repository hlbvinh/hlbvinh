from collections import Container, Mapping, Sequence
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def is_valid_feature_value(x):
    if x is None:
        return False
    return isinstance(x, (list, str)) or bool(np.isfinite(x))


def has_feature(d, key):
    return is_valid_feature_value(d.get(key))


def _to_native(x):
    """Normalize numpy data types to builtin data types."""
    type_ = type(x)
    if isinstance(x, str):
        return x
    if type_.__module__ == np.__name__:
        return x.item()
    if isinstance(x, Mapping):
        return type_({_to_native(k): _to_native(v) for k, v in x.items()})
    if isinstance(x, Sequence):
        return type_([_to_native(i) for i in x])
    return x


def group_by(key, records, keep_key=False):
    multikey = not isinstance(key, str) and isinstance(key, Container)
    records = [r.copy() for r in records]
    ret = {}
    get = dict.get if keep_key else dict.pop
    for r in records:
        if multikey:
            record_key = tuple(get(r, k) for k in key)
        else:
            record_key = get(r, key)
        if record_key not in ret:
            ret[record_key] = []
        ret[record_key].append(r)
    return ret


def intervals(timestamps, start, end):

    type_ = type(start - end)

    if end < start:
        raise ValueError("start {} later than end {}".format(start, end))

    durations = [
        min(t1, end) - max(t0, start) for t0, t1 in zip(timestamps[:-1], timestamps[1:])
    ]
    last_start = max(start, timestamps[-1])
    durations.append(max(end - last_start, type_(0)))

    return durations


def interp_ts_df(df, new_index):
    ret = np.zeros([len(new_index), len(df.columns)])
    for i, col in enumerate(df):
        ret[:, i] = np.interp(new_index.asi8, df.index.asi8, df[col].values)
    return pd.DataFrame(ret, columns=df.columns, index=new_index)


def argmax_dict(
    d: Dict[str, float], priority: Optional[List[str]] = None
) -> Optional[str]:
    priority = priority or sorted(d)
    return max([p for p in priority if p in d], key=d.get, default=None)  # type: ignore


def find_nearest(listlike, val, transform_function=None) -> Union[int, float]:
    if not isinstance(listlike, np.ndarray):
        listlike = np.array(listlike)
    if transform_function is None:
        idx = np.abs(listlike - val).argmin()
    else:
        idx = np.abs(transform_function(listlike) - val).argmin()
    return listlike.flat[idx].item()


def is_int(n):
    try:
        int(n)
        return True

    except (ValueError, TypeError):
        return False


def is_float(n):
    if is_int(n):
        return False

    try:
        float(n)
        return True

    except (ValueError, TypeError):
        return False


def drop_from_sequence(sequence, items):
    drop = set(items)
    return type(sequence)(item for item in sequence if item not in drop)
