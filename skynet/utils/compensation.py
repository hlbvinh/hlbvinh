import functools
import math
from typing import Sequence, TypeVar

import numpy as np
import pandas as pd

from ..sample.selection import COMPENSATE_COLUMN, COMPENSATION_MAP
from . import log_util
from .types import Record, RecordSequence, SensorReadings

log = log_util.get_logger(__name__)

TEMP_COMP = -4.0
HUM_COMP = 16.0

COMPENSATABLE_QUANTITIES = ["humidity", "temperature"]
Compensatable = TypeVar("Compensatable", pd.DataFrame, Record, RecordSequence)


@functools.singledispatch
def compensate_sensors(arg: Compensatable) -> Compensatable:
    raise NotImplementedError(f"Not implemented for {type(arg)})")


@compensate_sensors.register(pd.DataFrame)
def _compensate_sensors_df(df: pd.DataFrame) -> Compensatable:

    if COMPENSATE_COLUMN in df and any(df[COMPENSATE_COLUMN] == 1):
        raise ValueError("Found compensated rows in DataFrame passed.")

    df = df.copy()

    for quantity in COMPENSATABLE_QUANTITIES:
        df[quantity] = _compute_compensated(quantity, df)

    df[COMPENSATE_COLUMN] = True

    return df


def _compute_compensated(quantity: str, df: pd.DataFrame) -> pd.DataFrame:

    refined_column = f"{quantity}_refined"
    legacy_compensated = _do_compensate(quantity, df[quantity])

    if refined_column in df:
        compensated = df[refined_column].fillna(legacy_compensated)
    else:
        compensated = legacy_compensated

    return compensated


@compensate_sensors.register(dict)
def _compensate_sensors_dict(d: Record):

    if d.get(COMPENSATE_COLUMN, False):
        raise ValueError("Record already compensated.")

    d = d.copy()

    for quantity in COMPENSATABLE_QUANTITIES:

        refined_column = f"{quantity}_refined"
        refined_value = d.get(refined_column)

        if refined_value is not None and math.isfinite(refined_value):
            value = refined_value
        else:
            value = _do_compensate(quantity, d[quantity])

        d[quantity] = value

    d[COMPENSATE_COLUMN] = True

    return d


@compensate_sensors.register(Sequence)
def _compensate_sensors_dicts(sequence):
    return type(sequence)(_compensate_sensors_dict(d) for d in sequence)


@functools.singledispatch
def compensate_features(arg: Compensatable) -> Compensatable:
    raise NotImplementedError(f"Not implemented for {type(arg)})")


@compensate_features.register(dict)
def _compensate_features_dict(features: Record):

    # checking complicated because np.nan is truthy
    # (missing) -> compensate
    # nan       -> compensate
    # False     -> compensate
    # True      -> don't compensate
    # 1         -> don't compensate (True == 1, no need to check seperately)
    need_comp = {True: False, np.nan: True, False: True, None: True}[
        features.get(COMPENSATE_COLUMN)
    ]

    if need_comp:
        compensated = features.copy()
        for quantity, features_to_compensate in COMPENSATION_MAP.items():
            for feature in features_to_compensate:
                if feature in features:
                    compensated[feature] = _do_compensate(quantity, features[feature])
        compensated[COMPENSATE_COLUMN] = True
        return compensated
    return features


@compensate_features.register(Sequence)
def _compensate_features_dicts(sequence):
    return type(sequence)(_compensate_features_dict(d) for d in sequence)


@compensate_features.register(pd.DataFrame)
def _compensate_features_df(df: pd.DataFrame) -> Compensatable:

    to_compensate = {
        quantity: list(set(COMPENSATION_MAP[quantity]) & set(df.columns))
        for quantity in COMPENSATABLE_QUANTITIES
    }

    original_index = df.index
    df = df.reset_index(drop=True)
    if COMPENSATE_COLUMN not in df:
        df[COMPENSATE_COLUMN] = False

    # we consider only 2 cases
    # 0: the whole row is not compensated
    # 1: the whole row is compensated

    need_comp_idx = ~(df[COMPENSATE_COLUMN].fillna(0) == 1)
    need_comp = df.loc[need_comp_idx]

    for quantity, to_compensate_columns in to_compensate.items():
        for to_compensate_column in to_compensate_columns:
            df.loc[need_comp_idx, to_compensate_column] = _do_compensate(
                quantity, need_comp[to_compensate_column]
            )

    log.info(f"compensated {len(need_comp)} of {len(df)} rows")

    df[COMPENSATE_COLUMN] = True
    df.index = original_index
    return df


def compensate_temperature(t):
    return t + TEMP_COMP


def compensate_humidity(h):
    return np.clip(h + HUM_COMP, 0, 100.0)


def _do_compensate(family: str, value: SensorReadings) -> SensorReadings:
    if family == "temperature":
        return compensate_temperature(value)
    if family == "humidity":
        return compensate_humidity(value)
    raise ValueError(f"bad family {family}")


def ensure_compensated(fun):
    @functools.wraps(fun)
    def wrap(self, X, *args, **kwargs):
        if COMPENSATE_COLUMN not in X:
            raise ValueError(
                f"Missing '{COMPENSATE_COLUMN}' column, " "cannot ensure compensation."
            )

        log.info("ensuring compensation")
        return fun(
            self,
            compensate_features(X).drop(COMPENSATE_COLUMN, axis=1),
            *args,
            **kwargs,
        )

    return wrap
