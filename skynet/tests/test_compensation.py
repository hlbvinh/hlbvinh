import copy
import math
from itertools import cycle, islice

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from ..sample.selection import COMPENSATE_COLUMN
from ..utils.compensation import (
    HUM_COMP,
    TEMP_COMP,
    compensate_features,
    compensate_humidity,
    compensate_sensors,
    compensate_temperature,
    ensure_compensated,
)

np.random.seed(1)


@pytest.fixture
def n_rows():
    return 5


def _humidity(n):
    return np.random.rand(n) * 50 + 50


def _temperature(n):
    return np.random.randn(n) * 10 + 20


def _assert_values_equal(left, right, check_dtype=True, check_column_order=False):
    if not check_column_order:
        right = right[left.columns]
    assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        check_dtype=check_dtype,
    )


@pytest.fixture
def base_features(n_rows):
    features = pd.DataFrame(np.random.rand(n_rows), columns=["other"])
    features["temperature"] = _temperature(n_rows)
    features["humidity"] = _humidity(n_rows)
    return features


@pytest.fixture
def features_compensated(base_features):
    base_features[COMPENSATE_COLUMN] = True
    return base_features


@pytest.fixture
def features_uncompensated(base_features):
    return base_features


@pytest.fixture
def features_partially_compensated(base_features, n_rows):
    base_features[COMPENSATE_COLUMN] = list(islice(cycle([0, 1]), n_rows))
    return base_features


@pytest.fixture(params=["uncompensated", "compensated", "partially_compensated"])
def features_df(
    request,
    features_uncompensated,
    features_compensated,
    features_partially_compensated,
):
    if request.param == "uncompensated":
        return features_uncompensated
    if request.param == "compensated":
        return features_compensated
    if request.param == "partially_compensated":
        return features_partially_compensated
    raise ValueError(f"fixture {request.param} doesn't exist")


@pytest.fixture
def features_dicts(features_df):
    return features_df.to_dict("records")


@pytest.fixture
def sensors_base(n_rows):
    return pd.DataFrame(np.random.rand(n_rows), columns=["other"])


@pytest.fixture
def sensors_refined(sensors_base, n_rows):
    sensors_base["temperature_raw"] = _temperature(n_rows)
    sensors_base["temperature_refined"] = _temperature(n_rows)
    sensors_base["humidity_raw"] = _humidity(n_rows)
    sensors_base["humidity_refined"] = _temperature(n_rows)
    return sensors_base


@pytest.fixture
def sensors_legacy(sensors_base, n_rows):
    sensors_base["temperature"] = _temperature(n_rows)
    sensors_base["humidity"] = _humidity(n_rows)
    return sensors_base


@pytest.fixture(params=["refined", "legacy"])
def sensors(request, sensors_refined, sensors_legacy):
    if request.param == "refined":
        return sensors_refined
    return sensors_legacy


def assert_dict_equal(left, right):
    assert len(left) == len(right)
    for k, v in left.items():
        assert_equal(v, right[k])


def test_compensation_sensor(sensors):
    N = len(sensors)
    orig = sensors.copy()
    comp = compensate_sensors(sensors)
    if "humidity" in comp:
        np.testing.assert_array_less(comp["humidity"], np.ones(N) * 100.0001)
    pd.testing.assert_frame_equal(sensors, orig)

    d = {"temperature": 1.0, "humidity": 2.0}
    desired = {"temperature": 1.0 + TEMP_COMP, "humidity": 2.0 + HUM_COMP}
    compensated = compensate_sensors(d)
    assert compensated["humidity"] == desired["humidity"]
    assert compensated["temperature"] == desired["temperature"]
    assert compensated[COMPENSATE_COLUMN] is True

    d["humidity"] = 100.0
    h_comp = compensate_sensors(d)["humidity"]
    assert h_comp <= 100.0


def test_compensate_features_dict(features_df, features_dicts):
    orig = copy.deepcopy(features_dicts)
    from_dicts = pd.DataFrame(compensate_features(features_dicts))
    assert features_dicts == orig, "compensate_features modified dicts input"
    from_df = compensate_features(features_df)
    _assert_values_equal(from_dicts, from_df, check_dtype=False)


def test_compensate_features_input_unchanged(features_df):
    origial = features_df.copy()
    compensate_features(features_df)
    assert_frame_equal(origial, features_df)


@pytest.fixture
def features_df_compensated(features_df):
    return compensate_features(features_df)


def test_compensate_features_idempotent(features_df_compensated):
    _assert_values_equal(
        compensate_features(features_df_compensated), features_df_compensated
    )


def test_compensate_features_dicts_partial(features_dicts):
    original = features_dicts
    compensated = compensate_features(original)
    for new, orig in zip(compensated, original):
        if orig[COMPENSATE_COLUMN]:
            assert_dict_equal(new, orig)
        else:
            if "temperature" in orig:
                assert new["temperature"] == compensate_temperature(orig["temperature"])
            if "humidity" in orig:
                assert new["humidity"] == compensate_humidity(orig["humidity"])


def test_partially_compensated_df_vs_dicts_input(features_df, features_dicts):
    from_dicts = pd.DataFrame.from_records(compensate_features(features_dicts))
    # need to fix column order
    from_df = compensate_features(features_df)[from_dicts.columns]

    # XXX somehow from_df has object type for "other" column (?!)
    assert_frame_equal(
        from_df.reset_index(drop=True),
        from_dicts.reset_index(drop=True),
        check_dtype=False,
    )


@ensure_compensated
def _apply_ensure_compensated(_self, X):
    return X


def test_ensure_compensated_empty():
    # unable to ensure compensation due to mussing compensate column
    with pytest.raises(ValueError):
        _apply_ensure_compensated(None, pd.DataFrame())


def test_ensure_compensated_partial():
    # partially compensated dataframe should be compensated
    X = pd.DataFrame([[1, 0], [0, 0]], columns=[COMPENSATE_COLUMN, "temperature"])
    Y = _apply_ensure_compensated(None, X)
    assert Y.loc[0, "temperature"] == 0
    assert Y.loc[1, "temperature"] == compensate_temperature(0)


def test_ensure_compensated_already_compensated():
    # Already compensated dataframe should pass unchanged apart from having
    # the compensate column removed.
    X = pd.DataFrame([[1, 2], [1, 2]], columns=[COMPENSATE_COLUMN, "other"])
    Y = _apply_ensure_compensated(None, X)
    assert_frame_equal(X.drop(COMPENSATE_COLUMN, axis=1), Y)


REFINED_VALUES = [np.nan, None, 10.0, "omit"]


@pytest.fixture
def sensor_row():
    return {"temperature": 1.0, "humidity": 2.0}


@pytest.fixture(params=REFINED_VALUES)
def temperature_refined(request):
    return request.param


@pytest.fixture(params=REFINED_VALUES)
def humidity_refined(request):
    return request.param


@pytest.fixture
def sensor_row_with_nans(temperature_refined, humidity_refined, sensor_row):
    row = {
        "temperature_refined": temperature_refined,
        "humidity_refined": humidity_refined,
        **sensor_row,
    }
    return {key: value for key, value in row.items() if value != "omit"}


def test_compensate_sensors_regression_nan_progation(sensor_row_with_nans):
    """Ensure NaN values for refined columns don't overwrite legacy values."""
    compensated = compensate_sensors(sensor_row_with_nans)
    assert math.isfinite(compensated["temperature"])
    assert math.isfinite(compensated["humidity"])


@pytest.fixture
def sensor_df_with_nans(sensor_row_with_nans):
    return pd.DataFrame([sensor_row_with_nans])


def test_compensate_sensors_regression_nan_propagation_dataframe(
    sensor_df_with_nans, sensor_row_with_nans
):
    """Compensation via DataFrame or dicts must give the same results."""
    from_df = compensate_sensors(sensor_df_with_nans).iloc[0]
    from_dict = pd.Series(compensate_sensors(sensor_row_with_nans))
    # using assert_series_equal so that NaN and NaN compare equal
    assert_series_equal(from_df.sort_index(), from_dict.sort_index(), check_names=False)
