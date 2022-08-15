from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

from ..utils import data
from ..utils.enums import Power
from ..utils.misc import check_na
from ..utils.types import ApplianceState

APPLIANCE_STATE_FEATURES = [
    "power",
    "mode",
    "fan",
    "origin",
    "temperature_set",
    "louver",
    "swing",
]
WEATHER_IVAL = timedelta(days=1)
COMPENSATION_MAP: defaultdict = defaultdict(list)
COMPENSATE_COLUMN = "compensated"
TARGET_INTERVAL_MINUTES = 5
TARGET_INTERVAL = f"{TARGET_INTERVAL_MINUTES}Min"
STATIC_INTERPOLATION = timedelta(minutes=60)
STATIC_INTERPOLATION_MINUTES = STATIC_INTERPOLATION.seconds // 60
INTERPOLATION_INDEX = STATIC_INTERPOLATION_MINUTES // TARGET_INTERVAL_MINUTES

WEATHER_DELTA_IVAL = timedelta(hours=3)
SENSORS_INTERVAL = timedelta(minutes=20)
TIMESERIES_INTERVAL = timedelta(minutes=40)
SLOPE_INTERVAL = timedelta(minutes=3)
TIMESERIES_FREQUENCY = timedelta(seconds=150)

FeatureValue = Union[Number, str, bool, None]
HumidexArgs = NamedTuple("HumidexArgs", [("temperature", str), ("humidity", str)])


def get_interval(x: pd.Series, interval: Optional[timedelta] = None) -> pd.Series:
    if interval is None:
        return x
    return x[x.index > x.index.max() - interval]


def maximum(x: pd.Series, interval: Optional[timedelta] = None) -> Number:
    return get_interval(x, interval).max()


def minimum(x: pd.Series, interval: Optional[timedelta] = None) -> Number:
    return get_interval(x, interval).min()


def last(x: pd.Series) -> FeatureValue:
    if x.empty:
        return None
    last_value = x.tail(1).squeeze()
    return last_value


def mean(x: pd.Series, interval: Optional[timedelta] = None) -> float:
    return get_interval(x, interval).mean()


def sensors_mean(x: pd.Series, interval: Optional[timedelta] = SENSORS_INTERVAL):
    subset = x.last(interval) if interval is not None else x
    return mean(subset)


def slope(x: pd.Series, interval: Optional[timedelta] = SLOPE_INTERVAL) -> float:
    """Calculate per second slope of values in DataFrame default to zero."""

    if x.empty:
        return np.nan

    last = x.last(interval) if interval is not None else x

    if last.empty:
        return np.nan

    secs = np.asarray([float(q.strftime("%s")) for q in last.index])
    secs -= secs[0]
    try:
        value = np.polyfit(secs, np.array(last), 1)[0]
    except ValueError:
        value = np.nan
    return value


def previous(
    x: pd.Series, interval: Optional[timedelta] = TIMESERIES_INTERVAL
) -> List[float]:
    subset = x.last(interval) if interval is not None else x
    return subset.resample(TIMESERIES_FREQUENCY).mean().interpolate().tolist()


def get_appliance_state_features(state: ApplianceState) -> Dict[str, Any]:
    return {k: state.get(k) for k in APPLIANCE_STATE_FEATURES}


class Feature:
    def __init__(
        self,
        group: str,
        column: str,
        selector: Union[Callable, Callable],
        default_value: Any = None,
        required: bool = True,
        compensation_family: Optional[str] = None,
    ) -> None:
        self.group = group
        self.column = column
        self.selector = selector
        self.default_value = default_value
        self.required = required
        self.compensation_family = compensation_family

    def __call__(self, x: Dict[str, pd.DataFrame]) -> FeatureValue:
        try:
            value = self.selector(x[self.group][self.column])

            if isinstance(value, float) and np.isnan(value):
                value = self.default_value

            return data._to_native(value)

        except KeyError:
            return None

    def __str__(self):
        return "Feature: {} {} {} {}".format(
            self.group, self.column, self.selector.__name__, self.required
        )

    def __repr__(self):
        return self.__str__()


RECOMPUTATIONS = {
    "humidex": HumidexArgs("temperature", "humidity"),
    # This is not 100% correct but should be close enough for our purposes.
    "humidex_mean": HumidexArgs("temperature_mean", "humidity_mean"),
}


SELECTORS: Dict[str, Feature] = dict(
    fan_hist=Feature("states", "fan", last, "auto"),
    mode_hist=Feature("states", "mode", last, "cool"),
    temperature_set_last=Feature("states", "temperature_set", last),
    power_hist=Feature("states", "power", last, Power.OFF),
    compensated=Feature("sensors", COMPENSATE_COLUMN, last),
    humidex=Feature("sensors", "humidex", last),
    humidex_delta=Feature("sensors", "humidex", slope),
    humidex_mean=Feature("sensors", "humidex", sensors_mean),
    humidity=Feature("sensors", "humidity", last, compensation_family="humidity"),
    humidity_delta=Feature("sensors", "humidity", slope),
    humidity_mean=Feature(
        "sensors", "humidity", sensors_mean, compensation_family="humidity"
    ),
    luminosity_mean=Feature("sensors", "luminosity", sensors_mean, np.nan, False),
    luminosity_slope=Feature(
        "sensors",
        "luminosity",
        partial(slope, interval=SENSORS_INTERVAL),
        np.nan,
        False,
    ),
    pircount_mean=Feature("sensors", "pircount", sensors_mean, np.nan, False),
    temperature=Feature(
        "sensors", "temperature", last, compensation_family="temperature"
    ),
    temperature_delta=Feature("sensors", "temperature", slope),
    temperature_mean=Feature(
        "sensors", "temperature", sensors_mean, compensation_family="temperature"
    ),
    previous_temperatures=Feature(
        "sensors", "temperature", previous, compensation_family="temperature"
    ),
    temperature_out=Feature("weather", "temperature_out", last, np.nan),
    temperature_out_delta=Feature(
        "weather", "temperature_out", partial(slope, interval=WEATHER_DELTA_IVAL)
    ),
    temperature_out_mean_day=Feature(
        "weather", "temperature_out", partial(mean, interval=WEATHER_IVAL), np.nan
    ),
    temperature_out_max_day=Feature(
        "weather", "temperature_out", partial(maximum, interval=WEATHER_IVAL)
    ),
    temperature_out_min_day=Feature(
        "weather", "temperature_out", partial(minimum, interval=WEATHER_IVAL)
    ),
    temperature_out_mean_week=Feature("weather", "temperature_out", mean),
    humidity_out=Feature("weather", "humidity_out", last, np.nan),
    humidity_out_mean_day=Feature(
        "weather", "humidity_out", partial(mean, interval=WEATHER_IVAL)
    ),
    humidity_out_max_day=Feature(
        "weather", "humidity_out", partial(maximum, interval=WEATHER_IVAL)
    ),
    humidity_out_min_day=Feature(
        "weather", "humidity_out", partial(minimum, interval=WEATHER_IVAL)
    ),
    humidity_out_mean_week=Feature("weather", "humidity_out", mean),
    cloud_cover=Feature("weather", "cloud_cover", last, np.nan, False),
    apparent_temperature=Feature(
        "weather", "apparent_temperature", last, np.nan, False
    ),
    humidex_out=Feature("weather", "humidex_out", last, np.nan),
)

for name, feature in SELECTORS.items():
    if feature.compensation_family:
        COMPENSATION_MAP[feature.compensation_family].append(name)


def select_features(
    methods: Dict[str, Feature], data: Dict[str, pd.DataFrame], prediction: bool = False
) -> Dict[str, FeatureValue]:
    """Select (or compute) and extract features of DataFrame."""
    features = {}
    for name, selector in methods.items():
        try:
            features[name] = selector(data)
        except ValueError:
            if prediction:
                features[name] = None
            else:
                raise

        if features[name] is None:
            features[name] = selector.default_value

    return features


def check_select_features(
    methods: Dict[str, Feature],
    feature_data: Dict[str, pd.DataFrame],
    log: Callable,
    is_prediction: bool = False,
) -> Dict[str, FeatureValue]:
    features = select_features(methods, feature_data, is_prediction)
    missing = []
    for k, v in features.items():
        if methods[k].required and not data.is_valid_feature_value(v):
            missing.append(k)
    if missing:
        msg = "nan features after feature selection: {}".format(" ".join(missing))
        if not is_prediction:
            raise ValueError(msg)
        # More relaxed at prediction time, models can handle NaN and want to
        # make predictions if some features are missing
        log(msg)
    return features


@check_na(ValueError, "dataframe contrain nan after target selection")
def select_targets(
    target_data: pd.DataFrame, target_start: datetime, interval: str = TARGET_INTERVAL
) -> pd.DataFrame:
    """Resample, keeping original index base (in minutes)."""
    index = pd.date_range(
        start=target_start, end=target_data.index.max(), freq=interval
    )
    ret = data.interp_ts_df(target_data, index)
    ret.index.names = ["timestamp"]
    return ret


def _compute_static_target(ys: Union[np.ndarray, pd.Series]) -> float:
    assert len(ys) <= INTERPOLATION_INDEX
    m, b = np.polyfit(np.arange(ys.size), ys, deg=1)
    return m * STATIC_INTERPOLATION_MINUTES / TARGET_INTERVAL_MINUTES + b


def extrapolate_target(target: pd.DataFrame) -> pd.Series:
    return target.agg(_compute_static_target, axis=0)
