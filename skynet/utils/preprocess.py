import numpy as np
import pandas as pd

from . import config
from .enums import Power
from .types import ApplianceState

# these map the signals to numerical values with a case insensitive dict
# FAN_DICT = Cid({'auto':0, '1':1, '2':2, '3':3, '4':4, '5':5,
#     'low':1, 'med-low':2, 'med':3, 'med-high':4, 'high':5})
# MODE_DICT = Cid({'fan':0, 'dry':1, 'auto':2, 'cool':3, 'heat':4})
# FANS = FAN_DICT.keys()
# MODES = MODE_DICT.keys()


def prediction_signal(signal: ApplianceState) -> ApplianceState:
    """Map signal to state suitable for predictions

    Parameters
    ----------
    signal : dict
        appliance state

    Returns
    -------
    dict
        appliance state suitable for predictions
    """
    s = signal.copy()

    if "temperature" in s:
        s["temperature_set"] = s.pop("temperature")

    # XXX to remove once it is processed at the climate mode level
    if s["mode"] is not None and s["mode"].lower() in ["fan"]:
        s["temperature_set"] = config.DEFAULT_SETTINGS["temperature_set"]

    # use the default if power is off
    if s["power"].lower() == Power.OFF:
        s.update(config.DEFAULT_SETTINGS)

    # remove irrelevant properties
    s.pop("louver", None)
    s.pop("swing", None)
    s.pop("ventilation", None)

    return s


def has_prediction_fields(x) -> bool:
    # XXX Received temperature set could be 0
    # because set temperature was actually 0 for
    # the AC or because backend converts non parse-able
    # INT to 0. Might be fixed in the future.
    if x["temperature_set"] == 0:
        return False

    # fan could be None, on backend we use enum to
    # represent different values, if ir feature is not in
    # enum then None gets inserted. On AI for now, we don't
    # care about state of fan.
    return all([x[k] for k in config.PREDICTION_SETTINGS if k != "fan"])


def appliance_states_to_df(states, fields=config.PREDICTION_SETTINGS):
    index = np.array([np.datetime64(d["created_on"]) for d in states])
    dep_data = [[x[f] for f in fields] for x in states]
    data = pd.DataFrame(data=dep_data, index=index, columns=fields)
    return data


def interpolate(dates, data, method="time"):

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "parameter 'data' must be a DataFrame or TimeSeries"
            ", got {}".format(type(data))
        )

    ret = pd.DataFrame(index=dates, columns=data.columns, dtype=float)
    x = pd.concat([data, ret]).sort_index()
    x = x.interpolate(method=method)
    # TODO: solve this properly
    x = x.ffill().bfill()
    x = x.loc[dates]
    return x.groupby(x.index).last().sort_index()


def fill(dates, data):
    """Interpolate (forward fill) data at locations in dates.

    Parameters
    ----------
    dates : iterable of datetime objects
        the timestamps of the output data

    data : DataFrame
        the data

    Returns
    -------
    DataFrame
        filled DataFrame
    """
    if isinstance(data, pd.DataFrame):
        ret = pd.DataFrame(index=dates, columns=data.columns)
    elif isinstance(data, pd.Series):
        ret = pd.Series(index=dates, name=data.name)
    else:
        raise ValueError(
            "parameter data must be DataFrame or Series, got {}" "".format(type(data))
        )

    x = pd.concat([data, ret]).sort_index().ffill()
    x = x.loc[dates]
    return x.groupby(x.index).last()


def filter_time_between(dt, capture):
    """Remove captures if next capture is less than dt afterwards."""
    capture = sorted(capture, key=lambda x: x["created_on"])
    keep = []
    for c1, c2 in zip(capture[:-1], capture[1:]):
        if c2["created_on"] - c1["created_on"] >= dt:
            keep.append(c1)
    if capture:
        keep.append(capture[-1])
    return keep
