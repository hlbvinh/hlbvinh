from typing import Any, Union

import numpy as np
import pandas as pd

from .log_util import get_logger

log = get_logger(__name__)

MIN_CELSIUS = 8
MAX_CELSIUS = 50
DEFAULT_TEMPERATURE = 24


def humidex(
    temperature, humidity, a=6.122, b=7.5, name: str = "humidex"
) -> Union[pd.DataFrame, pd.Series]:
    e = a * 10.0 ** (b * temperature / (237.7 + temperature)) * humidity / 100.0
    hum = temperature + 5.0 / 9.0 * (e - 10.0)
    if isinstance(hum, pd.Series):
        hum.name = name
    elif isinstance(hum, pd.DataFrame):
        hum.columns = [name]
    return hum


def celsius_from_fahrenheit(t: float) -> float:

    return (t - 32) / 1.8


def fix_temperature(t) -> float:
    try:
        t = np.float64(t)
    except (TypeError, ValueError):
        return np.nan
    else:
        if t > MAX_CELSIUS:
            return celsius_from_fahrenheit(t)
        if t < MIN_CELSIUS:
            return t + DEFAULT_TEMPERATURE
        return t


fix_temperatures = np.vectorize(fix_temperature)


def is_int_one(x: Any) -> bool:
    try:
        int(x)
        return True

    except (ValueError, TypeError):
        return False


is_int = np.vectorize(is_int_one)
