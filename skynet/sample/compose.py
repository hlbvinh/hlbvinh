import numpy as np

from ..utils import thermo
from ..utils.compensation import compensate_sensors


def make_humidex(x):
    return thermo.humidex(x["temperature"], x["humidity"])


def make_outdoor_humidex(x):
    temperature_out = x["temperature_out"]
    humidity_out = x["humidity_out"]
    return thermo.humidex(temperature_out, humidity_out)


def compose(data, raise_on_missing=True):
    """Note: this does inplace modification on the input DataFrames."""

    if "weather" in data:
        weather = data["weather"]
        try:
            humidex_out = make_outdoor_humidex(weather).values
        except KeyError:
            humidex_out = np.nan
        weather["humidex_out"] = humidex_out

    sensors = data["sensors"]
    try:
        sensors = compensate_sensors(sensors)
        humidex = make_humidex(sensors).values
    except KeyError as exc:
        if raise_on_missing:
            raise ValueError("missing sensor data") from exc
    else:
        sensors["humidex"] = humidex
    data["sensors"] = sensors
    return data
