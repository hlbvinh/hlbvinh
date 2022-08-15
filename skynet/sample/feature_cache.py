from datetime import datetime, timedelta
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import pytz

from skynet.sample import selection
from skynet.prediction.climate_model import FEATURE_COLUMNS
from skynet.prediction.mode_model import HISTORICAL_FEATURE_COLUMNS
from ..control.util import AMBI_SENSOR_INTERVAL_SECONDS
from ..user.sample import (
    get_time_features,
    recent,
    timeseries,
    COMFORT_TIMESERIES_INTERVAL,
    USE_COMFORT_LSTM,
)
from ..utils import cache_util, thermo
from ..utils.async_util import multi
from ..utils.types import ApplianceState, Connections, Sensors, Weather

WEATHER_FEATURES = [
    "temperature_out",
    "humidity_out",
    "humidex_out",
    "temperature_out_mean_day",
]


# pylint: disable=no-value-for-parameter
def to_localtime(timezone, naive_utc_timestamp):
    return pytz.UTC.localize(naive_utc_timestamp).astimezone(timezone)


# pylint: enable=no-value-for-parameter


class FeatureData:
    def __init__(self, log_fun: Callable = print) -> None:
        self._data: Dict[
            str, Union[List[Sensors], List[Weather], List[ApplianceState]]
        ] = {"sensors": [], "weather": [], "appliance_state": []}
        self.log_fun = log_fun

    def add(self, data_type, values):
        if isinstance(values, list):
            self._data[data_type].extend(values)
        else:
            self._data[data_type].append(values)

    def get_history_features(self):
        f = {
            "weather": self.select_weather_features(),
            "sensors": self.select_sensor_features_climate(),
            "states": self.select_appliance_state_features(),
        }
        feature_selector = {
            k: v
            for k, v in selection.SELECTORS.items()
            if k in set(FEATURE_COLUMNS + HISTORICAL_FEATURE_COLUMNS)
        }
        feat = selection.check_select_features(
            feature_selector, f, self.log_fun, is_prediction=True
        )
        feat["appliance_id"] = f["states"]["appliance_id"].values[0]
        return feat

    def get_user_features(self, timezone, timestamp, is_in=False):
        f = self.select_weather_user_features()
        f.update(self.select_sensor_features_user(is_in))
        f.update(get_time_features(to_localtime(timezone, timestamp)))
        return f

    def select_sensor_features_climate(self):
        return pd.DataFrame(self._data["sensors"]).set_index("created_on")

    def select_sensor_features_user(self, is_in=False):
        sensors = self._data["sensors"][-1]
        f = {k: sensors[k] for k in ["temperature", "humidity", "humidex"]}
        f.update({"luminosity": sensors.get("luminosity", np.nan)})
        if is_in:
            # some users might have a comfort model sensible to humidex and
            # humidity, humidity readings are quite likely to vary a lot, here
            # we add a low pass filter with a simple temporal average to
            # stabilise the input to the comfort model and hopefully the whole
            # control loop as a consequence.
            recent_index = int(
                timedelta(minutes=20).total_seconds() // AMBI_SENSOR_INTERVAL_SECONDS
            )
            f["humidity"] = np.mean(
                self.get_historical_sensor("humidity")[-recent_index:]
            )
            f["humidex"] = thermo.humidex(f["temperature"], f["humidity"])
        if USE_COMFORT_LSTM:
            f.update(
                timeseries(
                    recent(
                        self._data["sensors"],
                        recent_duration=COMFORT_TIMESERIES_INTERVAL,
                    )
                )
            )
        return f

    def select_appliance_state_features(self):
        return (
            pd.DataFrame(self._data["appliance_state"])
            .rename(columns={"temperature": "temperature_set"})
            .set_index("created_on")
        )

    def select_weather_user_features(self):
        weather = self._data["weather"]
        if not weather:
            self.log_fun("no weather data")
            f = {
                "temperature_out": np.nan,
                "humidity_out": np.nan,
                "humidex_out": np.nan,
                "temperature_out_mean_day": np.nan,
            }
        else:
            f = {}
            f["temperature_out"] = np.mean([w["temperature_out"] for w in weather])
            f["humidity_out"] = np.mean([w["humidity_out"] for w in weather])
            f["humidex_out"] = thermo.humidex(f["temperature_out"], f["humidity_out"])
        return f

    def select_weather_features(self):
        weather = self._data["weather"]
        if not weather:
            return pd.DataFrame()
        weather = pd.DataFrame(weather).set_index("timestamp")
        weather["humidex_out"] = thermo.humidex(
            weather["temperature_out"], weather["humidity_out"]
        )
        return weather

    def get_historical_sensor(self, sensor_type):
        return [
            sensor[sensor_type] for sensor in interpolate_sensors(self._data["sensors"])
        ]


class RedisFeatureData(FeatureData):
    def __init__(
        self, connections: Connections, device_id: str, log_fun: Callable = print
    ) -> None:
        self.connections = connections
        self.device_id = device_id
        super().__init__(log_fun)

    async def load_state(self) -> None:
        end = datetime.utcnow()
        coroutines = {
            "sensors": cache_util.get_sensors_range(
                self.connections.redis,
                self.device_id,
                end - selection.TIMESERIES_INTERVAL,
                end,
            ),
            "weather": cache_util.fetch_weather_from_device(
                self.connections, self.device_id, end - timedelta(days=1), end
            ),
            "appliance_state": cache_util.fetch_appliance_state(
                self.connections, self.device_id
            ),
        }
        data = await multi(coroutines)

        for data_type, values in data.items():
            self.add(data_type, values)


def interpolate_sensors(sensors: List[Sensors]) -> List[Sensors]:
    df = pd.DataFrame(sensors)
    df = (
        df.set_index("created_on")
        .resample(str(AMBI_SENSOR_INTERVAL_SECONDS) + "S")
        .mean()
        .interpolate()
    )
    return df.reset_index().to_dict("records")
