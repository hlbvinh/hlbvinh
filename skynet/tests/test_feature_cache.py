from datetime import datetime, timedelta

import math
import numpy as np
import pandas as pd
import pytest
import pytz

from ..prediction import climate_model, mode_model
from ..control.util import AMBI_SENSOR_INTERVAL_SECONDS
from ..sample import feature_cache
from ..user.sample import COMFORT_FEATURES_WITH_TARGET, FEEDBACK_FEATURES
from ..utils import cache_util, thermo
from ..utils.enums import Power
from ..utils.types import Connections


def _add_data(fd, **kwargs):
    for k, v in kwargs.items():
        fd.add(k, v)


def _check(
    hist_f,
    sensors=None,
    weather=None,
    appliance_state=None,
    test_temperature=False,
    test_humidity=False,
    test_luminosity=False,
):
    if test_temperature:
        assert hist_f["temperature"] == sensors["temperature"]
    if test_humidity:
        assert hist_f["humidity"] == sensors["humidity"]
    if test_temperature and test_humidity:
        assert hist_f["humidex"] == thermo.humidex(
            sensors["temperature"], sensors["humidity"]
        )
    if test_luminosity:
        assert hist_f["luminosity"] == sensors["luminosity"]

    if weather:
        assert hist_f["temperature_out"] == weather["temperature_out"]
        assert hist_f["humidity_out"] == weather["humidity_out"]

    if appliance_state:
        assert hist_f["mode_hist"] == appliance_state["mode"]
        assert hist_f["power_hist"] == appliance_state["power"]
        assert hist_f["temperature_set_last"] == appliance_state["temperature"]


def make_history_feature(feature_data, **kwargs):
    _add_data(feature_data, **kwargs)
    hist_f = feature_data.get_history_features()
    return hist_f


def get_sample_data(timestamp=datetime(2015, 1, 1)):
    return {
        "sensors": {
            "temperature": 25.0,
            "humidity": 50.0,
            "luminosity": 9.9,
            "humidex": thermo.humidex(25.0, 50.0),
            "created_on": timestamp,
        },
        "weather": {
            "humidity_out": 70.0,
            "temperature_out": 17.0,
            "timestamp": timestamp,
        },
        "appliance_state": {
            "mode": "cool",
            "temperature": 24,
            "power": Power.ON,
            "created_on": timestamp,
            "appliance_id": "app_a",
            "appliance_state_id": 0,
        },
    }


@pytest.fixture
def feature_data():
    return feature_cache.FeatureData()


@pytest.fixture
def history_feature(feature_data):
    fd = feature_data
    return make_history_feature(fd, **get_sample_data())


def test_cached_climate_sample(history_feature):
    d = get_sample_data()
    _check(
        history_feature,
        **{k: d[k] for k in ["sensors", "weather", "appliance_state"]},
        test_temperature=True,
        test_humidity=True
    )


def test_add_values(feature_data):
    init_data = get_sample_data(datetime(2015, 1, 1))
    fd = feature_data
    _add_data(fd, **init_data)
    timestamp = datetime(2015, 1, 1, 0, 0, 30)
    sensors = {
        "temperature": 24.0,
        "humidity": 40.0,
        "humidex": thermo.humidex(24.0, 40.0),
        "created_on": timestamp,
    }
    weather = {"humidity_out": 80.0, "temperature_out": 18.0, "timestamp": timestamp}
    appliance_state = {
        "mode": "heat",
        "temperature": 23,
        "power": Power.OFF,
        "created_on": timestamp,
        "appliance_id": "app_b",
        "appliance_state_id": 1,
    }
    fd.add("sensors", sensors)
    fd.add("weather", weather)
    fd.add("appliance_state", appliance_state)
    hist_f = fd.get_history_features()
    _check(
        hist_f,
        sensors,
        weather,
        appliance_state,
        test_temperature=True,
        test_humidity=True,
    )
    np.allclose(
        hist_f["previous_temperatures"],
        (init_data["sensors"]["temperature"] + sensors["temperature"]) / 2,
    )


def test_missing_weather_data(feature_data):
    fd = feature_data
    d = get_sample_data()
    _add_data(fd, **{k: d[k] for k in ["sensors", "appliance_state"]})
    hist_f = fd.get_history_features()
    for feat in feature_cache.WEATHER_FEATURES:
        assert pd.isnull(hist_f[feat])


feature_data_bis = feature_data


def test_add_list(feature_data, feature_data_bis):
    fd0 = feature_data
    fd1 = feature_data_bis
    d = get_sample_data()
    _add_data(fd0, **d)
    fd1.add("sensors", [d["sensors"]] * 2)
    fd1.add("appliance_state", [d["appliance_state"]] * 2)
    fd1.add("weather", [d["weather"]] * 2)
    hist_f0 = fd0.get_history_features()
    hist_f1 = fd1.get_history_features()
    for (k0, v0), (k1, v1) in zip(sorted(hist_f0.items()), sorted(hist_f1.items())):
        assert k0 == k1
        assert v0 == v1 or np.isclose(v0, v1) or (np.isnan(v0) and np.isnan(v1))


def test_all_required_history_features_cached(history_feature):
    """Test if features required by climate and mode model are all cached."""
    input_later = [
        "mode",
        "power",
        "temperature_set",
        "target_humidity",
        "target_humidex",
        "target_temperature",
    ]
    feature_names = (
        climate_model.FEATURE_COLUMNS + mode_model.HISTORICAL_FEATURE_COLUMNS
    )
    feature_names = list(set(feature_names) - set(input_later))
    for name in feature_names:
        assert name in history_feature


def test_all_required_user_features_cached(feature_data):
    fd = feature_data
    _add_data(fd, **get_sample_data())

    # 2015 01 01 in HKT
    features = fd.get_user_features(
        pytz.timezone("Asia/Hong_Kong"), datetime(2015, 12, 31, 16)
    )

    required = set(COMFORT_FEATURES_WITH_TARGET) - set(FEEDBACK_FEATURES)
    for name in required:
        assert name in features


def test_luminosity_in_user_features(feature_data):
    data = get_sample_data()
    del data["sensors"]["luminosity"]

    fd = feature_data
    _add_data(fd, **data)

    features = fd.get_user_features(
        pytz.timezone("Asia/Hong_Kong"), datetime(2015, 12, 31, 16)
    )
    assert math.isnan(features["luminosity"])


def test_user_features_values(feature_data):
    fd = feature_data
    d = get_sample_data()
    _add_data(fd, **d)

    # 2015 01 01 in HKT
    features = fd.get_user_features(
        pytz.timezone("Asia/Hong_Kong"), datetime(2015, 12, 31, 16)
    )

    # XXX This should return 0 for sin and 1 for cos, but due to the bug in
    # skynet/user/sample the names of the features are reversed.
    assert features["tod_sin"] == 1.0, (
        "If you fixed the sin/cos bug, thanks," "please fix them here too."
    )
    assert features["tod_cos"] == 0.0

    # test the sensor features
    t = d["sensors"]["temperature"]
    h = d["sensors"]["humidity"]
    assert features["temperature"] == t
    assert features["humidity"] == h
    assert features["humidex"] == thermo.humidex(t, h)
    assert features["luminosity"] == d["sensors"]["luminosity"]

    # test weather features
    t_out = d["weather"]["temperature_out"]
    h_out = d["weather"]["humidity_out"]
    assert features["temperature_out"] == t_out
    assert features["humidity_out"] == h_out
    assert features["humidex_out"] == thermo.humidex(t_out, h_out)


def test_get_historical_sensor(feature_data, maxnum=5):
    timestamp = datetime.utcnow()
    sensors_values = [
        {
            "temperature": i,
            "humidity": i + maxnum,
            "created_on": timestamp + timedelta(seconds=(i * 30)),
        }
        for i in range(maxnum)
    ]

    # add test values
    feature_data.add("sensors", sensors_values)

    # test get_historical return proper output
    assert feature_data.get_historical_sensor("temperature") == list(range(maxnum))
    assert feature_data.get_historical_sensor("humidity") == list(
        range(maxnum, maxnum * 2)
    )


@pytest.fixture
async def feature_redis(rediscon, device_id):
    d = get_sample_data(datetime.utcnow())
    await cache_util.set_sensors(rediscon, device_id, d["sensors"])
    await cache_util.set_weather_redis(rediscon, device_id, [d["weather"]])
    await cache_util.set_appliance_state(
        redis=rediscon, key_arg=device_id, value=d["appliance_state"]
    )
    yield rediscon


@pytest.fixture
def feature_connections(feature_redis, pool):
    return Connections(redis=feature_redis, pool=pool)


@pytest.fixture
def redis_feature_data(feature_connections, device_id):
    return feature_cache.RedisFeatureData(feature_connections, device_id)


@pytest.mark.asyncio
async def test_stateless_feature_data(redis_feature_data):
    feature_data = redis_feature_data
    await feature_data.load_state()
    for data in feature_data._data.values():
        assert len(data) == 1


@pytest.fixture
def sensors():
    timestamp = datetime.utcnow()
    return [
        {"created_on": timestamp, "temperature": 24},
        {"created_on": timestamp + timedelta(seconds=60), "temperature": 22},
        {"created_on": timestamp + timedelta(seconds=120), "temperature": 20},
        {"created_on": timestamp + timedelta(seconds=180), "temperature": 18},
    ]


def test_interpolate_sensors(sensors):
    result = feature_cache.interpolate_sensors(sensors)
    sensors_interval = [
        (nxt["created_on"] - cur["created_on"]).total_seconds()
        for cur, nxt in zip(result, result[1:])
    ]
    assert all(
        [interval == AMBI_SENSOR_INTERVAL_SECONDS for interval in sensors_interval]
    )
