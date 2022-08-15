from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from asynctest import mock

from skynet.user.sample import OPTIONAL_FEATURE
from ...user import analytics_service, sample
from ...utils import testing
from ...utils.enums import Power

from ...sample.selection import COMPENSATE_COLUMN

np.random.seed(0)
FEATURES = sample.FEATURES_STORED + ["timestamp"]
COMPENSATED_FEATURES = FEATURES + [
    sensor + "_" + qualifier
    for sensor in ["humidity", "temperature"]
    for qualifier in ["raw", "refined"]
]
LEGACY_FEATURES = list(set(FEATURES) - set([COMPENSATE_COLUMN]))
NUM_SAMPLES = 10
IP = "127.0.0.1"
USER_ID = "USER_ID"
FIGURES = [
    "comfort_map",
    "comfort_factors",
    "temperature_and_humidity",
    "ac_running_time",
    "ac_set_points",
    "mode_usage",
]


@pytest.fixture
def states_to_aggregate():
    return [
        {
            "created_on": datetime(2015, 1, 1),
            "power": Power.OFF,
            "mode": "cool",
            "origin": "irdeployment",
        },
        {
            "created_on": datetime(2015, 1, 2),
            "power": Power.ON,
            "mode": "cool",
            "origin": "irdeployment",
        },
    ]


@pytest.fixture
def targets_to_aggregate():
    return [{"created_on": datetime(2015, 1, 1), "quantity": "manual"}]


def test_aggregate(states_to_aggregate, targets_to_aggregate):
    df = analytics_service.aggregate(
        states_to_aggregate,
        targets_to_aggregate,
        start=datetime(2015, 1, 1, 12),
        end=datetime(2015, 1, 2, 12),
    )
    assert df["created_on"].iat[0] == datetime(2015, 1, 1, 12)
    assert df["created_on"].iat[-1] == datetime(2015, 1, 2, 12)


@pytest.fixture
def timestamp_to_trim():
    return pd.DataFrame(
        [datetime(2015, 1, 1), datetime(2015, 1, 2)], columns=["created_on"]
    )


@pytest.fixture
def borders_to_trim():
    return [
        (datetime(2015, 1, 1), datetime(2015, 1, 2)),
        (datetime(2015, 1, 1, 12), datetime(2015, 1, 2, 2)),
    ]


def test_trim_df(timestamp_to_trim, borders_to_trim):
    for start, end in borders_to_trim:
        trimmed = analytics_service.trim_df(timestamp_to_trim, start=start, end=end)
        assert trimmed["created_on"].iat[0] == start
        assert trimmed["created_on"].iat[-1] == end

    # trimming before the start should raise an error, that data
    # does not exist
    with pytest.raises(ValueError):
        analytics_service.trim_df(
            timestamp_to_trim, start=datetime(2014, 1, 1), end=datetime(2015, 1, 2)
        )


def st(created_on, power, mode, temperature, origin):
    return {
        "created_on": created_on,
        "power": power,
        "mode": mode,
        "temperature": temperature,
        "origin": origin,
    }


def tr(created_on, quantity):
    return {"created_on": created_on, "quantity": quantity}


@pytest.fixture
def mode_usage_test_start():
    return datetime(2015, 1, 1)


@pytest.fixture
def mode_usage_test_end():
    return datetime(2015, 1, 13)


@pytest.fixture
def state_with_control_modes(mode_usage_test_start):
    return [
        st(mode_usage_test_start, Power.OFF, "cool", 24, "reverse"),
        st(datetime(2015, 1, 2), Power.ON, "cool", 25, "skynet"),
        st(datetime(2015, 1, 3), Power.OFF, "cool", 24, "skynet"),
        st(datetime(2015, 1, 4), Power.ON, "cool", 25, "skynet"),
        st(datetime(2015, 1, 5), Power.OFF, "cool", 24, "SkyNet"),
        st(datetime(2015, 1, 6), Power.OFF, "cool", 24, "Reverse"),
        st(datetime(2015, 1, 7), Power.ON, "heat", 25, "IrDeployment"),
        st(datetime(2015, 1, 8), Power.OFF, "heat", 30, "irdeployment"),
        st(datetime(2015, 1, 9), Power.ON, "heat", 25, "OPENAPI"),
        st(datetime(2015, 1, 10), Power.ON, "cool", 25, "OpenApi"),
        st(datetime(2015, 1, 10, 5), Power.OFF, "cool", 25, "openapi"),
        st(datetime(2015, 1, 11, 0), Power.ON, "cool", 25, "reverse"),
        st(datetime(2015, 1, 12, 0), Power.OFF, "cool", 25, "reverse"),
    ]


@pytest.fixture
def target_with_control_modes(mode_usage_test_start):
    return [
        tr(mode_usage_test_start, "manual"),
        tr(datetime(2015, 1, 2), "climate"),
        tr(datetime(2015, 1, 3), "away_temperature_upper"),
        tr(datetime(2015, 1, 6), "off"),
        tr(datetime(2015, 1, 7), "manual"),
        tr(datetime(2015, 1, 7, 18), "temperature"),
        tr(datetime(2015, 1, 8), "off"),
        tr(datetime(2015, 1, 9), "manual"),
        tr(datetime(2015, 1, 10), "manual"),
        tr(datetime(2015, 1, 10, 5), "off"),
        tr(datetime(2015, 1, 11), "manual"),
    ]


@pytest.fixture
def power_states(
    state_with_control_modes,
    target_with_control_modes,
    mode_usage_test_start,
    mode_usage_test_end,
):
    df = analytics_service.aggregate(
        state_with_control_modes,
        target_with_control_modes,
        start=mode_usage_test_start,
        end=mode_usage_test_end,
    )
    return analytics_service.power_mode_stats(
        df, mode_usage_test_start, mode_usage_test_end
    )


def test_mode_usage(power_states):
    assert power_states["comfort"] == 24
    assert power_states["temperature"] == 6
    assert power_states["away_on"] == 24
    assert power_states["away_off"] == 2 * 24
    assert power_states["remote_on"] == 24
    assert power_states["remote_off"] == 24 * 3
    assert power_states["api_on"] == 29
    assert power_states["api_off"] == 19
    assert power_states["manual_on"] == 18
    assert power_states["manual_off"] == 24
    assert power_states["on"] == (
        power_states["comfort"]
        + power_states["temperature"]
        + power_states["away_on"]
        + power_states["remote_on"]
        + power_states["api_on"]
        + power_states["manual_on"]
    )
    assert power_states["off"] == (
        power_states["remote_off"]
        + power_states["api_off"]
        + power_states["manual_off"]
    )
    assert power_states["manual"] == (
        power_states["remote_on"] + power_states["api_on"] + power_states["manual_on"]
    )


@pytest.fixture
def current_weather():
    return {"temperature_out": 20.0, "humidity_out": 50.0}


@pytest.fixture(params=[LEGACY_FEATURES, COMPENSATED_FEATURES])
def features(request, device_intervals, current_weather):
    """
    generate a feature DF of size 1000, with 10 distinct user_id
    and each user_id has 2 distance device_id
    """
    random_features_cols = [
        f
        for f in request.param
        if f not in sample.WEATHER_FEATURES + sample.TIMESERIES_FEATURES
    ]
    features = []
    for device in device_intervals:
        feature = testing.gen_feature_matrix(random_features_cols, NUM_SAMPLES)
        feature["user_id"] = USER_ID
        feature["device_id"] = device["device_id"]
        feature["temperature_out"] = np.random.normal(
            current_weather["temperature_out"], size=NUM_SAMPLES
        )
        feature["humidity_out"] = np.random.normal(
            current_weather["humidity_out"], size=NUM_SAMPLES
        )
        features.append(feature)
    return pd.concat(features)


@pytest.fixture
async def client_service(
    user_sample_store, rediscon, pool, cassandra_session, features, port
):
    """
    activate AnalyticsService at given IP and port
    """
    for f in features.to_dict("records"):
        user_sample_store.upsert(f)

    s = analytics_service.AnalyticsService(
        ip=IP,
        port=port,
        redis=rediscon,
        pool=pool,
        session=cassandra_session,
        user_sample_store=user_sample_store,
    )
    # Dealear Actor still need a logger
    c = DealerActor(ip=IP, port=port, log=analytics_service.log)
    yield c, s

    user_sample_store.clear()


@pytest.fixture
def client(client_service):
    return client_service[0]


@pytest.mark.asyncio
async def test_with_data(get_response, client, device_intervals):
    for device in device_intervals:
        msg = {
            "method": "UserAnalyticsActor",
            "params": {
                "user_id": USER_ID,
                "device_id": device["device_id"],
                "timestamp": device["end"],
            },
            "session_id": None,
        }

        # sent the messages to clients and received responses
        resp = await get_response(client, msg)

        # check if we have data for all figures
        for key in FIGURES:
            assert resp[key]

        # check if response contain all features' feature importances
        types = [r["type"] for r in resp["comfort_factors"]]

        for key in analytics_service.FEATURES_GROUP:
            assert key in types

        # check if they all return a data field with the number of
        # points used for the spider chart data
        datas = [r["data"] for r in resp["comfort_factors"]]
        assert isinstance(datas[0], int)
        for d in datas:
            assert d == datas[0]

        for key in ["indoor", "outdoor"]:
            assert resp["temperature_and_humidity"][key]


@pytest.mark.asyncio
async def test_bad_request(get_response, client):
    for params in [
        {},
        {"user_id": "U"},
        {"device_id": "D"},
        {"timestamp": datetime(2015, 1, 1)},
    ]:
        msg = {"method": "UserAnalyticsActor", "session_id": None, "params": params}
        resp = await get_response(client, msg)
        assert resp == []


@pytest.mark.asyncio
async def test_no_data(get_response, client):
    msg = {
        "method": "UserAnalyticsActor",
        "session_id": None,
        "params": {"user_id": "u", "device_id": "d"},
    }
    resp = await get_response(client, msg)
    # check if we have all figures keys
    for key in FIGURES:
        assert key in resp


@pytest.mark.asyncio
async def test_missing_data(get_response, client, device_intervals):
    for device in device_intervals:
        msg = {
            "method": "UserAnalyticsActor",
            "params": {
                "user_id": USER_ID,
                "device_id": device["device_id"],
                "timestamp": device["end"],
            },
            "session_id": None,
        }

        # sent the messages to clients and received responses

        # needs to mock the wrapping of make_comfort_stats as well
        m = analytics_service.log_error(device_id="dev", user_id="user")(
            mock.CoroutineMock(side_effect=Exception)
        )

        with mock.patch("skynet.user.analytics_service.make_comfort_stats", m):
            resp = await get_response(client, msg)
            for key in ["comfort_factors", "comfort_map"]:
                assert key in resp


@pytest.fixture
def missing_columns_data():
    return pd.DataFrame(np.zeros([10, 2]), columns=["wrong", "columns"])


def test_feature_importance_of_optional_feature():
    features = set(
        analytics_service.FEATURES_TRAIN + analytics_service.COMFORT_MAP_FEATURES
    )
    df = testing.gen_feature_matrix(features, 100)
    df[OPTIONAL_FEATURE] = np.nan
    importance = analytics_service.comfort_factors(df)
    optional_feature_group = next(
        k for k, v in analytics_service.FEATURES_GROUP.items() if OPTIONAL_FEATURE in v
    )
    assert importance[optional_feature_group] == 0.0


def test_missing_comfort_data_columns(missing_columns_data):
    # previously logging was coupled with classes, monkey patching the test
    old_log = analytics_service.log
    log = mock.MagicMock()
    analytics_service.log = log
    factors = analytics_service.comfort_factors(missing_columns_data)
    assert factors == {}
    log.error.assert_called_once_with(mock.ANY)
    # test with real logger
    analytics_service.log = old_log
    factors = analytics_service.comfort_factors(missing_columns_data)
    assert factors == {}


@pytest.fixture(params=["temperature_out", "humidity_out"])
def missing_weather_samples(request, features):
    return features.drop(request.param, axis=1)


def test_comfort_map(features, current_weather):
    assert analytics_service.comfort_map(features, current_weather)


def test_regression_comfort_samples_missing_weather(
    current_weather, missing_weather_samples
):
    # When all features had no weather but current weather data was
    # available this used to raise a KeyError due to missing temperature_out
    # and humidity_out fields.
    assert analytics_service.comfort_map(missing_weather_samples, current_weather)


def test_regression_comfort_samples_all_missing_weather(features, current_weather):
    both_missing = features.drop(["temperature_out", "humidity_out"], axis=1)
    assert analytics_service.comfort_map(both_missing, current_weather)


@pytest.fixture
def empty_sensor_data():
    return []


def test_regression_aggregate_sensors_missing_data(empty_sensor_data):
    # This used to fail trying to group by "created_on" key
    assert analytics_service.aggregate_sensor_data(empty_sensor_data) == []


def test_regression_comfort_sample_double_compensation(features, current_weather):
    for comfort in analytics_service.comfort_map(features, current_weather):
        assert comfort[COMPENSATE_COLUMN]
