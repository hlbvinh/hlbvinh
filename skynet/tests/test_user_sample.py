from datetime import datetime
import math
import pprint
from datetime import timedelta

import pytest
import pytz
import pandas as pd

from ..user import sample as smp
from ..utils.database import queries
from ..utils.testing import gen_feature_matrix


def _check_non_feedback_features(sample):
    for feature in smp.NON_FEEDBACK_FEATURES:
        assert feature in sample, "feature {} missing in {}" "".format(
            feature, pprint.pformat(sample)
        )


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_compare_non_feedback_features(
    feedback_db, pool, cassandra_session, device_intervals
):
    for device in device_intervals:
        device_id = device["device_id"]
        feedback_query = queries.query_user_feedback(
            device_id=device_id, start=device["start"], end=device["end"]
        )
        feedback_rows = queries.execute(feedback_db, *feedback_query)

        assert feedback_rows

        for feedback in feedback_rows:
            fc = await smp.fetch_non_feedback_features(
                pool, cassandra_session, device_id, feedback["created_on"]
            )
            _check_non_feedback_features(fc)


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.usefixtures("feedback_db")
@pytest.mark.asyncio
async def test_user_samples_upload(user_sample_store, pool, cassandra_session):
    user_sample_store.clear_all()
    samples = await smp.make_all_feedback_samples(
        user_sample_store, pool, cassandra_session, ignore_watermark=True
    )
    fetched_samples = user_sample_store.get()

    def _sort(s):
        sorted(s, key=lambda x: (x["device_id"], x["timestamp"]))

    assert _sort(samples) == _sort(fetched_samples)
    user_sample_store.clear_all()


@pytest.mark.asyncio
async def test_fetch_sensors(cassandra_session, device_intervals):
    stypes = smp.COMFORT_SERVICE_SENSORS_REQUIRED
    for device in device_intervals:
        s = await smp.fetch_sensors(cassandra_session, stypes=stypes, **device)
        assert not s.empty, "non sensors fetched"

        # check if empty intervals (swapping start and end) raise
        with pytest.raises(smp.InsufficientData):
            await smp.fetch_sensors(
                cassandra_session,
                stypes=stypes,
                device_id=device["device_id"],
                start=device["end"],
                end=device["start"],
            )


@pytest.fixture
def invalid_sensor_samples():
    return pd.DataFrame(smp.INVALID_SENSORS, columns=["temperature", "humidity"])


@pytest.fixture
def valid_sensor_samples():
    return pd.DataFrame({"temperature": [3, 4], "humidity": [54, 55]})


@pytest.fixture
def raw_samples_from_mongo():
    return pd.DataFrame(
        {
            "compensated": [None],
            "temperature": [32.8],
            "humidity": [54.8],
            "temperature_out": [28.0],
            "humidity_out": [82.0],
            "timestamp": ["2015-06-09 12:34:40"],
        },
        columns=[
            "compensated",
            "temperature",
            "humidity",
            "temperature_out",
            "humidity_out",
            "timestamp",
        ],
    )


def test_filter_invalid_sensor_samples(invalid_sensor_samples):
    samples = smp.filter_invalid_sensor_samples(invalid_sensor_samples)
    assert samples.empty


def test_not_filter_valid_sensor_samples(valid_sensor_samples):
    samples = smp.filter_invalid_sensor_samples(valid_sensor_samples)
    assert len(samples) == len(valid_sensor_samples)


def test_optional_feature_are_added_to_features(device_id):
    timezone = pytz.timezone("Asia/Hong_Kong")
    timestamp = datetime.utcnow()
    weather = []
    sensors = gen_feature_matrix(
        ["temperature", "humidity", "created_on"], 100
    ).to_dict("records")
    features = smp.prepare_non_feedback_features(
        sensors, weather, timezone, timestamp, device_id, prediction=True
    )
    assert math.isnan(features["luminosity"])


def test_prepare_dataset(raw_samples_from_mongo):
    samples = smp.prepare_dataset(raw_samples_from_mongo)
    assert samples.iloc[0]["compensated"] is not None
    assert samples.iloc[0]["humidex"] is not None
    assert samples.iloc[0]["humidex_out"] is not None


def test_most_recent_works_with_single_sensor(sensors):
    assert [sensors] == smp.recent([sensors], recent_duration=timedelta(seconds=1))
