import asyncio
import functools
from datetime import datetime, timedelta, tzinfo
from itertools import chain
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pytz

from ..control.util import AMBI_SENSOR_INTERVAL_SECONDS
from ..prediction.util import time_of_day, time_of_week, time_of_year
from ..utils import data, preprocess, thermo
from ..utils.async_util import multi
from ..utils.compensation import (
    COMPENSATE_COLUMN,
    compensate_features,
    compensate_sensors,
)
from ..utils.database import cassandra_queries, queries
from ..utils.database.cassandra import CassandraSession
from ..utils.database.dbconnection import Pool
from ..utils.log_util import get_logger
from ..utils.types import Sensors, Weather

log = get_logger(__name__)

SAMPLE_CONCURRENCY = 5
GROUP_INTERVAL = "1H"

# Don't fetch data from less than FETCH_DELAY ago. New data could it.
MIN_FEEDBACK_INTERVAL = timedelta(minutes=1)
WEATHER_AVERAGE_INTERVAL = timedelta(days=1)
WEATHER_FREQ = "1H"
COMFORT_TIMESERIES_FREQ = timedelta(seconds=150)
COMFORT_TIMESERIES_INTERVAL = timedelta(minutes=30)

USE_COMFORT_LSTM = False

COMFORT_MODEL_TARGET = "feedback"

# XXX sin/cos order is actually reversed in the function that computes them
# fix this
DAY_FEATURES = ["tod_cos", "tod_sin"]
WEEK_FEATURES = ["tow_cos", "tow_sin"]
YEAR_FEATURES = ["toy_cos", "toy_sin"]
TIME_FEATURES = DAY_FEATURES + WEEK_FEATURES + YEAR_FEATURES

OUTDOOR_TEMPERATURE = "temperature_out"
OUTDOOR_HUMIDITY = "humidity_out"
WEATHER_FEATURES = [OUTDOOR_TEMPERATURE, OUTDOOR_HUMIDITY]

SENSOR_FEATURES = ["temperature", "humidity", "pirload", "pircount", "luminosity"]

OPTIONAL_FEATURE = "luminosity"
FEEDBACK_FEATURES = ["device_id", "user_id", "feedback"]

TIMESERIES_FEATURES = ["previous_temperatures", "previous_humidities"]

FEATURES_STORED_REQUIRED = FEEDBACK_FEATURES + ["humidity", "temperature"]
FEATURES_STORED = (
    TIME_FEATURES
    + WEATHER_FEATURES
    + SENSOR_FEATURES
    + FEEDBACK_FEATURES
    + TIMESERIES_FEATURES
    + [COMPENSATE_COLUMN]
)

NON_FEEDBACK_FEATURES = sorted(set(FEATURES_STORED) - set(FEEDBACK_FEATURES))


COMFORT_FEATURES_TRAIN = [
    "device_id",
    "user_id",
    "luminosity",
    "humidex",
    "humidity",
    "temperature",
    "humidex_out",
    "humidity_out",
    "temperature_out",
    "tod_cos",
    "tod_sin",
    "tow_cos",
    "tow_sin",
    "toy_cos",
    "toy_sin",
]
COMFORT_FEATURES_TRAIN = (
    COMFORT_FEATURES_TRAIN + TIMESERIES_FEATURES
    if USE_COMFORT_LSTM
    else COMFORT_FEATURES_TRAIN
)
COMFORT_FEATURES_WITH_TARGET = COMFORT_FEATURES_TRAIN + [COMFORT_MODEL_TARGET]

COMFORT_SERVICE_SENSORS_REQUIRED = ["temperature", "humidity", "luminosity"]

USER_FEEDBACK_TYPE = "user_feedback"
INVALID_SENSORS = [
    {"temperature": 0, "humidity": 0},
    {"temperature": -4, "humidity": 16},
]

# Sensors within last 2 minutes are considered recent
RECENT_SENSORS_TIMEDELTA = timedelta(seconds=(AMBI_SENSOR_INTERVAL_SECONDS * 4))

MAPS = {
    "humidex": lambda x: thermo.humidex(x["temperature"], x["humidity"]),
    "humidex_out": lambda x: thermo.humidex(
        x["temperature_out"], x["humidity_out"], name="humidex_out"
    ),
}


class InsufficientData(Exception):
    pass


def get_log_msg(msg, device_id, start=None, end=None):
    log_msg = "{} {}".format(device_id, msg)
    if start is not None:
        log_msg += " from {}".format(start)
    if end is not None:
        log_msg += " to {}".format(end)
    return log_msg


def prepare_dataset(dataset):

    dataset = compensate_features(dataset)

    for feat, fun in MAPS.items():
        log.info(f"computing {feat} feature")
        dataset[feat] = fun(dataset)

    dataset = filter_invalid_sensor_samples(dataset)

    log.debug({"n_samples": len(dataset), "event": "prepare_dataset"})
    if "timestamp" in dataset:
        return dataset.set_index("timestamp")
    return dataset


def filter_invalid_sensor_samples(samples: pd.DataFrame) -> pd.DataFrame:
    # some of the devices have broken sensors. Broken sensors have
    # temperature, humidity set to 0, 0 or -4, 16.

    invalidity_criteria = [
        (
            (samples["temperature"] == sensor["temperature"])
            & (samples["humidity"] == sensor["humidity"])
        )
        for sensor in INVALID_SENSORS
    ]

    valid_samples = get_valid_samples(samples, invalidity_criteria)
    log.info(
        f"{(1 - len(valid_samples)/len(samples))*100}% samples filtered because of broken sensors."
    )
    return valid_samples


def get_valid_samples(
    samples: pd.DataFrame, invalidity_criteria: List[pd.Series]
) -> pd.DataFrame:
    return samples.loc[
        np.invert(
            np.bitwise_or.reduce(invalidity_criteria)  # pylint: disable=maybe-no-member
        )
    ]


async def fetch_sensors(sess, stypes, device_id, start, end):
    rows = await sess.execute_async(
        *cassandra_queries.query_sensor(device_id, start, end, stypes=stypes)
    )
    if not rows:
        raise InsufficientData(get_log_msg("no sensor data", device_id, start, end))

    return pd.DataFrame(rows).set_index("created_on")


async def fetch_weather(pool, device_id, start, end):
    weather = await queries.get_weather_api_data_from_device(
        pool,
        device_id,
        start - WEATHER_AVERAGE_INTERVAL,
        end,
        columns=[
            "timestamp",
            "temperature as " + OUTDOOR_TEMPERATURE,
            "100 * humidity as " + OUTDOOR_HUMIDITY,
        ],
    )

    if weather:
        weather_df = pd.DataFrame.from_records(weather, index="timestamp")
        n_points = int(WEATHER_AVERAGE_INTERVAL.total_seconds() / 3600.0)
        daily = (
            weather_df.resample(WEATHER_FREQ)
            .mean()
            .dropna()
            .rolling(window=n_points, center=False)
            .mean()
            .dropna()
        )
        # this now has one point before the start timestamp
        if not daily.empty:
            return daily[daily.index >= start]
        return daily
    return pd.DataFrame(columns=[OUTDOOR_TEMPERATURE, OUTDOOR_HUMIDITY])


async def fetch(pool, sess, device_id, start, end, interval=GROUP_INTERVAL):
    """Fetch PREDICTION sample data and aggregate into DataFrame
    Returns
    -------
    data: DataFrame
        columns: humidex, pircount, power

    Raises
    ------
    InsufficientData
        When sensor data is missing.
    """

    stypes = COMFORT_SERVICE_SENSORS_REQUIRED

    data = await multi(
        {
            "weather": fetch_weather(pool, device_id, start, end),
            "sensor_df": fetch_sensors(sess, stypes, device_id, start, end),
        }
    )
    weather = data["weather"]
    sensor_df = data["sensor_df"].resample(interval).mean()

    log.debug(
        get_log_msg(
            "fetched weather", device_id, weather.index.min(), weather.index.max()
        )
    )
    log.debug(
        get_log_msg(
            "fetched sensors", device_id, sensor_df.index.min(), sensor_df.index.max()
        )
    )

    weather = (
        preprocess.interpolate(sensor_df.index, weather)
        .rolling(window=3, min_periods=1, center=False)
        .mean()
    )
    return pd.concat([sensor_df, weather], axis=1)


async def fetch_non_feedback_features(
    pool, session, device_id, timestamp, prediction=False
):
    """Fetch non-feedback features for comfort model

    NOTE: The user_id is a feedback feature so needs to be set separately if
          it is required.
    """
    log.debug(get_log_msg("fetching features at {}".format(timestamp), device_id))
    data = await multi(
        {
            "sensors": fetch_sensors_timestamp(session, device_id, timestamp),
            "weather": fetch_weather_from_device(pool, device_id, timestamp),
            "timezone": queries.get_time_zone(pool, device_id),
        }
    )

    features = prepare_non_feedback_features(
        **data, timestamp=timestamp, device_id=device_id, prediction=prediction
    )
    return features


async def fetch_sensors_timestamp(
    session: CassandraSession, device_id: str, timestamp: datetime
):
    sensor_query = cassandra_queries.query_sensor(
        device_id,
        timestamp - COMFORT_TIMESERIES_INTERVAL,
        timestamp,
        stypes=SENSOR_FEATURES,
    )
    rows = await session.execute_async(*sensor_query)
    return [compensate_sensors(row._asdict()) for row in rows]


async def fetch_weather_from_device(pool: Pool, device_id: str, timestamp: datetime):
    location = await pool.execute(
        *queries.query_location_from_device(device_id, ["location_id"])
    )
    if location:
        location_id = location[0]["location_id"]
        weather_query = queries.query_weather(location_id, timestamp)
        weather = await pool.execute(*weather_query)
        return weather


def prepare_non_feedback_features(
    sensors: List[Sensors],
    weather: List[Weather],
    timezone: tzinfo,
    timestamp: datetime,
    device_id: str,
    prediction: bool,
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}

    if sensors:
        features.update(average(recent(sensors, RECENT_SENSORS_TIMEDELTA)))
        features.update(timeseries(sensors))

    if weather:
        features.update(average(weather))

    time_features = make_time_features(timezone, timestamp)
    features.update(time_features)

    if prediction:
        for feature_name, fun in MAPS.items():
            try:
                features[feature_name] = fun(features)
            except KeyError:
                pass

    # throw away nan features for training samples
    features = {k: v for k, v in features.items() if np.all(np.isfinite(v))}

    # add static features
    features["device_id"] = device_id

    if OPTIONAL_FEATURE not in features:
        features[OPTIONAL_FEATURE] = np.nan
    return features


def recent(records: List[Sensors], recent_duration: timedelta) -> List[Sensors]:
    recent_timestamp = records[-1]["created_on"]
    timestamp = recent_timestamp - recent_duration
    return [record for record in records if record["created_on"] >= timestamp]


def average(records: List[Sensors]) -> pd.Series:
    return pd.DataFrame(records).mean()


def timeseries(records: List[Sensors]) -> Dict[str, List[float]]:
    resampled = (
        pd.DataFrame(records)
        .set_index("created_on")
        .resample(COMFORT_TIMESERIES_FREQ)
        .mean()
        .interpolate()
        .dropna()
    )
    return {
        "previous_temperatures": resampled["temperature"].tolist(),
        "previous_humidities": resampled["humidity"].tolist(),
    }


def make_time_features(timezone: tzinfo, timestamp: datetime):
    return get_time_features(local_timestamp(timezone, timestamp))


def local_timestamp(timezone: tzinfo, timestamp: datetime) -> datetime:
    return pytz.utc.localize(timestamp).astimezone(timezone)


async def fetch_feedback_samples(
    pool: Pool,
    session: CassandraSession,
    device_id: str,
    start: datetime,
    end: datetime,
):
    feedback_query = queries.query_user_feedback(
        device_id=device_id, start=start, end=end
    )
    feedback_rows = await pool.execute(*feedback_query)

    # throw away feedback that lasted less than 1 minute
    filtered = preprocess.filter_time_between(MIN_FEEDBACK_INTERVAL, feedback_rows)

    # TODO handle start and end of interval properly
    samples = await multi(
        [
            fetch_feedback_sample(pool, session, device_id, feedback)
            for feedback in filtered
        ]
    )

    checked_samples = []
    for sample in samples:
        missing = [
            k for k in FEATURES_STORED_REQUIRED if not data.has_feature(sample, k)
        ]
        if missing:
            log.debug(
                get_log_msg(
                    "missing features {} for sample {}" "".format(missing, sample),
                    device_id,
                )
            )
        else:
            checked_samples.append(sample)

    return checked_samples


async def fetch_feedback_sample(pool, session, device_id, feedback):
    sample = await fetch_non_feedback_features(
        pool, session, device_id, feedback["created_on"]
    )
    sample["feedback"] = feedback["feedback"]
    sample["device_id"] = feedback["device_id"]
    sample["user_id"] = feedback["user_id"]
    sample["type"] = USER_FEEDBACK_TYPE
    sample["timestamp"] = feedback["created_on"]
    return sample


def get_time_features(timestamps, index=None):
    day = functools.partial(time_feature, time_of_day)
    week = functools.partial(time_feature, time_of_week)
    year = functools.partial(time_feature, time_of_year)

    if isinstance(timestamps, Iterable):
        timestamps = pd.DatetimeIndex(timestamps)

    feats = np.hstack([f(timestamps) for f in [day, week, year]])

    if isinstance(timestamps, Iterable):
        return pd.DataFrame(feats, columns=TIME_FEATURES, index=index)
    return dict(zip(TIME_FEATURES, feats))


def time_feature(fun, timestamps):
    if isinstance(timestamps, Iterable):
        t = pd.DatetimeIndex(timestamps)
    else:
        t = pd.Timestamp(timestamps)

    return fun(t)


async def make_all_feedback_samples(
    sample_store, pool, session, ignore_watermark=False, device_id=None
):
    online_data = await pool.execute(*queries.query_feedback_intervals())
    online_data = sorted(online_data, key=lambda x: x["device_id"], reverse=True)
    run_timestamp = datetime.utcnow()
    if device_id is not None:
        device_ids = [device_id]
    else:
        device_ids = [d["device_id"] for d in online_data]

    watermarks = [sample_store.get_watermark(device_id) for device_id in device_ids]
    start_dates = [
        max(d["start"], watermark)
        if watermark is not None and not ignore_watermark
        else d["start"]
        for d, watermark in zip(online_data, watermarks)
    ]
    end_dates = [min(d["end"], run_timestamp) for d in online_data]

    to_fetch = (
        fetch_param
        for fetch_param in zip(device_ids, start_dates, end_dates)
        if has_new_data(*fetch_param)
    )

    new_watermark = run_timestamp - MIN_FEEDBACK_INTERVAL

    semaphore = asyncio.Semaphore(SAMPLE_CONCURRENCY)
    all_samples = await multi(
        fetch_upload_samples(
            pool, session, sample_store, new_watermark, *fetch_param, semaphore
        )
        for fetch_param in to_fetch
    )
    flatten_samples = list(chain.from_iterable(all_samples))

    return flatten_samples


def has_new_data(device_id, start, end):
    fetching = end - start > MIN_FEEDBACK_INTERVAL
    if not fetching:
        log.debug("no new data for {}, skipping".format(device_id))
    return fetching


async def fetch_upload_samples(
    pool, session, sample_store, new_watermark, device_id, start, end, semaphore
):
    log.debug(get_log_msg("fetching samples", device_id, start, end))

    async with semaphore:
        samples = await fetch_feedback_samples(pool, session, device_id, start, end)
    sample_store.set_watermark(device_id, new_watermark)
    for smp in samples:
        sample_store.upsert(smp)
    return samples
