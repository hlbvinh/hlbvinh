"""User Insights/Analytics.

    TODO:
          [ ] error handling such that individual figures come through
              even if data unavailable for other figures
          [ ] combine sensor data correctly for non-matching timestamps

"""
import operator
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.zmq_actor import RouterActor
from cachetools import TTLCache, keys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVR
from voluptuous import Invalid, Optional, Required, Schema

from skynet.utils.database.cassandra_queries import query_sensor

from ..sample.selection import COMPENSATE_COLUMN
from ..utils import cache_util, misc, thermo, weather
from ..utils.async_util import multi
from ..utils.compensation import compensate_features, compensate_sensors
from ..utils.database import queries
from ..utils.database.cassandra import CassandraSession
from ..utils.enums import Power
from ..utils.log_util import get_logger
from ..utils.sklearn_utils import permutation_feature_importance
from ..utils.status import StatusActor

log = get_logger(__name__)
MIN_NUM_DATA = 10
MAX_NUM_DATA_FACTORS = 200
FEATURES_GROUP = {
    "Metabolism": ["tod_sin", "tod_cos", "pirload"],
    "Indoor Temperature": ["temperature"],
    "Indoor Humidity": ["humidity"],
    "Weather ": ["temperature_out", "humidity_out"],
    "Luminosity": ["luminosity"],
}
FEATURES_TRAIN = list({x for vals in FEATURES_GROUP.values() for x in vals})
COMFORT_MAP_FEATURES = ["temperature", "feedback", "timestamp", COMPENSATE_COLUMN]
TEMPERATURE_FILTER_RANGE = 10.0
HUMIDITY_FILTER_RANGE = 30
SENSOR_INTERVAL = timedelta(days=7)
AGGREGATE_INTERVAL = timedelta(days=28)

lru_cache = TTLCache(maxsize=10000, ttl=300)


def aggregate(states, targets, start, end):
    """Aggreate AC state and control target data

    Raises
    ------
    KeyError:
        records empty or missing 'quantity', 'mode', 'power', 'created_on'
                                 'set_temperature' fields
    """
    first_state = states[0]
    first_target = targets[0]
    first = first_state.copy()
    first.update(first_target)
    first["created_on"] = max(
        first_state["created_on"], first_target["created_on"], start
    )

    records = sorted(
        [first] + states[1:] + targets[1:], key=operator.itemgetter("created_on")
    )

    last_record = records[-1].copy()
    last_record["created_on"] = end
    records.append(last_record)

    # lowercase (This is faster than operation on df below if we have
    #            about 400 or less records on average)
    # records = [{k: v.lower() if k in {'quantity', 'power', 'mode'} else v
    #             for k, v in record.items()} for record in records]

    df = pd.DataFrame.from_dict(records).ffill()

    # lowercase (faster if > 400 records)
    for column in ["quantity", "mode", "power", "origin"]:
        df[column] = df[column].str.lower()

    return df


# TODO: use motor for async mongodb, or async some other way


def fetch_comfort_data(user_sample_store, device_id, user_id):
    return pd.DataFrame(
        user_sample_store.get(
            key={"device_id": device_id, "user_id": user_id}, sort="timestamp"
        )
    )


# wraps async functions to be used with multi


def log_error(user_id, device_id):
    def outer(fun):
        async def wrapper(*args, **kwargs):
            try:
                r = await fun(*args, **kwargs)
            except Exception as e:
                log.error(
                    repr(e),
                    extra={"data": {"device_id": device_id, "user_id": user_id}},
                )
                r = None
            return r

        return wrapper

    return outer


async def fetch_one(pool, query):
    return await pool.execute(*query)


async def fetch_multi(pool, qs):
    return await multi({name: fetch_one(pool, query) for name, query in qs.items()})


async def fetch_location(pool, device_id):
    rows = await fetch_one(pool, queries.query_location_from_device(device_id))
    return rows[0]["location_id"]


async def fetch_states(pool, device_id, start, end):
    rows = await fetch_one(pool, queries.query_current_appliance_from_device(device_id))

    try:
        appliance_id = rows[0]["appliance_id"]
    except IndexError:
        return {"states": tuple(), "targets": tuple()}

    qs = state_queries(device_id, appliance_id, start, end)
    return await fetch_multi(pool, qs)


async def fetch_sensors(pool, session, device_id, location_id, start, end):
    cassandra_qs = cassandra_sensor_queries(device_id, start, end)
    futures = {"sensors": session.execute_async(*cassandra_qs)}

    if location_id is not None:
        weather_query = mysql_weather_query(location_id, start, end)
        futures.update({"weather": fetch_one(pool, weather_query)})

    ret = await multi(futures)

    ret["sensors"] = aggregate_sensor_data(ret["sensors"])

    return ret


def cassandra_sensor_queries(device_id, start, end):
    return query_sensor(device_id, start, end, stypes=["temperature", "humidity"])


def mysql_weather_query(location_id, start, end):
    return queries.query_weather_interval(location_id, start, end)


def aggregate_sensor_data(sensors, hours=4):
    if sensors:
        df = (
            pd.DataFrame(list(sensors))
            .resample(f"{hours}h", on="created_on")
            .mean()
            .dropna(how="all")
            .reset_index()
        )
        return df.rename(columns={"created_on": "timestamp"}).to_dict("records")
    return []


def state_queries(device_id, appliance_id, start, end):
    # to be able to split manual mode into remote and app
    # we also query the origin column
    return {
        "states": queries.query_appliance_states(
            appliance_id,
            start,
            end,
            columns=["created_on", "power", "mode", "temperature", "origin"],
        ),
        "targets": queries.query_control_targets(
            device_id, start, end, columns=["created_on", "quantity"]
        ),
    }


@misc.json_timeit(service="analytics_service", event="compute_comfort_stats")
async def make_comfort_stats(
    redis, pool, user_sample_store, device_id, user_id, location_id, timestamp
):
    rows = await fetch_one(pool, queries.query_weather(location_id, timestamp, N=1))
    current_weather = rows[0] if rows else {}
    X = fetch_comfort_data(user_sample_store, device_id, user_id)
    return await comfort_stats(redis, user_id, device_id, X, current_weather)


async def comfort_stats(redis, user_id, device_id, X, current_weather):
    factors = await cache_util.get_comfort_factors_redis(redis, user_id, device_id)
    if not factors:
        factors = [
            {"type": k, "value": v, "data": len(X)}
            for k, v in sorted(comfort_factors(X).items(), reverse=True)
        ]
        await cache_util.set_comfort_factors_redis(redis, factors, user_id, device_id)

    weather_type = (
        weather.classify_weather(
            current_weather["temperature_out"], current_weather["humidity_out"]
        )
        if current_weather
        else "unknown"
    )
    filtered = comfort_map(X, current_weather) if not X.empty else []
    comfort_data = [{"type": weather_type, "data": len(filtered)}, filtered]

    return {"comfort_factors": factors, "comfort_map": comfort_data}


def comfort_factors(X):
    """
    Take mongo client and in message that have either a user_id or
    user_id and device_id, then fetch samples of requested device_id and/or
    user_id. Computer feature importance then return it as records.

    Run time for 100 samples (fetch data and computation) is ~ 5ms

    Parameters
    ----------
    X: DataFrame
        user feedback data

    Returns
    -------
    feature_importances: records
                         list of dicts that has the format like:
                         [{'time of day': 0.2, 'humidity': 0.2, 'weather': 0.2,
                         'other': 0.2,}, {....}...]
    """
    if len(X) >= MIN_NUM_DATA:
        X = X.tail(MAX_NUM_DATA_FACTORS)
        estimator = make_pipeline(SimpleImputer(), StandardScaler(), SVR(gamma="auto"))
        try:
            imps = permutation_feature_importance(
                estimator,
                X[FEATURES_TRAIN],
                X["feedback"],
                3,
                feature_groups=FEATURES_GROUP,
            )
        except KeyError as exc:
            log.error(exc)
            return {}
        else:
            return imps.to_dict()

    return {}


def comfort_map(
    X,
    current_weather,
    temperature_range=TEMPERATURE_FILTER_RANGE,
    humidity_range=HUMIDITY_FILTER_RANGE,
):

    features_have_weather = "temperature_out" in X and "humidity_out" in X

    if features_have_weather and current_weather:
        filtered = X.loc[
            (
                (X["temperature_out"] - current_weather["temperature_out"]).abs()
                < temperature_range / 2.0
            )
            & (
                (X["humidity_out"] - current_weather["humidity_out"]).abs()
                < humidity_range / 2.0
            ),
            :,
        ]
    else:
        filtered = X
    return compensate_features(filtered)[COMFORT_MAP_FEATURES].to_dict("records")


@misc.json_timeit(service="analytics_service", event="compute_ac_stats")
async def make_ac_stats(pool, device_id, timestamp):
    start = timestamp - AGGREGATE_INTERVAL
    end = timestamp
    query_data = await fetch_states(pool, device_id, start, end)
    return ac_usage_stats(query_data, start, end)


def ac_usage_stats(query_data, start, end):
    try:
        df = aggregate(query_data["states"], query_data["targets"], start, end)
    except IndexError:
        return {"ac_set_points": [], "mode_usage": [], "ac_running_time": []}

    chunks = df_chunks(df, week_intervals(end))
    weekly_stats = [
        power_mode_stats(chunk["data"], chunk["start"], chunk["end"])
        for chunk in chunks
    ]

    return {
        "ac_set_points": ac_set_points(chunks),
        "mode_usage": mode_usage(weekly_stats),
        "ac_running_time": ac_running_time(weekly_stats),
    }


def ac_set_points(chunks):
    return [
        {
            "from": chunk["start"],
            "to": chunk["end"],
            "value": average_set_temperature(chunk["data"]),
        }
        for chunk in chunks
        if average_set_temperature(chunk["data"]) is not None
    ]


def average_set_temperature(df):
    on = df["power"] == Power.ON

    # only use modes with temperature settings
    temperature_mode = df["mode"].isin(["cool", "heat", "dry", "auto"])

    # get rid of small -2, -1, ..., +2 values
    celsius_or_fahrenheit = df["temperature"] > 8

    df = df.loc[on & temperature_mode & celsius_or_fahrenheit].dropna()

    # If we have no rows left it doesn't make sense to try to compute an
    # average set temperature.
    if df.empty:
        return None

    # convert Fahrenheit temperatures to Celsius
    df["temperature"] = thermo.fix_temperatures(df["temperature"].values)
    seconds = df["duration"].astype("timedelta64[s]")
    average_set_temp = df["temperature"].dot(seconds) / seconds.sum()
    return average_set_temp if np.isfinite(average_set_temp) else None


def mode_usage(weekly_power_mode_stats):
    keys_ = [
        "comfort",
        "temperature",
        "away_on",
        "away_off",
        "remote_on",
        "remote_off",
        "manual_on",
        "manual_off",
        "api_on",
        "api_off",
        "on",
        "off",
        "manual",
    ]
    return [
        {
            "from": stats["start"],
            "to": stats["end"],
            "values": {k: stats[k] for k in keys_},
        }
        for stats in weekly_power_mode_stats
    ]


def ac_running_time(weekly_power_mode_stats):
    key_map = {"start": "from", "end": "to", "on": "value"}
    return [
        {m: stats[k] for k, m in key_map.items()} for stats in weekly_power_mode_stats
    ]


def power_mode_stats(df, start, end):
    duration = df["duration"] = df["created_on"].shift(-1) - df["created_on"]
    on = df["power"] == Power.ON
    off = ~on
    in_away_mode = df["quantity"].str.startswith("away")
    is_manual = df["quantity"] == "manual"
    manual_from_remote = df["origin"] == "reverse"
    manual_from_app = df["origin"] == "irdeployment"
    from_api = df["origin"] == "openapi"

    rows = {
        # comfort mode do not necessarily trigger new appliance state
        # comfort mode do trigger control target
        "comfort": on & df["quantity"].isin(["comfort", "climate"]),
        # temperature mode do not necessarily trigger new appliance state
        # temperature mode do trigger control target
        "temperature": on & (df["quantity"] == "temperature"),
        # away mode do not necessarily trigger new appliance state
        # away mode do trigger control target
        "away_on": on & in_away_mode,
        "away_off": off & in_away_mode,
        # remote do trigger new appliance state
        # temperature mode off control target may not indicate
        # that it is from remote
        "remote_on": on & manual_from_remote & is_manual,
        "remote_off": off & manual_from_remote & (~in_away_mode),
        # manual do trigger new appliance state
        # manual mode off control target may not indicate
        # that it is from remote
        "manual_on": on & manual_from_app & is_manual,
        "manual_off": off & manual_from_app & (~in_away_mode),
        # api do trigger new appliance state
        # api off control target may not indicate
        # that it is from remote
        "api_on": on & from_api & is_manual,
        "api_off": off & from_api & (~in_away_mode),
        # preserve previous labels
        "on": on,
        "off": off & (~in_away_mode),
        "manual": on & (df["quantity"] == "manual"),
    }

    times = {k: duration.loc[v].sum() for k, v in rows.items()}
    times = {
        k: v.to_pytimedelta().total_seconds() / 3600.0 if pd.notnull(v) else 0.0
        for k, v in times.items()
    }
    times["start"] = start
    times["end"] = end
    return times


@misc.json_timeit(service="analytics_service", event="make_sensors_data")
async def make_sensors(pool, session, device_id, location_id, timestamp):
    query_data = await fetch_sensors(
        pool,
        session,
        device_id,
        location_id,
        start=timestamp - SENSOR_INTERVAL,
        end=timestamp,
    )
    return temperature_and_humidity(
        query_data.pop("sensors"), query_data.pop("weather", [])
    )


def temperature_and_humidity(sensors, weather):
    return {
        "temperature_and_humidity": {
            "indoor": compensate_sensors(sensors),
            "outdoor": weather,
        }
    }


@misc.json_timeit(service="analytics_service", event="compute_co2_stats")
async def make_co2_stats(
    session: CassandraSession, device_id: str, timestamp: datetime
) -> Dict[str, List[Dict]]:
    sensors = await fetch_co2(
        session, device_id, start=timestamp - SENSOR_INTERVAL, end=timestamp
    )
    return {"co2_daily_aggregate": co2_stats(sensors, timestamp)}


async def fetch_co2(
    session: CassandraSession, device_id: str, start: datetime, end: datetime
) -> List:
    return await session.execute_async(
        *query_sensor(device_id, start, end, stypes=["co2"])
    )


def co2_stats(sensors: List, end: datetime) -> List[Dict]:
    return [
        {
            "value": chunk["data"]["co2"].mean(),
            "start": chunk["start"],
            "end": chunk["end"],
        }
        for chunk in df_chunks(pd.DataFrame(sensors), day_intervals(end))
    ]


def day_intervals(end):
    return [
        {"start": end - timedelta(days=i + 1), "end": end - timedelta(days=i)}
        for i in reversed(range(7))
    ]


def week_intervals(end):
    return [
        {"start": end - timedelta(days=7 * (i + 1)), "end": end - timedelta(days=7 * i)}
        for i in [3, 2, 1, 0]
    ]


def df_chunks(df, intervals):
    chunks = []
    for interval in intervals:
        if interval["end"] > df["created_on"].iat[0]:
            try:
                chunk = {"data": trim_df(df, interval["start"], interval["end"])}
                chunk.update(interval)
                chunks.append(chunk)
            except ValueError:
                pass
    return chunks


def trim_df(df, start, end):
    if start < df["created_on"].iat[0]:
        raise ValueError("Can't trim to earlier than first timestamp in df")
    start_index = df["created_on"].searchsorted(start, side="left") - 1
    end_index = df["created_on"].searchsorted(end, side="left") - 1
    df = df.loc[start_index:end_index].reset_index(drop=True)
    df = df.append(df.iloc[-1], ignore_index=True)
    df["created_on"].iat[0] = start
    df["created_on"].iat[len(df) - 1] = end
    return df


class UserAnalyticsActor(Actor):

    schema = Schema(
        {
            Required("user_id"): str,
            Required("device_id"): str,
            Optional("timestamp"): datetime,
        }
    )

    def __init__(self, user_sample_store, redis, pool, session):
        self.user_sample_store = user_sample_store
        self.redis = redis
        self.pool = pool
        self.session = session
        super().__init__(log)

    async def _process(self, user_id, device_id, timestamp=None):
        key = keys.hashkey(user_id, device_id, timestamp)
        if key in lru_cache:
            return lru_cache[key]

        try:
            location_id = await fetch_location(self.pool, device_id)
        except (TypeError, KeyError, IndexError):
            log.error("{} no location_id".format(device_id))
            location_id = None

        timestamp = (timestamp or datetime.utcnow()).replace(tzinfo=None)

        wrap_info = log_error(user_id, device_id)
        futures = {
            "ac_stats": wrap_info(make_ac_stats)(self.pool, device_id, timestamp),
            "sensors": wrap_info(make_sensors)(
                self.pool, self.session, device_id, location_id, timestamp
            ),
            "comfort_stats": wrap_info(make_comfort_stats)(
                self.redis,
                self.pool,
                self.user_sample_store,
                device_id,
                user_id,
                location_id,
                timestamp,
            ),
            "co2_stats": wrap_info(make_co2_stats)(self.session, device_id, timestamp),
        }

        data = {
            "comfort_map": [],
            "comfort_factors": [],
            "temperature_and_humidity": {"indoor": [], "outdoor": []},
            "ac_running_time": [],
            "ac_set_points": [],
            "mode_usage": [],
            "co2_daily_aggregate": [],
        }
        status = 200
        ret = await multi(futures)
        for part, part_data in ret.items():
            if not part_data:
                log.error(f"could not get data for {part}")
            else:
                data.update(part_data)
        result = data, status
        lru_cache[key] = result
        return result

    @misc.json_timeit(service="analytics_service", event="analytics")
    async def do_tell(self, req):
        params = req.params
        log.info("processing {}".format(params))
        try:
            try:
                self.schema(params)
            except Invalid as exc:
                log.error("invalid params {}".format(params))
                log.error(exc)
                data = []
                status = 400
            else:
                data, status = await self._process(
                    params["user_id"], params["device_id"], params.get("timestamp")
                )
        except Exception as exc:
            log.exception(exc)
            data = {}
            status = 400
        reply = {
            "method": "UserAnalyticsActor",
            "data": data,
            "status": status,
            "context_id": req.context_id,
            "message_id": req.message_id,
        }
        msger = self.context.find_actor("msger")
        msger.tell(reply)


class AnalyticsService(MicroService):
    def __init__(self, ip, port, redis, pool, session, user_sample_store):
        self.ip = ip
        self.port = port
        self.redis = redis
        self.pool = pool
        self.session = session
        self.user_sample_store = user_sample_store
        super().__init__(None, log)

    def setup_resources(self):
        log.debug("Setting up resources")
        self.actor_ctx = ActorContext(log)
        self.actor_ctx.add_actor(
            "msger", RouterActor(ip=self.ip, port=self.port, log=log)
        )
        self.actor_ctx.add_actor(
            "UserAnalyticsActor",
            UserAnalyticsActor(
                user_sample_store=self.user_sample_store,
                redis=self.redis,
                pool=self.pool,
                session=self.session,
            ),
        )
        self.actor_ctx.add_actor("StatusActor", StatusActor())
