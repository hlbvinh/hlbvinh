from datetime import timedelta

from ..utils.async_util import multi
from ..utils.database import cassandra_queries, queries
from .selection import SENSORS_INTERVAL, TIMESERIES_INTERVAL

STATE_IVAL = timedelta(minutes=20)
SENSOR_IVAL = max(SENSORS_INTERVAL, TIMESERIES_INTERVAL)
WEATHER_INTERVAL = timedelta(days=7)
WEATHER_FORECAST_INTERVAL = timedelta(hours=1)
WEATHER_COLUMNS = [
    "timestamp",
    "temperature as temperature_out",
    "100.0 * humidity as humidity_out",
    "apparent_temperature",
    "cloud_cover",
]


def missing_keys(d):
    """Return True if all values of dictionary are True."""
    return [k for k in d if not d[k]]


def check_query(result):
    missing = missing_keys(result)
    if missing:
        raise ValueError("empty query results {}".format(missing))


async def query_target(session, device_id, timestamp, end):
    return {
        "sensors": await session.execute_async(
            *cassandra_queries.query_sensor(
                device_id, start=timestamp, end=end, stypes=["humidity", "temperature"]
            )
        )
    }


async def query_feature(pool, session, device_id, timestamp, end):

    data = {
        "states": pool.execute(
            *queries.query_appliance_states_from_device(
                device_id, timestamp - STATE_IVAL, end
            )
        ),
        "weather": queries.get_weather_api_data_from_device(
            pool,
            device_id,
            timestamp - WEATHER_INTERVAL,
            end + WEATHER_FORECAST_INTERVAL,
            columns=WEATHER_COLUMNS,
        ),
        "sensors": session.execute_async(
            *cassandra_queries.query_sensor(device_id, timestamp - SENSOR_IVAL, end)
        ),
    }

    return await multi(data)
