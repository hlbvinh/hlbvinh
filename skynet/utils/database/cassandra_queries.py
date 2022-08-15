import itertools
from datetime import datetime
from typing import Optional

import pandas as pd

# pylint: disable=no-name-in-module
from cassandra.query import SimpleStatement, ValueSequence

# pylint: enable=no-name-in-module

from . import result_format
from .. import config, thermo
from ..compensation import compensate_sensors
from .cassandra import CassandraSession

STYPES = ["temperature", "humidity", "luminosity", "pircount"]


def _columns(stypes, with_timestamp=True):
    columns = ["created_on"] if with_timestamp else []
    columns.extend(
        itertools.chain(*[config.CASSANDRA_SENSOR_FIELDS[s] for s in stypes])
    )
    return columns


def _months(start, end):
    start_month = int(start.strftime("%Y%m"))
    end_month = int(end.strftime("%Y%m"))
    return ValueSequence(list(range(start_month, end_month + 1)))


def query_sensor(device_id, start, end, stypes=STYPES, columns=None):
    columns = columns or _columns(stypes)
    month = _months(start, end)
    q = """
    SELECT {columns} FROM sensor
    WHERE
        device_id = %(device_id)s
    AND
        month IN %(month)s
    AND
        created_on > %(start)s
    AND
        created_on <= %(end)s
    """.format(
        columns=", ".join(c for c in columns)
    )
    return q, {"device_id": device_id, "month": month, "start": start, "end": end}


def query_sensor_limit(device_id, timestamp, N=60, stypes=STYPES):
    columns = _columns(stypes, with_timestamp=True)
    month = _months(timestamp, timestamp)[0]
    q = """
    SELECT {columns} FROM sensor
    WHERE
        device_id = %(device_id)s
    AND
        month = %(month)s
    AND
        created_on <= %(timestamp)s
    ORDER BY
        created_on
    DESC
        LIMIT %(N)s
    """.format(
        columns=", ".join(c for c in columns)
    )
    return (
        SimpleStatement(q, fetch_size=None),  # disable paging with fetch_size=None
        {"device_id": device_id, "month": month, "timestamp": timestamp, "N": N},
    )


async def query_feedback_humidex(
    session: CassandraSession, device_id: str, created_on: datetime, N: int = 1
) -> Optional[float]:
    rows = await session.execute_async(
        *query_sensor_limit(
            device_id, created_on, N=N, stypes=["temperature", "humidity"]
        )
    )
    if rows:
        sensors: pd.DataFrame = compensate_sensors(
            result_format.format_sensors_cassandra(rows)
        )
        return thermo.humidex(sensors["temperature"], sensors["humidity"])

    return None
