import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
from cachetools import TTLCache, keys

from skynet.utils import parse

from ..utils.async_util import multi
from ..utils.database.dbconnection import Pool
from ..utils.database.queries import (
    query_appliance_states_from_device,
    query_devices_from_location_id,
    query_timezone,
)
from ..utils.enums import Power
from ..utils.log_util import get_logger
from ..utils.types import ApplianceState

log = get_logger(__name__)

AC_KW_H = 0.960

# TODO: replace by a redis cache to store the last 30 days of disaggregation for each user
ttl_cache = TTLCache(maxsize=10000, ttl=3600 * 24 * 30)


async def dummy_disaggregation(connections, location_id: str, start: str, end: str):
    devices, timezone = await multi(
        [
            connections.pool.execute(*query_devices_from_location_id(location_id)),
            connections.pool.execute(*query_timezone(location_id)),
        ]
    )
    timezone = timezone[0]["timezone"]
    data: Dict[str, List[Dict[str, Any]]] = {
        device["device_id"]: [] for device in devices
    }

    for date, (date_start, date_end) in date_range(start, end, timezone):
        runtimes = await multi(
            {
                device_id: device_runtime(
                    connections.pool, device_id, date_start, date_end
                )
                for device_id in data
            }
        )
        for device_id, runtime in runtimes.items():
            value = disaggregation(runtime)
            if value is not None:
                data[device_id].append({"date": date, "value": value})

    status = 200
    return data, status


def date_range(start: str, end: str, timezone: Optional[str] = None):
    end_of_day = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    dates = pd.date_range(start, end_of_day, freq="D", tz=timezone or "Asia/Singapore")
    utc_dates = dates.tz_convert(pytz.utc).tz_localize(None).to_pydatetime()
    yield from zip(dates.strftime("%Y-%m-%d"), zip(utc_dates, utc_dates[1:]))


def disaggregation(ac_runtime: Optional[timedelta]) -> Optional[float]:
    if ac_runtime is not None:
        return ac_runtime.total_seconds() / 3600 * AC_KW_H
    return None


async def device_runtime(
    pool: Pool, device_id: str, start: datetime, end: datetime
) -> Optional[timedelta]:
    key = keys.hashkey(device_id, start, end)
    if key in ttl_cache:
        return ttl_cache[key]
    try:
        states = await pool.execute(
            *query_appliance_states_from_device(device_id, start, end)
        )
    except asyncio.TimeoutError as exc:
        log.info(exc)
        return None

    ttl_cache[key] = ac_runtime(start, end, parse.lower_dicts(states))
    return ttl_cache[key]


def ac_runtime(
    start: datetime, end: datetime, states: List[ApplianceState]
) -> Optional[timedelta]:
    if not states:
        return None
    df = (
        pd.DataFrame(states)
        .append(pd.DataFrame({"created_on": [start, end]}), sort=False)
        .sort_values(by=["created_on"])
        .fillna(method="ffill")
    )
    df["state_duration"] = (
        df["created_on"].shift(-1) - df["created_on"]
    ).dt.total_seconds()
    df = df.dropna(subset=["state_duration"])
    df = df[(df.created_on >= start) & (df.created_on < end)]

    return timedelta(seconds=sum(df[df["power"] == Power.ON]["state_duration"]))
