from datetime import datetime, timedelta

import pytest

from skynet.utils.database import queries

from ...disaggregation.dummy import (
    ac_runtime,
    date_range,
    device_runtime,
    disaggregation,
    dummy_disaggregation,
)
from ...utils.enums import Power
from ...utils.types import Connections


@pytest.mark.parametrize(
    "start,end,states,result",
    [
        pytest.param(
            datetime(2019, 9, 27),
            datetime(2019, 9, 28),
            [],
            None,
            id="no states no computation",
        ),
        pytest.param(
            datetime(2019, 9, 27),
            datetime(2019, 9, 29),
            [
                dict(created_on=datetime(2019, 9, 27, 6), power=Power.ON),
                dict(created_on=datetime(2019, 9, 27, 20), power=Power.OFF),
            ],
            timedelta(hours=20 - 6),
        ),
        pytest.param(
            datetime(2019, 9, 27),
            datetime(2019, 9, 29),
            [dict(created_on=datetime(2019, 9, 25, 6), power=Power.ON)],
            timedelta(days=2),
        ),
        pytest.param(
            datetime(2019, 9, 27),
            datetime(2019, 9, 29),
            [dict(created_on=datetime(2019, 9, 25, 6), power=Power.OFF)],
            timedelta(),
        ),
    ],
)
def test_ac_runtime(start, end, states, result):
    assert ac_runtime(start, end, states) == result


def test_date_range():
    assert list(date_range("2019-09-27", "2019-09-29")) == [
        ("2019-09-27", (datetime(2019, 9, 26, 16), datetime(2019, 9, 27, 16))),
        ("2019-09-28", (datetime(2019, 9, 27, 16), datetime(2019, 9, 28, 16))),
        ("2019-09-29", (datetime(2019, 9, 28, 16), datetime(2019, 9, 29, 16))),
    ]


@pytest.fixture
async def location_id(device_id, pool):
    rows = await pool.execute(*queries.query_location_from_device(device_id))
    return rows[0]["location_id"]


@pytest.fixture
def dates(device_id, device_intervals):
    for interval in device_intervals:
        if interval["device_id"] == device_id:
            return interval["start"], interval["end"]
    raise ValueError


@pytest.mark.asyncio
async def test_device_runtime(pool, device_id, dates):
    start, end = dates
    assert isinstance(await device_runtime(pool, device_id, start, end), timedelta)


@pytest.fixture
def connections(rediscon, pool, cassandra_session):
    return Connections(redis=rediscon, pool=pool, session=cassandra_session)


@pytest.mark.asyncio
async def test_dummy_dissaggregation(connections, device_id, location_id, dates):
    start, end = dates[0].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d")
    result, _ = await dummy_disaggregation(connections, location_id, start, end)
    assert {start, end}.issubset(d["date"] for d in result[device_id])


def test_regression_disaggregation():
    assert disaggregation(timedelta()) is not None
    assert disaggregation(timedelta()) == 0
