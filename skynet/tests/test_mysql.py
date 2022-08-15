import pytz

import pytest

from ..utils.database import queries


@pytest.fixture
def expected_timezone():
    return [
        "Asia/Hong_Kong",
        "Asia/Baku",
        "Asia/Hong_Kong",
        "Asia/Baku",
        "Asia/Hong_Kong",
    ]


@pytest.fixture
def locations(expected_timezone):
    return ["tz_test_{}".format(i) for i in range(len(expected_timezone))]


@pytest.fixture
def devices(expected_timezone):
    return ["device_test_{}".format(i) for i in range(len(expected_timezone))]


def insert_device_ids(dbcon, device_ids):
    for device_id in device_ids:
        queries.execute(
            dbcon,
            f"INSERT INTO Device (device_id, "
            f"room_name, created_on) VALUES ('{device_id}'"
            f", 'test_room_name', NOW())",
        )


def delete_device_ids(dbcon, device_ids):
    for device_id in device_ids:
        queries.execute(
            dbcon, f"DELETE FROM Device WHERE device_id = " f"'{device_id}'"
        )


def insert_location_ids(dbcon, location_ids, timezone):
    for location_id, tz in zip(location_ids, timezone):
        queries.execute(
            dbcon,
            f"INSERT INTO Location (location_id, timezone, "
            f"name, latitude, longitude) VALUES ("
            f"'{location_id}', '{tz}', 'test_loc_name', "
            f"22.396428, 114.109497)",
        )


def delete_location_ids(dbcon, location_ids):
    for location_id in location_ids:
        queries.execute(
            dbcon, f"DELETE FROM Location WHERE location_id = " f"'{location_id}'"
        )


def insert_location_device_list(dbcon, device_ids, location_ids):
    for device_id, location_id in zip(device_ids, location_ids):
        queries.execute(
            dbcon,
            f"INSERT INTO LocationDeviceList "
            f"(device_id, location_id) VALUES ("
            f"'{device_id}', '{location_id}')",
        )


def delete_location_device_list(dbcon, device_ids):
    for device_id in device_ids:
        queries.execute(
            dbcon, f"DELETE FROM LocationDeviceList" f" WHERE device_id = '{device_id}'"
        )


@pytest.fixture
def db_with_data(db, devices, locations, expected_timezone):
    insert_location_ids(db, locations, expected_timezone)
    insert_device_ids(db, devices)
    insert_location_device_list(db, devices, locations)
    yield db

    delete_location_device_list(db, devices)
    delete_location_ids(db, locations)
    delete_device_ids(db, devices)


@pytest.mark.usefixtures("db_with_data")
@pytest.mark.asyncio
async def test_timezones(pool, expected_timezone, devices):
    for device_id, expected_tz_str in zip(devices, expected_timezone):
        expected_tz = pytz.timezone(expected_tz_str)
        assert await queries.get_time_zone(pool, device_id) == expected_tz
