"""Update timezone."""
import time
from contextlib import closing
from datetime import datetime

import click
import requests
import yaml

from skynet.utils.database import queries
from skynet.utils.database.dbconnection import DBConnection


def get_locations(dbcon, devices):
    return [
        queries.get_location_from_device(
            dbcon,
            device_id=device["device_id"],
            columns=["location_id", "latitude", "longitude"],
        )
        for device in devices
    ]


def get_device_no_timezone(dbcon):
    q = """
    SELECT
        device_id
    FROM
        Device
    WHERE
        timezone IS NULL
    AND
        device_id NOT LIKE 'TEST%'
    AND
        LENGTH(device_id) = 24
    """
    return queries.execute(dbcon, q)


@queries.commit_wrap
def update_device_timezone(dbcon, device_id, timezone):
    q = """
    UPDATE
        Device
    SET
        timezone = %s
    WHERE
        device_id = %s
    """
    with closing(dbcon.cursor()) as cursor:
        return cursor.execute(q, (timezone, device_id))  # pylint: disable=no-member


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="local")
def main(config, mysql):

    # make a connection to the MySQL database
    config = yaml.safe_load(open(config))
    db = DBConnection(**config[mysql])

    url = "https://maps.googleapis.com/maps/api/timezone/json"

    devices = get_device_no_timezone(db)
    locations = get_locations(db, devices)
    d = datetime.utcnow()
    t = time.mktime(d.timetuple())
    for device, location in zip(devices, locations):
        if location:
            loc = [location[0]["latitude"], location[0]["longitude"]]
            params = {"location": "{},{}".format(*loc), "timestamp": t}
            resp = requests.get(url, params=params)
            json = resp.json()
            timezone = json["timeZoneId"]
            time.sleep(0.2)
            print("updating", device["device_id"], timezone)

            update_device_timezone(db, device["device_id"], timezone)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
