"""Update default locations set by iOS setup.

Check ./scripts/parse_device_logs.sh for a script to get the logs.
"""

import pprint

import click
import yaml
from requests_futures.sessions import FuturesSession, ThreadPoolExecutor

from skynet.utils import data
from skynet.utils.database import dbconnection, queries

# API = 'https://freegeoip.net/json/'
# docker run -p8899:8080 --restart=always -d fiorix/freegeoip
API = "http://localhost:8899/json/"


def parse(sess, resp):  # pylint: disable=unused-argument
    resp.data = resp.json()


def get_locations(dbcon, lat, lon):
    return list(queries.get_device_from_location(dbcon, latitude=lat, longitude=lon))


def get_device_ips(dbcon):
    sql = """
    SELECT
        D.device_id, D.ip, D.created_on
    FROM
        DeviceConnection D
    JOIN (
        SELECT
            max(row_id) as row_id
        FROM
            DeviceConnection
        WHERE
            device_id NOT LIKE "TEST%%"
        AND
            ip IS NOT NULL
        GROUP BY
            device_id
    ) AS M
    USING
        (row_id)
    """
    return queries.execute(dbcon, sql, tuple())


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="local")
@click.option(
    "--update", default=False, is_flag=True, help="Actually execute the updates."
)
def main(config, mysql, update):

    cnf = yaml.safe_load(open(config))
    db_cnf = cnf[mysql]

    dbcon = dbconnection.DBConnection(**db_cnf)

    locations = get_locations(dbcon, lat=22, lon=114) + get_locations(
        dbcon, lat=0, lon=0
    ) + get_locations(
        dbcon, lat=9999, lon=9999
    )

    locations = data.group_by("location_id", locations)

    print("#### DEFAULT LOCATIONS #####")
    pprint.pprint(locations)
    print()

    device_ips = data.group_by("device_id", get_device_ips(dbcon), keep_key=True)
    pprint.pprint(device_ips)

    print("Fetched IPs of {} devices.".format(len(device_ips)))

    session = FuturesSession(executor=ThreadPoolExecutor(max_workers=1))

    futures = {}
    for location_id, devices in locations.items():
        location_devices = [
            device_ips[d["device_id"]][0]
            for d in devices
            if d["device_id"] in device_ips
        ]
        if location_devices:
            device = sorted(location_devices, key=lambda x: x["created_on"])[-1]
            device_id = device["device_id"]
            if device_id in device_ips:
                ip = device_ips[device_id][0]["ip"]
                print(location_id, ip)
                device_ids = tuple(d["device_id"] for d in devices)
                futures[(location_id, device_ids)] = session.get(
                    API + ip, background_callback=parse
                )

    print("Looked up {} devices locations by IP.".format(len(futures)))

    for (location_id, device_ids), future in futures.items():
        result_data = future.result().data
        print(
            "would update",
            location_id,
            device_ids,
            result_data["latitude"],
            result_data["longitude"],
        )
        if update:
            print(
                "updating",
                location_id,
                result_data["latitude"],
                result_data["longitude"],
            )
            queries.update_device_location(
                dbcon, location_id, result_data["latitude"], result_data["longitude"]
            )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
