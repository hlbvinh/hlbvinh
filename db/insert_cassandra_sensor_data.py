#!/usr/bin/env python
from dateutil.parser import parse

import pandas as pd
import click

from cassandra.cluster import Cluster
from scripts import aggregate_sensors_cassandra


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=9042)
@click.option("--fname", required=True)
@click.option("--keyspace", default="test")
@click.option("--drop", default=False, is_flag=True)
def main(host, port, fname, keyspace, drop):

    cluster = Cluster([host], port)
    session = cluster.connect()
    session.execute(
        "CREATE KEYSPACE IF NOT EXISTS {} WITH replication = "
        "{{'class':'SimpleStrategy', 'replication_factor' : 3}}"
        "".format(keyspace)
    )
    session.set_keyspace(keyspace)

    if drop:
        session.execute(f"DROP TABLE {aggregate_sensors_cassandra.AGGREGATED_TABLE}")

    aggregate_sensors_cassandra.create_table_if_not_exists(session)

    records = pd.read_csv(fname).to_dict("records")
    for row in records:
        row["created_on"] = parse(row["created_on"])

    aggregate_sensors_cassandra.insert(session, records)


if __name__ == "__main__":
    main()
