import time
from collections import defaultdict

import click
import pandas as pd

# pylint:disable=no-name-in-module
from cassandra.cluster import Cluster, Event
from cassandra.io.asyncorereactor import AsyncoreConnection
from cassandra.query import BatchStatement, BatchType

# pylint:enable=no-name-in-module


SENSOR_UPDATE_FREQUENCY = "30s"
if pd.__version__ < "0.18.0":
    raise ImportError("Need pandas>=0.18.0")

AGGREGATED_TABLE = "sensor"

TABLES = {
    "humidity": {
        "table": "sensorhumidity3",
        "columns": "created_on, value AS humidity",
    },
    "temperature": {
        "table": "sensortemperature3",
        "columns": "created_on, value AS temperature",
    },
    "luminosity": {
        "table": "sensorluminosity3",
        "columns": "created_on, full_spectrum, infrared_spectrum, " "overall_lux",
    },
    "pircount": {
        "table": "sensorpircount3",
        "columns": "created_on, value AS pircount",
    },
    "pirload": {
        "table": "sensorpirload3",
        "columns": "created_on, one_min, five_min, fifteen_min",
    },
}

NEW_COLUMNS = [
    ("device_id", "text"),
    ("month", "int"),
    ("created_on", "timestamp"),
    ("temperature", "float"),
    ("temperature_raw", "float"),
    ("temperature_refined", "float"),
    ("humidity", "float"),
    ("humidity_raw", "float"),
    ("humidity_refined", "float"),
    ("full_spectrum", "float"),
    ("infrared_spectrum", "float"),
    ("overall_lux", "float"),
    ("pircount", "float"),
    ("one_min", "float"),
    ("five_min", "float"),
    ("fifteen_min", "float"),
]


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("{} took: {:.3f} sec".format(f.__name__, te - ts))
        return result

    return timed


def create_table_if_not_exists(session, table_name=AGGREGATED_TABLE):
    cql = """
    CREATE TABLE IF NOT EXISTS {table_name} ( {columns},
    PRIMARY KEY ((device_id, month), created_on))
    """.format(
        table_name=table_name,
        columns=", ".join("{} {}".format(*ct) for ct in NEW_COLUMNS),
    )
    return session.execute(cql)


@timeit
def insert(session, params, table_name=AGGREGATED_TABLE):
    cql = "INSERT INTO {table_name} ({columns}) VALUES ({values})" "".format(
        table_name=table_name,
        columns=", ".join(c for c, _ in NEW_COLUMNS),
        values=", ".join("?" for _ in NEW_COLUMNS),
    )
    prepared = session.prepare(cql)
    query_batch_size = 100
    async_batch_size = 100
    query_batches = [
        slice(i * query_batch_size, (i + 1) * query_batch_size)
        for i in range(len(params) // query_batch_size + 1)
    ]

    for i, query_batch in enumerate(query_batches):

        if i % async_batch_size == 0:
            batch_statements = []

        batch_statement = BatchStatement(batch_type=BatchType.UNLOGGED)

        for p in params[query_batch]:
            batch_statement.add(prepared, p)

        batch_statements.append(batch_statement)
        # session.execute(batch_statement)

        if i % async_batch_size == async_batch_size - 1 or i == len(query_batches) - 1:
            futures = [session.execute_async(b) for b in batch_statements]
            handlers = [PagedResultHandler(future) for future in futures]
            for handler in handlers:
                handler.wait()

    def _check():
        # check if all rows in cassandra
        device_id, month = [params[0][k] for k in ["device_id", "month"]]
        res = session.execute(
            "SELECT count(*) FROM {} WHERE device_id = %s AND "
            "month = %s".format(table_name),
            (device_id, month),
        )

        if res[0].count >= len(params):
            return True
        print(
            "NOT ALL PARAMS INSERTED, {} params but {} rows "
            "returned.".format(len(params), res[0].count)
        )
        return False

    n_retries = 3
    for _ in range(n_retries):
        if _check():
            break

    else:
        raise ValueError("NOT ALL PARAMS INSERTED")


def get_insert_params(device_id, df):
    month = df.index.strftime("%Y%m").astype(int)
    values = df.reset_index()
    values["device_id"] = device_id
    values["month"] = month

    dicts = values.to_dict("records")
    # XXX insert missing values
    for params in dicts:
        params["created_on"] = params["created_on"].to_datetime()
        for k, _ in NEW_COLUMNS:
            params[k] = params.get(k)
    return [tuple(d[k] for k, _ in NEW_COLUMNS) for d in dicts]


@timeit
def fetch_device(session, key_params):
    queries = [get_query(**params) for params in key_params]
    results = exec_async(session, queries)
    rows = defaultdict(list)
    for params, res in zip(key_params, results):
        rows[params["stype"]].extend(res)
    return rows


class PagedResultHandler:
    def __init__(self, future):
        self.error = None
        self.finished_event = Event()
        self.future = future
        self._rows = []
        self.future.add_callbacks(callback=self.handle_page, errback=self.handle_error)

    def handle_page(self, rows):
        if rows:
            self._rows.extend(rows)
        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
        else:
            self.finished_event.set()

    def handle_error(self, exc):
        self.error = exc
        self.finished_event.set()

    def wait(self):
        self.finished_event.wait()
        if self.error:
            raise self.error  # pylint:disable=raising-bad-type


def exec_async(session, queries):
    futures = [session.execute_async(*query) for query in queries]
    handlers = [PagedResultHandler(future) for future in futures]
    for handler in handlers:
        handler.wait()
    return [handler._rows for handler in handlers]


def get_query(
    table, columns, device_id, month, **kwargs
):  # pylint:disable=unused-argument
    query = "SELECT {} FROM {} WHERE device_id = %s AND month = %s" "".format(
        columns, table
    )
    return query, (device_id, month)


def get_partition_keys(session):
    keys = []
    for stype, table_meta in TABLES.items():
        res = session.execute(
            "SELECT DISTINCT device_id, month " "FROM {}".format(table_meta["table"]),
            timeout=300.0,
        )
        df = pd.DataFrame(res, columns=["device_id", "month"])
        df["stype"] = stype
        df["table"] = TABLES[stype]["table"]
        df["columns"] = TABLES[stype]["columns"]
        keys.append(df)
    return pd.concat(keys).groupby("device_id")


def aggregate_sensors(sensors):
    dfs = [
        pd.DataFrame(sensors[stype], columns=sensors[stype][0]._fields).set_index(
            "created_on"
        )
        for stype in sorted(sensors)
    ]

    # XXX pandas compat 0.17 vs 0.18
    df = (
        pd.concat(dfs, axis=1)
        .sort_index()
        .resample(SENSOR_UPDATE_FREQUENCY)
        .mean()
        .interpolate()
        .dropna(how="all")
    )

    # print(df.head())
    return df


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=9042)
def main(host, port):
    cluster = Cluster([host], port=port)
    cluster.connection_class = AsyncoreConnection
    session = cluster.connect()
    session.set_keyspace("sensordata")
    device_partitions = get_partition_keys(session)

    create_table_if_not_exists(session=session)

    for device_id, device_keys in device_partitions:

        if device_id.startswith("TEST"):
            continue

        # group by month, to avoid consuming too much memory
        for _, keys in device_keys.groupby("month"):

            sensors = fetch_device(session, keys.to_dict("records"))
            df = aggregate_sensors(sensors)
            params = get_insert_params(device_id, df)
            insert(session=session, params=params)
            print("inserted {} cells".format(len(params)))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
