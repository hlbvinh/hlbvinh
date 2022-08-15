import pickle
import sys
from datetime import datetime, timedelta

import click
import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta
from joblib import Memory

from skynet.prediction import climate_model
from skynet.sample import climate_sample_store
from skynet.utils.async_util import run_sync
from skynet.utils.database import queries
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")

N_MONTHS = 3

STORAGE_DIRECTORY = "data/climate"
BASELINE_FILENAME = "baseline_df"
SENESOR_SAMPLES_FILENAME = "sensor_samples_df"

maybe_cache = Memory(cachedir="plotly_cache", verbose=2).cache


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="viewer")
@click.option("--mongo", default="test")
@click.option("--storage_directory", default=STORAGE_DIRECTORY)
def main(config, mysql, mongo, storage_directory):

    with open(config) as filehandle:
        cnf = yaml.safe_load(filehandle)

    init_logging_from_config("plotly", cnf=cnf, log_directory="log")

    connections = get_connections(cnf, mysql=mysql, mongo=mongo)

    sample_store = climate_sample_store.ClimateSampleStore(connections.mongo)

    device_appliance_df = pd.DataFrame(
        run_sync(connections.pool.execute, *queries.query_device_appliance_list())
    )

    device_appliance_df = device_appliance_df.drop_duplicates(
        subset=["device_id"], keep=False
    )

    dfs = get_all_dataframes(sample_store, N_MONTHS)

    subsitute_appliance_id_with_device_id(dfs, device_appliance_df)

    add_colors_for_each_target(dfs)

    add_hover_text(dfs)

    process_y_long(dfs)

    create_sensor_samples_df(dfs)

    to_plot_dfs = {
        "sensor_samples_df": dfs["sensor_samples_df"].sort_values(
            ["sample_id", "t_timestamp"]
        )
    }

    del dfs

    to_plot_dfs["baseline_df"] = to_plot_dfs["sensor_samples_df"][
        [
            "sample_id",
            "mode",
            "temperature",
            "t_temperature",
            "t_humidex",
            "t_humidity",
            "t_timestamp",
            "temperature_out",
            "temperature_set",
        ]
    ]

    adjust_targets_for_baseline_df(to_plot_dfs)

    save_as_pickled_object(
        to_plot_dfs["sensor_samples_df"],
        f"{storage_directory}/{SENESOR_SAMPLES_FILENAME}.pkl",
    )
    save_as_pickled_object(
        to_plot_dfs["baseline_df"], f"{storage_directory}/{BASELINE_FILENAME}.pkl"
    )

    print("done")


@maybe_cache
def get_all_dataframes(sample_store, n_months):

    fetching_timestamp = datetime.utcnow() - relativedelta(months=n_months)

    log.info(f"Fetching range from {fetching_timestamp} to {datetime.utcnow()}")

    feats, targs = climate_sample_store.get_climate_samples(
        sample_store, key={"timestamp": {"$gt": fetching_timestamp}}, limit=0
    )
    log.info(f"fetched {len(feats)} samples")

    X, y = climate_model.make_static_climate_dataset(feats, targs)

    y = y[climate_model.QUANTITIES]
    X = climate_model.prepare_dataset(X)

    X = X[
        [
            "appliance_id",
            "humidex",
            "humidity",
            "temperature",
            "mode",
            "temperature_out",
            "temperature_set",
            "timestamp",
        ]
    ].reset_index()
    targs = targs.loc[y.index]
    targs = targs[["humidex", "humidity", "temperature", "timestamp"]]

    X = minimize_the_size_of_each_column(X)
    y = minimize_the_size_of_each_column(y)
    targs = minimize_the_size_of_each_column(targs)

    return {"X": X, "y": y, "y_long": targs}


def minimize_the_size_of_each_column(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in df.columns:
        if df.dtypes[column_name] == "int64":
            df[column_name] = df[column_name].astype(np.uint32)
        elif df.dtypes[column_name] == "float64":
            df[column_name] = df[column_name].astype(np.float16)
    return df


def subsitute_appliance_id_with_device_id(dfs, device_appliance_rows):
    print(f"Initially, len of X: {len(dfs['X'])}")
    # some device_id which don't have one-on-one mapping with appliance_id are dropped
    dfs["X"] = pd.merge(dfs["X"], device_appliance_rows, on="appliance_id", how="inner")
    dfs["X"] = dfs["X"].drop("appliance_id", axis=1)
    print(f"After subsituting appliance_id for device_id, len of X: {len(dfs['X'])}")


def add_colors_for_each_target(dfs):
    dfs["X"]["humidex_color"] = "#ff7f0e"
    dfs["X"]["humidity_color"] = "#d62728"
    dfs["X"]["temperature_color"] = "#1f77b4"


def add_hover_text(dfs):
    dfs["X"]["text"] = (
        "set_temp: "
        + dfs["X"]["temperature_set"].map(str)
        + ", mode: "
        + dfs["X"]["mode"]
        + ", temp_out: "
        + dfs["X"]["temperature_out"].map(str)
    )


def process_y_long(dfs):
    dfs["y_long"]["shape"] = "circle"
    dfs["y_long"].rename(
        columns={
            "humidex": "t_humidex",
            "temperature": "t_temperature",
            "humidity": "t_humidity",
            "timestamp": "t_timestamp",
        },
        inplace=True,
    )
    dfs["y_long"].reset_index(inplace=True)


def create_sensor_samples_df(dfs):
    dfs["long_targets_and_features"] = pd.merge(
        dfs["y_long"], dfs["X"], on="sample_id", how="inner"
    )
    create_x_subset(dfs)
    dfs["features_and_features"] = pd.merge(
        dfs["X_subset"], dfs["X"], on="sample_id", how="inner"
    )
    process_y(dfs)
    dfs["targets_and_features"] = pd.merge(
        dfs["y"], dfs["X"], on="sample_id", how="inner"
    )
    adjust_targets_for_targets_and_features(dfs)
    dfs["sensor_samples_df"] = dfs["long_targets_and_features"].append(
        dfs["features_and_features"], ignore_index=True
    )
    dfs["sensor_samples_df"] = dfs["sensor_samples_df"].append(
        dfs["targets_and_features"], ignore_index=True
    )


def create_x_subset(dfs):
    dfs["X_subset"] = dfs["X"][
        ["humidex", "humidity", "temperature", "timestamp", "sample_id"]
    ]
    dfs["X_subset"]["shape"] = "square"
    dfs["X_subset"].rename(
        columns={
            "humidex": "t_humidex",
            "temperature": "t_temperature",
            "humidity": "t_humidity",
            "timestamp": "t_timestamp",
        },
        inplace=True,
    )


def process_y(dfs):
    dfs["y"].reset_index(inplace=True)
    dfs["y"]["shape"] = "cross"
    dfs["y"]["t_timestamp"] = timedelta(hours=3)
    dfs["y"].rename(
        columns={
            "humidex": "t_humidex",
            "humidity": "t_humidity",
            "temperature": "t_temperature",
        },
        inplace=True,
    )


def adjust_targets_for_targets_and_features(dfs):
    dfs["targets_and_features"]["t_temperature"] += dfs["targets_and_features"][
        "temperature"
    ]
    dfs["targets_and_features"]["t_humidex"] += dfs["targets_and_features"]["humidex"]
    dfs["targets_and_features"]["t_humidity"] += dfs["targets_and_features"]["humidity"]
    dfs["targets_and_features"]["t_timestamp"] += dfs["targets_and_features"][
        "timestamp"
    ]


def adjust_targets_for_baseline_df(to_plot_dfs):
    to_plot_dfs["baseline_df"]["t_temperature"] -= to_plot_dfs["sensor_samples_df"][
        "temperature"
    ]
    to_plot_dfs["baseline_df"]["t_humidex"] -= to_plot_dfs["sensor_samples_df"][
        "humidex"
    ]
    to_plot_dfs["baseline_df"]["t_humidity"] -= to_plot_dfs["sensor_samples_df"][
        "humidity"
    ]
    to_plot_dfs["baseline_df"]["t_timestamp"] -= to_plot_dfs["sensor_samples_df"][
        "timestamp"
    ]


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, "wb") as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx : idx + max_bytes])


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
