import os
from datetime import datetime, timedelta
from numbers import Number

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dateutil.parser import parse

from skynet import utils
from skynet.control.util import chunks
from skynet.prediction import climate_model, predict
from skynet.prediction.climate_model import QUANTITIES, QUANTITY_MAP
from skynet.sample import sample
from skynet.utils.async_util import run_sync, multi
from skynet.utils.log_util import get_logger, init_logging
from skynet.utils.storage import get_storage
from skynet.utils.enums import Power
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")
FIG_DIR = "./fig"
plt.rcParams["figure.figsize"] = 16, 10
STATES = [
    {"mode": mode, "temperature": temperature_set, "power": Power.ON}
    for mode in ["heat", "cool", "fan", "auto", "dry", "off"]
    for temperature_set in range(16, 33)
]

COLORMAP_FOR_MODES = {
    "heat": "Reds",
    "cool": "Blues",
    "dry": "Purples",
    "fan": "Oranges",
    "off": "Greys",
    "auto": "Greens",
}


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--device_id",
    "-d",
    default="6409FF383939473443067324",
    help="Device ID for predictions.",
)
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="viewer")
@click.option(
    "--timestamp", default=None, help="make prediction at timestamp YYYY-MM-DD:HH:MM"
)
@click.option(
    "--create_timeseries",
    default=False,
    is_flag=True,
    help="Helps plot the climate prediction graph similar to the one on scalyr",
)
@click.option(
    "--back_days",
    default=1,
    type=int,
    help="make predictions from timestamp minus back_days to timestamp",
)
@click.option(
    "--modes",
    "-m",
    type=str,
    default="cool",
    multiple=True,
    help="the modes for which you want to make timeseries predictions for.",
)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="file")
@click.option(
    "--stride",
    type=int,
    default=2,
    help="controls the step between set temperatures used for predicting states",
)
def main(
    config,
    device_id,
    mysql,
    cassandra,
    timestamp,
    create_timeseries,
    back_days,
    modes,
    storage,
    stride,
):

    init_logging("micro_model")
    cnf = yaml.safe_load(open(config))

    connections = get_connections(config=cnf, mysql=mysql, cassandra=cassandra)

    predictor = get_predictor(cnf, storage)

    timestamp = parse(
        timestamp, fuzzy=True
    ) if timestamp is not None else datetime.utcnow()
    print("sample timestamp {}".format(timestamp.isoformat()))

    if create_timeseries:
        timestamps = pd.date_range(
            timestamp - timedelta(days=back_days), timestamp, freq="15Min"
        ).to_pydatetime()
    else:
        timestamps = [timestamp]

    history_features = run_sync(
        get_history_feature_for_timestamps, device_id, timestamps, connections
    )

    predictions = create_climate_model_predictions_for_history_features(
        history_features, predictor, timestamps
    )

    create_climate_prediction_plots_for_timestamp(
        modes, create_timeseries, history_features, predictions, timestamps, stride
    )

    fname = get_figure_name(device_id, create_timeseries, timestamp)

    plt.savefig(fname)
    log.debug("Saved figure to {}".format(fname))


def get_predictor(cnf, storage):
    model_store = get_storage(storage, **cnf["model_store"], directory="data/models")
    fitted_model = model_store.load(climate_model.ClimateModel.get_storage_key())
    predictor = predict.Predictor(fitted_model)
    return predictor


async def get_history_feature_for_timestamps(device_id, timestamps, connections):
    samples = [
        sample.PredictionSample(device_id, timestamp) for timestamp in timestamps
    ]
    fetched_samples = []
    for sample_chunk in chunks(samples, connections.pool.maxsize):
        fetched_sample = await multi(
            [smp.fetch(connections.pool, connections.session) for smp in sample_chunk]
        )
        fetched_samples.extend(fetched_sample)
    return [smp.get_history_feature_or_none() for smp in fetched_samples]


def create_climate_model_predictions_for_history_features(
    history_features, predictor, timestamps
):
    empty_y_preds = [[None] * len(QUANTITIES)] * len(STATES)
    y_preds = [
        predictor.predict(history_feature, STATES) if history_feature else empty_y_preds
        for history_feature in history_features
    ]
    predictions = []
    prediction = pd.DataFrame(STATES).rename(columns={"temperature": "temperature_set"})

    for y_pred, timestamp in zip(y_preds, timestamps):
        prediction["timestamp"] = timestamp
        for quantity, idx in QUANTITY_MAP.items():
            prediction.loc[:, quantity] = np.array(y_pred)[:, idx]
        predictions.append(prediction.copy())
    return pd.concat(predictions)


def create_climate_prediction_plots_for_timestamp(
    modes, create_timeseries, history_features, predictions, timestamps, stride
):
    if len(history_features) > 1 and create_timeseries:
        create_climate_prediction_timeseries_for_set_temperatures(
            modes, predictions, timestamps, stride
        )
    else:
        create_climate_prediction_graph(history_features, predictions)


def create_climate_prediction_timeseries_for_set_temperatures(
    modes, predictions, timestamps, stride
):

    plt.clf()

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(60, 30))

    ticks = range(0, len(timestamps), 10)

    axes[0].set_title("Temperature Predictions")
    axes[1].set_title("Humidity Predictions")

    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([timestamps[i].isoformat()[:13] for i in ticks])

    for mode in modes:

        df = predictions.loc[predictions["mode"] == mode]

        humidity_df = df[["timestamp", "temperature_set", "humidity"]]
        temperature_df = df[["timestamp", "temperature_set", "temperature"]]

        humidity_df = humidity_df.pivot(
            index="timestamp", columns="temperature_set", values="humidity"
        ).reset_index()
        temperature_df = temperature_df.pivot(
            index="timestamp", columns="temperature_set", values="temperature"
        ).reset_index()

        temperature_df.plot(
            x="timestamp",
            y=[x for x in range(16, 33, stride)],
            colormap=COLORMAP_FOR_MODES[mode],
            ax=axes[0],
        )
        humidity_df.plot(
            x="timestamp",
            y=[x for x in range(16, 33, 2)],
            colormap=COLORMAP_FOR_MODES[mode],
            ax=axes[1],
        )


def create_climate_prediction_graph(history_features, predictions):
    history_feature = history_features[0]

    if history_feature is None:
        print("Cannot plot for the timestamp.")
        exit(0)
    else:
        _, axes = plt.subplots(1, len(QUANTITIES))
        for ax, quantity in zip(axes, QUANTITIES):
            current = history_feature[quantity]
            print("current value {}".format(current))
            current = current if (
                current and np.isfinite(float(current))
            ) else utils.thermo.DEFAULT_TEMPERATURE

            for s in STATES:
                if "temperature" in s:
                    t = s.pop("temperature")
                    if isinstance(t, Number):
                        s["temperature_set"] = t
                    else:
                        s["temperature_set"] = np.nan

            table = predictions.pivot_table(
                values=quantity, columns=["mode"], index="temperature_set"
            )

            print(table)

            table.plot(ax=ax, marker="o", linestyle=":", legend=False)
            ax.set_ylabel(quantity)
            ax.plot(
                [table.index.min(), table.index.max()],
                [current, current],
                "k--",
                lw=2,
                label="current value",
            )
            if quantity == "humidity":
                ax.legend(loc="best")
            ax.grid()


def get_figure_name(device_id, create_timeseries, timestamp):
    try:
        os.mkdir(FIG_DIR)
    except OSError:
        pass
    fname = os.path.join(
        FIG_DIR,
        f"predictions_{'timeseries' if create_timeseries else ''}_"
        f"{device_id}_{timestamp.isoformat().replace(':', '_')}.png",
    )
    return fname


if __name__ == "__main__":
    main()
