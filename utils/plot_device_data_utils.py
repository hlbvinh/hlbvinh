import os
from datetime import datetime, timedelta, tzinfo
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from pandas import DataFrame, DatetimeIndex

from skynet import user
from skynet.prediction import mode_model
from skynet.sample import sample
from skynet.user import comfort_model
from skynet.user import sample as user_sample
from skynet.utils import analyze, preprocess, thermo
from skynet.utils.async_util import multi
from skynet.utils.compensation import compensate_sensors
from skynet.utils.config import MAX_COMFORT, MIN_COMFORT
from skynet.utils.database import cassandra_queries, queries
from skynet.utils.database.cassandra import CassandraSession
from skynet.utils.database.dbconnection import DBConnection, Pool
from skynet.utils.enums import Power
from skynet.utils.log_util import get_logger
from skynet.utils.parse import lower_dict
from skynet.utils.storage import get_storage

plt.rcParams["figure.figsize"] = 19, 12
plt.rcParams["lines.linewidth"] = 2
sns.set_style("whitegrid")
# sns.set_style({'font.family': ['Lato']})

log = get_logger("skynet")
START = (datetime.utcnow() - timedelta(hours=72)).isoformat()
END = datetime.utcnow().isoformat()
NPOINTS = 500
SMOOTH_POINTS = 1
STATE_LW = 4
STATE_ALPHA = 0.5
COMFORT_CMAP = "RdBu_r"
COLS = {
    "heat": "#FA9931",
    "cool": "#07C1FF",
    "fan": "#666666",
    "temperature": "#FF5353",
    "temperature_out": "#B3410A",
    "humidex": "#E10000",
    "dry": "#FF8FF8",
    "auto": "#05C20C",
    "ambi_mode_target": "#F93FDD",
    "away_h_up": "#FCABFB",  # magenta
    "away_t_up": "#FCABFB",  # magenta
    "away_t_low": "#570FA7",  # purple
    "target": "#AD0000",
    "target_humidex": "#DD0088",
    "luminosity": "#049DCC",
    "humidity": "#14529F",
    "comfort": "#2B9F04",
    "comfort_mode": "#0EFF82",
    "manual": "#666666",
    "original_comfort": "#6600CC",
}
SOLID_CAPSTYLE = "butt"
RANGES = {
    # 'temperature': np.arange(-4, 5),
    "humidex": np.arange(-4, 5)
}
RESAMPLE_INTERVAL = "5Min"
OUT_INTERVAL = "15Min"
MODES = ["auto", "cool", "heat", "fan", "dry"]

ControlModes = NamedTuple(
    "ControlModes",
    [
        ("manual", DataFrame),
        ("comfort", DataFrame),
        ("temperature", DataFrame),
        ("humidex", DataFrame),
        ("away_humidity_upper", DataFrame),
        ("away_temperature_upper", DataFrame),
        ("away_temperature_lower", DataFrame),
    ],
)
ControlModes.__new__.__defaults__ = (None,) * len(ControlModes._fields)


async def fetch_weather(
    pool: Pool, location_id: str, start: datetime, end: datetime
) -> DataFrame:
    query, params = queries.query_time_series(
        "WeatherAPI",
        columns=["timestamp", "temperature", "humidity"],
        conditions={"location_id": location_id},
        start=start,
        end=end,
        timestamp_column="timestamp",
    )
    weather = await pool.execute(query, params)
    if weather:
        weather = DataFrame.from_records(weather, index="timestamp")
        weather = pad(weather, start, end)
    else:
        print("no weather data fetched")
        weather = DataFrame()
    return weather


async def fetch_sensors(
    session: CassandraSession, device_id: str, start: datetime, end: datetime
) -> DataFrame:
    seconds = (end - start).total_seconds()
    resample = str(int(seconds / NPOINTS)) + "s"
    sensors = await session.execute_async(
        *cassandra_queries.query_sensor(device_id, start, end)
    )
    sensors = (
        DataFrame([s for s in sensors])
        .set_index("created_on")
        .resample(resample)
        .ffill()
    )
    sensors = compensate_sensors(sensors)
    sensors["humidex"] = thermo.humidex(sensors["temperature"], sensors["humidity"])
    return sensors


async def fetch_timezone(pool: Pool, device_id: str, timezone: str) -> Optional[tzinfo]:
    if timezone == "local":
        tz = await queries.get_time_zone(pool, device_id)
    else:
        tz = None
    return tz


async def get_comfort(
    dbcon: DBConnection,
    session: CassandraSession,
    model: Dict,
    device_id: str,
    start: datetime,
    end: datetime,
    user_id: str,
) -> DataFrame:

    fb = await get_predicted_feedback_range(
        model, dbcon, session, device_id, start, end, user_id
    )
    return DataFrame(fb).set_index("created_on")


async def get_predicted_feedback_range(
    model: Dict,
    pool: Pool,
    cassandra_session: CassandraSession,
    device_id: str,
    start: datetime,
    end: datetime,
    user_id: str = None,
    clip: bool = True,
) -> Dict:
    try:
        raw_data = await user_sample.fetch(
            pool, cassandra_session, device_id, start, end, RESAMPLE_INTERVAL
        )
    except user_sample.InsufficientData:
        return []

    fb_rows, predictions, dataset = await get_user_feedback(
        pool, device_id, raw_data, model, start, end, user_id
    )

    data = process_predictions(fb_rows, predictions, dataset, clip)

    # convert to native python datetime objects
    for record in data:
        record["created_on"] = record["created_on"].to_pydatetime()

    return data


async def get_user_feedback(
    pool: Pool,
    device_id: str,
    raw_data: DataFrame,
    model: Dict,
    start: datetime,
    end: datetime,
    user_id: str,
) -> Tuple[List, pd.Series, Any]:

    timezone = await queries.get_time_zone(pool, device_id)
    samples = make_samples(
        device_id, raw_data, timezone, timestamps=raw_data.index, user_id=user_id
    )
    dataset = user_sample.prepare_dataset(samples)
    y_pred = model.predict(dataset)
    predictions = pd.Series(y_pred, index=dataset.index)

    fb_query = queries.query_user_feedback(
        device_id=device_id, user_id=user_id, start=start, end=end
    )
    fb_rows = await pool.execute(*fb_query)

    return (fb_rows, predictions, dataset)


def make_samples(device_id, raw_data, timezone, timestamps, user_id=None):
    """Make PREDICTION samples for comfort service range predictions."""

    feature_names = (
        user_sample.WEATHER_FEATURES + user_sample.COMFORT_SERVICE_SENSORS_REQUIRED
    )

    sensor_features = preprocess.interpolate(timestamps, raw_data[feature_names])

    localtime = timestamps.tz_localize("UTC").tz_convert(timezone)

    # TODO use user information
    time_features = user_sample.get_time_features(localtime, index=timestamps)

    dataset = pd.concat([sensor_features, time_features], axis=1)

    dataset.reset_index(inplace=True)
    dataset.rename(columns={"created_on": "timestamp"}, inplace=True)

    dataset["device_id"] = device_id
    dataset["user_id"] = user_id
    dataset["type"] = "static"

    good_idx = dataset[user_sample.COMFORT_SERVICE_SENSORS_REQUIRED].dropna().index
    return dataset.loc[good_idx]


def process_predictions(fb_rows, predictions, dataset, clip: bool) -> Dict:
    if fb_rows:
        fb = DataFrame.from_records(fb_rows, columns=["created_on", "feedback"])
        fb.index = fb["created_on"]
        fb["feedback_humidex"] = preprocess.fill(fb.index, dataset["humidex"])
        fb["feedback_prediction"] = preprocess.fill(fb.index, predictions)
        fb = preprocess.fill(dataset.index, fb)
        fb["current_humidex"] = dataset["humidex"]
        fb.index.name = "timestamp"
        for f, (i, prediction) in zip(
            fb.reset_index().to_dict("records"), predictions.iteritems()
        ):
            predictions[i] = comfort_model.get_adjusted_prediction(
                prediction,
                f["timestamp"],
                feedback=f["feedback"],
                feedback_timestamp=f["created_on"],
                feedback_humidex=f["feedback_humidex"],
                feedback_prediction=f["feedback_prediction"],
                current_humidex=f["current_humidex"],
            )

    log.debug(predictions)

    predictions = predictions.resample(OUT_INTERVAL).mean()
    predictions.index.name = "created_on"
    predictions.name = "comfort"

    predictions = analyze.convolve_with_gaussian(predictions, 1.0)

    if clip:
        predictions = np.clip(predictions, MIN_COMFORT, MAX_COMFORT)

    data = predictions.reset_index().to_dict("records")

    return data


async def get_mode_probas(
    pool: Pool,
    session: CassandraSession,
    trained_mode_model,
    device_id: str,
    timestamps: DatetimeIndex,
) -> DataFrame:

    samples = [sample.PredictionSample(device_id, t) for t in timestamps]
    await multi([smp.fetch(pool, session) for smp in samples])

    dataset = []
    for t, smp in zip(timestamps, samples):

        history_features = smp.get_history_feature()
        feats = [
            mode_model.make_features(history_features, quantity, value)
            for quantity, values in RANGES.items()
            for value in values
        ]
        feats = DataFrame(feats)

        y_prob = trained_mode_model.predict_proba(feats)

        # try using new multi mode model, otherwise fall back to old model
        try:
            y_prob = DataFrame.from_records(y_prob)["first_layer_heat"]
        except KeyError:
            y_prob = y_prob[:, 1]

        feats["p_heat"] = y_prob
        feats["timestamp"] = t
        dataset.append(feats)

    return pd.concat(dataset).pivot_table(
        index=["timestamp"], columns=["target"], values=["p_heat"]
    )


async def fetch_user_feedback(
    pool: Pool, device_id: str, start: datetime, end: datetime
) -> List:

    query, params = queries.query_user_feedback(
        device_id=device_id, start=start, end=end
    )
    feedback = await pool.execute(query, params)

    return feedback


async def fetch_user_control_targets(
    pool: Pool, device_id: str, start: datetime, end: datetime
) -> List:

    query, params = queries.query_control_targets(
        device_id=device_id, start=start, end=end
    )
    control_targets = await pool.execute(query, params)
    return [lower_dict(c) for c in control_targets]


async def fetch_appliance_state(
    pool: Pool, device_id: str, start: datetime, end: datetime
) -> List:

    query, params = queries.query_appliance_states_from_device(device_id, start, end)
    states = await pool.execute(query, params)

    return [lower_dict(s) for s in states]


async def fetch_location_from_device(pool: Pool, device_id: str) -> str:

    query, params = queries.query_location_from_device(device_id)
    location_id = await pool.execute(query, params)

    return location_id[0]["location_id"]


def fetch_comfort_pred(
    need_prediction: Dict[str, bool],
    pool: Pool,
    session: CassandraSession,
    model: Dict,
    device_id: str,
    user_id: str,
    start: datetime,
    end: datetime,
) -> Optional[DataFrame]:

    if need_prediction["comfort"]:
        return get_comfort(
            pool, session, model["comfort"], device_id, start, end, user_id
        )

    else:
        return None


def fetch_mode_probas(
    need_prediction: Dict[str, bool],
    pool: Pool,
    session: CassandraSession,
    model: Dict,
    device_id: str,
    user_id: str,
    start: datetime,
    end: datetime,
) -> Optional[DataFrame]:

    if need_prediction["mode"]:
        timestamps = pd.date_range(start, end, freq="15Min").to_pydatetime()
        return get_mode_probas(pool, session, model["mode"], device_id, timestamps)

    else:
        return None


def process_sensors(sensors: DataFrame, start: datetime, end: datetime) -> DataFrame:

    if "luminosity" in sensors:
        sensors["luminosity"] = analyze.convolve_with_gaussian(
            sensors["luminosity"], SMOOTH_POINTS
        )
        sensors["luminosity"] /= np.max(sensors["luminosity"])

    sensors = pad(sensors, start, end)
    return sensors


def process_states(
    states: List, start: datetime, end: datetime
) -> Tuple[DataFrame, Dict]:

    if states:
        states = make_step_series(states, start, end)
        states["temperature_set"] = thermo.fix_temperatures(states["temperature_set"])

        states = states.set_index("created_on")
        states.loc[states["power"] == Power.OFF, "temperature_set"] = np.nan
        mode_states = make_mode_states_to_plot(states)
    else:
        mode_states = {}
    return (states, mode_states)


def make_mode_states_to_plot(states: DataFrame) -> Dict[str, DataFrame]:
    fill_states = states.copy()
    fill_states.loc[
        fill_states["mode"].isin(["auto", "dry"]), "temperature_set"
    ].fillna(24, inplace=True)
    mode_states = {ac_mode: fill_states.copy() for ac_mode in MODES}
    for ac_mode, data in mode_states.items():
        data.loc[data["mode"] != ac_mode, "temperature_set"] = np.nan
    return mode_states


def process_control_targets(
    control_targets: List, start: datetime, end: datetime
) -> Tuple[DataFrame, ControlModes]:

    if control_targets:
        control_targets = make_step_series(control_targets, start, end)
        control_targets = control_targets.set_index("created_on")

        control_modes = [
            "manual",
            "climate",
            "temperature",
            "humidex",
            "away_humidity_upper",
            "away_temperature_upper",
            "away_temperature_lower",
        ]

        control_modes = ControlModes(
            *[process_control_modes(x, control_targets) for x in control_modes]
        )
    else:
        control_modes = ControlModes()

    return (control_targets, control_modes)


def process_control_modes(quantity: str, df: DataFrame) -> DataFrame:
    df = df.copy()
    df.loc[df["quantity"] != quantity, "value"] = np.nan
    if quantity == "manual" or quantity == "climate":
        df.loc[df["quantity"] == quantity, "value"] = 23.5

    return df


def process_feedback(feedback: List) -> DataFrame:
    if feedback:
        return DataFrame(feedback).set_index("created_on")


def create_subplot_temp_and_comfort(
    gs,
    sensors: DataFrame,
    feedback: DataFrame,
    tz: tzinfo,
    weather: DataFrame,
    comfort_pred: Optional[DataFrame],
    control_modes: ControlModes,
    mode_states: Dict,
    need_prediction: Dict,
    start: datetime,
    end: datetime,
) -> None:

    ax = plt.subplot(gs[0])
    ax2 = ax.twinx()
    plot_sensor(ax=ax, data=sensors["temperature"], tz=tz)
    plot_sensor(ax=ax, data=sensors["humidex"], tz=tz)

    if not weather.empty:
        plot_sensor(
            ax=ax,
            data=weather["temperature"],
            label="outdoor temperature",
            color="temperature_out",
            tz=tz,
        )

    for ac_mode, data in mode_states.items():
        plot_mode(ax, data=data, name=ac_mode, tz=tz)

    plot_step(ax, control_modes.manual, "-", COLS["manual"], "manual mode", tz=tz)
    plot_step(
        ax, control_modes.comfort, "-", COLS["comfort_mode"], "comfort mode", tz=tz
    )
    plot_step(
        ax, control_modes.temperature, "-", COLS["target"], "temperature target", tz=tz
    )
    plot_step(
        ax, control_modes.humidex, "-", COLS["target_humidex"], "humidex target", tz=tz
    )
    plot_step(
        ax,
        control_modes.away_temperature_lower,
        "-",
        COLS["away_t_low"],
        "away T low",
        tz=tz,
    )
    plot_step(
        ax,
        control_modes.away_temperature_upper,
        "-",
        COLS["away_t_up"],
        "away T up",
        tz=tz,
    )
    plot_feedback(ax2, feedback, tz=tz)

    if need_prediction["comfort"]:
        plot_sensor(
            ax=ax2,
            data=comfort_pred["comfort"],
            label="estimated comfort",
            color="comfort",
            tz=tz,
            secondary=True,
        )
    ax2.set_ylabel("comfort")
    ax2.set_ylim(-3, 3)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15), ncol=1)
    ax2.grid("off")

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=7)
    ax.set_ylabel("temperature [Celsius]")
    format_time_axis(ax, start, end)


def plot_sensor(ax, data, label=None, color=None, tz=None, secondary=False, **kwargs):
    data = localize(data, tz=tz)
    label = label if label is not None else data.name
    color = color if color is not None else data.name
    color = COLS.get(color, color)
    if secondary:
        # workaround: twinx + df.plot + datetime index doesn't render right
        ax.plot(
            data.index, data.values, label=label, color=color, marker=None, **kwargs
        )
    else:
        data.plot(ax=ax, label=label, color=color, marker=None, **kwargs)


def plot_step(ax, data, style, color, label, tz=None):
    data = localize(data, tz=tz)
    if data is not None and not data["value"].dropna().empty:
        ax.plot_date(
            data.index,
            data["value"],
            style,
            drawstyle="steps-post",
            label=label,
            lw=STATE_LW,
            alpha=STATE_ALPHA,
            color=color,
            solid_capstyle=SOLID_CAPSTYLE,
        )


def plot_mode(ax, data, name, tz=None):
    data = localize(data, tz=tz)
    t_set = data["temperature_set"]
    if not t_set.dropna().empty:
        t_set.plot(
            ax=ax,
            drawstyle="steps-post",
            label=name,
            lw=STATE_LW,
            alpha=STATE_ALPHA,
            color=COLS[name],
            solid_capstyle=SOLID_CAPSTYLE,
        )


def plot_feedback(ax, feedback, tz=None):
    if feedback is not None:
        lf = localize(feedback, tz=tz)
        ax.scatter(
            lf.index,
            lf["feedback"],
            marker="o",
            s=80,
            c=lf["feedback"],
            alpha=0.5,
            edgecolors="black",
            vmin=-3,
            vmax=3,
            cmap=COMFORT_CMAP,
        )


def create_subplot_humidity(
    gs,
    sensors: DataFrame,
    tz: tzinfo,
    control_modes: ControlModes,
    start: datetime,
    end: datetime,
) -> None:

    ax = plt.subplot(gs[1])
    plot_sensor(ax=ax, data=sensors["humidity"], tz=tz)
    plot_step(
        ax,
        control_modes.away_humidity_upper,
        "-",
        COLS["away_h_up"],
        "away H up",
        tz=tz,
    )
    ax.set_ylabel("humidity")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=7)
    format_time_axis(ax, start, end)


def create_subplot_luminosity(
    gs,
    sensors: DataFrame,
    tz: tzinfo,
    mode_probas: Optional[DataFrame],
    mode: bool,
    start: datetime,
    end: datetime,
) -> None:

    ax = plt.subplot(gs[2])
    legends, labels = [], []
    if mode:
        plot_date(ax, [start, end], [0.5, 0.5], "k:", alpha=0.5, tz=tz)
        mode_probas.plot(ax=ax, colormap="jet")
        plot_date(ax, [start, end], [0.5, 0.5], color="blue", alpha=0.05, tz=tz)

    else:
        if "luminosity" in sensors:
            fill_between(ax=ax, data=sensors["luminosity"])
            legends.append(
                plt.Rectangle((0, 0), 1, 1, fc=COLS["luminosity"], alpha=0.5)
            )
            labels.append("luminosity")

        ax.set_yticks([])

    if legends:
        ax.legend(legends, labels, loc="best")

    format_time_axis(ax, start, end)
    ax.set_xticklabels([])
    ax.set_xticklabels([], minor=True)


def plot_date(ax, *args, **kwargs):
    return ax.plot_date(*args, **kwargs)


def fill_between(ax, data, label=None, color=None, alpha=0.5, tz=None):
    data = localize(data, tz=tz)
    label = label if label is not None else data.name
    color = color if color is not None else data.name
    ax.fill_between(
        data.index, data, label=label, color=COLS.get(color, color), alpha=alpha
    )


def create_grid_spec():
    return plt.GridSpec(3, 1, height_ratios=[2, 1, 1])


def save_plot(plt, fig_dir, device_id, start_raw, end_raw):
    fname = os.path.join(fig_dir, f"device_{device_id}_{start_raw}-{end_raw}.png")
    plt.savefig(fname)
    print("saved to", fname)


def load_trained_models(storage, flags: Dict, cnf, directory="data/models") -> Dict:

    model = {"user": None, "comfort": None, "mode": None}
    if any(flags.values()):
        model_store = get_storage(
            storage, **cnf["model_store"], directory="data/models"
        )
        if flags["comfort"]:
            model["comfort"] = model_store.load(
                comfort_model.ComfortModel.get_storage_key()
            )
        if flags["mode"]:
            model["mode"] = model_store.load(mode_model.ModeModel.get_storage_key())
    return model


def create_figure_directory(data_dir: str, fig_dir: str) -> None:
    for directory in [data_dir, fig_dir]:
        try:
            os.mkdir(directory)
        except OSError:
            pass


def format_time_axis(ax, start: datetime, end: datetime):
    days = (end - start).total_seconds() / (24 * 3600)
    if days <= 1:
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(HourLocator(range(0, 24, 1)))
        ax.xaxis.set_minor_formatter(DateFormatter("\n%I %p"))
    elif days <= 7:
        ax.xaxis.set_major_locator(HourLocator([0]))
        ax.xaxis.set_major_formatter(DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(HourLocator([6, 12, 18]))
        ax.xaxis.set_minor_formatter(DateFormatter("%I %p"))
    ax.grid(b=True, which="minor", linestyle=":")


def localize(data, tz):
    if tz is not None:
        data = data.copy()
        local = [pytz.utc.localize(x).astimezone(tz) for x in data.index]
        data.index = [d.replace(tzinfo=None) for d in local]
        return data

    else:
        return data


def make_step_series(series, start: datetime, end: datetime) -> DataFrame:
    first = series[0].copy()
    first["created_on"] = start
    last = series[-1].copy()
    last["created_on"] = end
    with_nans = [first]
    for a, b in zip(series[:-1], series[1:]):
        a, b, empty = a.copy(), b.copy(), b.copy()
        a["created_on"] = b["created_on"]
        for key in empty:
            if key != "created_on":
                empty[key] = np.nan
        with_nans.append(a)
        with_nans.append(empty)
        with_nans.append(b)
    with_nans.append(last)
    return DataFrame.from_records(with_nans)


def pad(df: DataFrame, start: datetime, end: datetime) -> DataFrame:
    df = df.copy()
    df.loc[end] = df.iloc[-1]  # this won't work properly if order is reversed
    df.loc[start] = df.iloc[0]
    df = df.sort_index()
    return df
