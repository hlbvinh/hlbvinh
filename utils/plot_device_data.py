from collections import ChainMap
from datetime import datetime, timedelta, tzinfo
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import click
import matplotlib.pyplot as plt
import yaml
from dateutil.parser import parse
from pandas import DataFrame

import plot_device_data_utils as utils
from skynet.utils.async_util import multi, run_sync
from skynet.utils.database import dbconnection
from skynet.utils.database.cassandra import CassandraSession
from skynet.utils.database.dbconnection import Pool
from skynet.utils.log_util import init_logging

START = (datetime.utcnow() - timedelta(hours=72)).isoformat()
END = datetime.utcnow().isoformat()

Data = NamedTuple(
    "Data",
    [
        ("sensors", DataFrame),
        ("feedback", Union[List, DataFrame]),
        ("control_targets", Union[List, DataFrame]),
        ("states", Union[List, DataFrame]),
        ("tz", Optional[tzinfo]),
        ("location_id", str),
        ("weather", DataFrame),
        ("comfort_pred", Optional[DataFrame]),
        ("mode_probas", Optional[DataFrame]),
    ],
)


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--device_id",
    default="1274fdb6-87b4-4ce5-8bcf-e3c8cbef2d8d",
    help="Device ID for predictions.",
)
@click.option(
    "--user_id",
    default="1e5aa52a-9302-11e3-a724-0683ac059bd8",
    help="User ID for comfort predictions.",
)
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="viewer")
@click.option("--start", default=START, help="From when to fetch data (UTC).")
@click.option("--end", default=END, help="Until when to fetch data (UTC).")
@click.option("--data_dir", default="data", help="Directory to save csv.")
@click.option("--fig_dir", default="fig", help="Directory to save csv.")
@click.option("--timezone", default="utc", type=click.Choice(["local", "utc"]))
@click.option("--comfort", default=False, is_flag=True)
@click.option("--mode", default=False, is_flag=True)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="file")
def main(
    config,
    device_id,
    user_id,
    mysql,
    cassandra,
    start,
    end,
    data_dir,
    fig_dir,
    timezone,
    comfort,
    mode,
    storage,
):
    utils.create_figure_directory(data_dir, fig_dir)
    create_plots(
        config,
        start,
        end,
        comfort,
        mode,
        storage,
        device_id,
        user_id,
        timezone,
        fig_dir,
        mysql,
        cassandra,
    )


def create_plots(
    config,
    start,
    end,
    comfort,
    mode,
    storage,
    device_id,
    user_id,
    timezone,
    fig_dir,
    mysql="viewer",
    cassandra="viewer",
):
    init_logging("plot_device_data")
    start_raw, end_raw = start, end
    start = parse(start_raw)
    end = parse(end_raw)

    model_flags = {"comfort": comfort, "mode": mode}

    with open(config) as filehandle:
        cnf = yaml.safe_load(filehandle)

    db_cnf = cnf[mysql]

    model = utils.load_trained_models(
        storage, model_flags, cnf, directory="data/models"
    )

    dbcon = dbconnection.DBConnection(**db_cnf)
    pool = dbconnection.Pool.from_dbconnection(dbcon)
    session = CassandraSession(**cnf["cassandra"][cassandra])

    data_from_db = Data(
        **run_sync(
            fetch_all_data,
            session,
            pool,
            device_id,
            user_id,
            start,
            end,
            timezone,
            model_flags,
            model,
        )
    )

    processed_data, control_modes, mode_states = process_data(data_from_db, start, end)

    plt = create_graphs(
        processed_data, control_modes, mode_states, model_flags, start, end
    )

    utils.save_plot(plt, fig_dir, device_id, start_raw, end_raw)


async def fetch_all_data(
    session: CassandraSession,
    pool: Pool,
    device_id: str,
    user_id: str,
    start: datetime,
    end: datetime,
    timezone: str,
    need_prediction: Dict[str, bool],
    model: Dict[str, Any],
) -> Dict[str, Any]:

    location_id = await utils.fetch_location_from_device(pool, device_id)

    datasets = {
        "sensors": utils.fetch_sensors(session, device_id, start, end),
        "feedback": utils.fetch_user_feedback(pool, device_id, start, end),
        "control_targets": utils.fetch_user_control_targets(
            pool, device_id, start, end
        ),
        "states": utils.fetch_appliance_state(pool, device_id, start, end),
        "tz": utils.fetch_timezone(pool, device_id, timezone),
        "weather": utils.fetch_weather(pool, location_id, start, end),
        "comfort_pred": utils.fetch_comfort_pred(
            need_prediction, pool, session, model, device_id, user_id, start, end
        ),
        "mode_probas": utils.fetch_mode_probas(
            need_prediction, pool, session, model, device_id, user_id, start, end
        ),
    }

    datasets_to_fetch = {k: v for k, v in datasets.items() if v is not None}
    response = await multi(datasets_to_fetch)
    response["location_id"] = location_id

    return ChainMap(response, datasets)


def process_data(
    data_from_db: Data, start: datetime, end: datetime
) -> Tuple[Data, utils.ControlModes, Dict]:

    data = data_from_db._asdict()

    data["sensors"] = utils.process_sensors(data["sensors"], start, end)

    data["states"], mode_states = utils.process_states(data["states"], start, end)

    data["control_targets"], control_modes = utils.process_control_targets(
        data["control_targets"], start, end
    )

    data["feedback"] = utils.process_feedback(data["feedback"])

    return (Data(**data), control_modes, mode_states)


def create_graphs(
    data: Data,
    control_modes: utils.ControlModes,
    mode_states: Dict,
    need_prediction: Dict[str, bool],
    start: datetime,
    end: datetime,
) -> None:

    gs = utils.create_grid_spec()

    utils.create_subplot_temp_and_comfort(
        gs,
        data.sensors,
        data.feedback,
        data.tz,
        data.weather,
        data.comfort_pred,
        control_modes,
        mode_states,
        need_prediction,
        start,
        end,
    )

    utils.create_subplot_humidity(gs, data.sensors, data.tz, control_modes, start, end)

    utils.create_subplot_luminosity(
        gs, data.sensors, data.tz, data.mode_probas, need_prediction["mode"], start, end
    )

    return plt


if __name__ == "__main__":
    main()
