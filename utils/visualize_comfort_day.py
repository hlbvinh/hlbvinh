import os

import click
import matplotlib.pyplot as plt
import pytz
import seaborn as sns
import yaml
from dateutil.parser import parse

from skynet.user.store import UserSampleStore
from skynet.utils import thermo
from skynet.utils.async_util import run_sync
from skynet.utils.compensation import compensate_features
from skynet.utils.database import dbconnection, queries
from skynet.utils.mongo import Client

plt.rcParams["figure.figsize"] = 10, 6
plt.rcParams["lines.linewidth"] = 2
sns.set_style("whitegrid")
# sns.set_style({'font.family': ['Lato']})

START = "2015-1-1"
END = "2016-1-1"


def localize(dates, tz):
    local = [pytz.utc.localize(x).astimezone(tz) for x in dates]
    return [d.replace(tzinfo=None) for d in local]


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--device_id", default="6409FF383939473443067324", help="Device ID for predictions."
)
@click.option("--mysql", default="replication")
@click.option("--mongo", default="test")
@click.option("--start", default=START, help="From when to fetch data (UTC).")
@click.option("--end", default=END, help="Until when to fetch data (UTC).")
@click.option("--data_dir", default="data", help="Directory to save csv.")
@click.option("--fig_dir", default="fig", help="Directory to save csv.")
@click.option("--timezone", default="local", type=click.Choice(["local", "utc"]))
def main(config, device_id, mysql, start, end, data_dir, fig_dir, timezone, mongo):

    start_raw, end_raw = start, end
    start = parse(start_raw)
    end = parse(end_raw)

    cnf = yaml.safe_load(open(config))

    db_cnf = cnf[mysql]
    pool = dbconnection.get_pool(**db_cnf)

    mongo_cnf = cnf["mongo"][mongo]
    mongo_client = Client(**mongo_cnf)
    sample_store = UserSampleStore(mongo_client)

    data = sample_store.get_samples(
        key={"type": "user_feedback", "device_id": device_id}, limit=0
    )
    data = data.set_index("timestamp")

    if timezone == "local":
        tz = run_sync(queries.get_time_zone, dbcon, device_id)
    else:
        tz = "UTC"

    print(data)
    data.index = localize(data.index, tz)
    data = compensate_features(data)
    data["humidex"] = thermo.humidex(data["temperature"], data["humidity"])

    data["time"] = data.index.hour + data.index.minute / 60.0

    plt.scatter(data["time"], data["humidex"], c=data["feedback"], cmap="jet")
    plt.colorbar()
    plt.ylabel("humidex")
    plt.xlabel("time of day")

    # fetch all data
    plt.savefig(
        os.path.join(
            fig_dir, "comfort_device_{}_{}-{}.png".format(device_id, start_raw, end_raw)
        )
    )


if __name__ == "__main__":
    main()
