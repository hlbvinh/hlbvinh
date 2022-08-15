import sys

import click
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly


from skynet.user.store import UserSampleStore
from skynet.utils import thermo
from skynet.utils.async_util import run_sync
from skynet.utils.compensation import compensate_features
from skynet.utils.database import dbconnection, queries
from skynet.utils.mongo import Client
from utils.visualize_comfort_day import localize


@click.command()
@click.option("--config", default="config.yml")
@click.option("--device_id", "-d", required=True)
@click.option("--user_id", "-u")
@click.option("--mongo", default="test")
@click.option("--mysql", default="viewer")
@click.option("--output", "-o")
@click.option(
    "--feature",
    "-f",
    type=click.Choice(
        [
            "humidity",
            "humidity_out",
            "luminosity",
            "pircount",
            "pirload",
            "temperature",
            "temperature_out",
            "time",
            "humidex",
        ]
    ),
    multiple=True,
    required=True,
    default=["humidex", "humidity", "temperature"],
    help="Plot distribution of samples with features.",
)
@click.option(
    "--plotly_feature",
    type=click.Choice(
        [
            "humidity",
            "humidity_out",
            "luminosity",
            "pircount",
            "pirload",
            "temperature",
            "temperature_out",
            "humidex",
        ]
    ),
    default=None,
    help="Plotly y-axis, ex: humidex (y-axis) vs timestamp, feedback color coded.",
)
@click.option("--timezone", type=click.Choice(["local", "utc"]), default="local")
def main(
    config, device_id, user_id, mongo, mysql, output, feature, plotly_feature, timezone
):

    cnf = yaml.safe_load(open(config))
    db_cnf = cnf[mysql]
    mongo_cnf = cnf["mongo"][mongo]

    mongo_client = Client(**mongo_cnf)
    sample_store = UserSampleStore(mongo_client)
    pool = dbconnection.get_pool(**db_cnf)

    key = {"type": "user_feedback", "device_id": device_id}
    if user_id:
        key.update({"user_id": user_id})
    data = pd.DataFrame(sample_store.get(key=key, limit=0))

    if data.empty:
        print("No data for this device id, user id.")
        sys.exit(1)

    unique_user_id = "\n".join(data["user_id"].unique())

    subset_on_user_id = input(
        f"Which user id you would like to see the samples for "
        f"(Press enter if all): \n{unique_user_id}:\n"
    )

    if subset_on_user_id in data["user_id"].unique():
        data = data.loc[data["user_id"] == subset_on_user_id]
        title = (
            f'Time series of Feedback for device_id: {data.iloc[0]["device_id"]} '
            f'& user_id: {data.iloc[0]["user_id"]}'
        )
    else:
        title = f'Time series of Feedback for device_id: {data.iloc[0]["device_id"]}'

    data = data.set_index("timestamp")

    if timezone == "local":
        tz = run_sync(queries.get_time_zone, pool, device_id)
    else:
        tz = "UTC"

    data.index = localize(data.index, tz)
    data = compensate_features(data)
    data["humidex"] = thermo.humidex(data["temperature"], data["humidity"])

    data["time"] = data.index.hour + data.index.minute / 60.0

    cols = set(feature) & set(data.columns)

    data = data.reset_index().rename(columns={"index": "timestamp"})

    create_plot(data.reset_index(), title, feature=plotly_feature)

    sns_plot = sns.pairplot(
        data=data,
        hue="feedback",
        vars=cols,
        hue_order=list(range(-3, 4)),
        palette=sns.color_palette("RdBu_r", 7),
    )
    if not output:
        plt.show()
    else:
        sns_plot.savefig(output)


def create_tooltip(x):
    text = {
        "feedback": x["feedback"],
        "temperature": x["temperature"],
        "humidity": x["humidity"],
        "humidex": x["humidex"],
        "user_id_last_four": x["user_id"][-4:],
    }

    return str(text)


def create_plot(raw_samples, title, feature=None):
    raw_samples["text"] = raw_samples.apply(lambda x: create_tooltip(x), axis=1)

    trace = go.Scatter(
        x=raw_samples["timestamp"],
        y=raw_samples[feature] if feature is not None else raw_samples["feedback"],
        text=raw_samples["text"],
        marker={
            "size": 8,
            "cmin": -3,
            "cmax": 3,
            "color": raw_samples["feedback"],
            "colorscale": [
                [0, "#4F607F"],
                [0.14285, "#4F607F"],
                [0.14285, "#728DBE"],
                [0.285714, "#728DBE"],
                [0.285714, "#A0BCE2"],
                [0.42857, "#A0BCE2"],
                [0.42857, "#73C9C0"],
                [0.57142, "#73C9C0"],
                [0.57142, "#F7A9AA"],
                [0.7142, "#F7A9AA"],
                [0.7142, "#F4918E"],
                [0.9, "#F4918E"],
                [0.9, "#F27573"],
                [1, "#F27573"],
            ],
        },
        mode="markers",
    )

    data = [trace]
    layout = dict(
        title=title,
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=5, label="5m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(),
            type="date",
        ),
        yaxis=dict(title=feature if feature is not None else "feedback"),
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
