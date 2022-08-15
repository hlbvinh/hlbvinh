from datetime import datetime, timedelta

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from user_interactivity import get_valid_real_devices
from skynet.utils.database import dbconnection, queries


def get_device_appliance_history(dbcon) -> pd.DataFrame:
    q = """
    SELECT
       B.real_device_id,
       B.device_id,
       appliance_id,
       irprofile_id,
       DeviceApplianceHistory.created_on 
    FROM
       DeviceApplianceHistory 
       INNER JOIN
          Device AS B 
          ON DeviceApplianceHistory.device_id = B.device_id    
    """
    device_appliance_history = pd.DataFrame(queries.execute(dbcon, q)).sort_values(
        ["real_device_id", "created_on"]
    )
    device_appliance_history["created_on"] = pd.to_datetime(
        device_appliance_history["created_on"], errors="coerce"
    )
    return device_appliance_history


def time_to_setup(x: pd.Series) -> float:
    x = x.reset_index(drop=True)
    first_nan = pd.isnull(x).any(1).nonzero()[0][0]
    timestamp = x.iloc[first_nan]["created_on"]
    timeframe = x.loc[
        (x["created_on"] > timestamp - timedelta(minutes=10))
        & (x["created_on"] < timestamp + timedelta(minutes=10))
    ]
    return (
        timeframe["created_on"].max() - timeframe["created_on"].min()
    ).total_seconds()


@click.command()
@click.option("--config", default="config.yml")
@click.option("--cut_off", default=1000)
@click.option("--months", default=1)
def main(config, cut_off, months):

    with open(config) as f:
        cnf = yaml.load(f)

    dbcon = dbconnection.DBConnection(**cnf["viewer"])

    start, end = datetime(2018, 1, 1), datetime.utcnow() - timedelta(days=months * 30)

    device = get_valid_real_devices(dbcon, start, end)
    device_appliance_history = get_device_appliance_history(dbcon)
    device_appliance = pd.merge(
        device_appliance_history, device[["real_device_id"]], on=["real_device_id"]
    ).fillna(0)

    df = (
        device_appliance.append(device, sort=False)
        .sort_values(["real_device_id", "created_on"])
        .reset_index(drop=True)
    )
    results = df.groupby(["real_device_id"]).apply(time_to_setup).reset_index()
    results = results.loc[results[0] <= cut_off]

    plt.figure()
    plt.xlabel("seconds")
    plt.ylabel("count")
    sns.distplot(results[0].dropna())
    plt.savefig("user_setup_time.png")


if __name__ == "__main__":
    main()
