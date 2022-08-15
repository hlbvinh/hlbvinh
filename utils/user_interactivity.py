from datetime import datetime, timedelta
from typing import List

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from skynet.utils.database import dbconnection, queries


def get_daikin_devices(dbcon) -> List[str]:
    q = """
    SELECT
       real_device_id, created_on 
    FROM
       Device 
    WHERE
       device_id IN 
       (
          SELECT
             device_id 
          FROM
             UserDeviceList 
          WHERE
             user_id IN 
             (
                SELECT
                   User.user_id 
                FROM
                   User 
                   INNER JOIN
                      (
                         SELECT DISTINCT
                            user_id 
                         FROM
                            DaikinUserList
                      )
                      AS B 
                      ON User.user_id = B.user_id 
                WHERE
                   User.created_on > "2019-11-30"
             )
       )
    """
    return pd.DataFrame(queries.execute(dbcon, q))["real_device_id"].to_list()


def get_valid_real_devices(dbcon, start: datetime, end: datetime) -> pd.DataFrame:
    q = """
    SELECT
       real_device_id,
       MIN(created_on) AS created_on 
    FROM
       (
          SELECT
             Device.real_device_id,
             Device.device_id,
             FirstDates.created_on 
          FROM
             Device 
             INNER JOIN
                (
                   SELECT
                      device_id,
                      MIN(created_on) AS created_on 
                   FROM
                      ApplianceControlTarget 
                   GROUP BY
                      device_id
                )
                AS FirstDates 
                ON Device.device_id = FirstDates.device_id
       )
       AS RealDeviceDates 
    GROUP BY
       real_device_id 
    """
    device = pd.DataFrame(queries.execute(dbcon, q))
    device["created_on"] = pd.to_datetime(device["created_on"], errors="coerce")
    device = device[(device["created_on"] > start) & (device["created_on"] < end)]
    return device


def get_control_target(dbcon, valid_real_devices: pd.DataFrame) -> pd.DataFrame:
    q = """
    SELECT
       SM.real_device_id,
       created_on,
       quantity,
       origin 
    FROM
       ApplianceControlTarget 
       INNER JOIN
          (
             SELECT
                Device.device_id,
                Device.real_device_id 
             FROM
                Device 
                INNER JOIN
                   (
                      SELECT
                         real_device_id,
                         MIN(created_on) AS created_on 
                      FROM
                         (
                            SELECT
                               Device.real_device_id,
                               Device.device_id,
                               FirstDates.created_on 
                            FROM
                               Device 
                               INNER JOIN
                                  (
                                     SELECT
                                        device_id,
                                        MIN(created_on) AS created_on 
                                     FROM
                                        ApplianceControlTarget 
                                     GROUP BY
                                        device_id
                                  )
                                  AS FirstDates 
                                  ON Device.device_id = FirstDates.device_id
                         )
                         AS RealDeviceDates 
                      GROUP BY
                         real_device_id
                   )
                   AS B 
                   ON Device.real_device_id = B.real_device_id 
             WHERE
                B.created_on > "2018 - 01 - 01"
          )
          AS SM 
          ON SM.device_id = ApplianceControlTarget.device_id    
    """
    control_target = pd.DataFrame(queries.execute(dbcon, q))
    control_target = pd.merge(
        control_target.sort_values(["real_device_id", "created_on"]),
        valid_real_devices[["real_device_id"]],
        on="real_device_id",
    )
    control_target["created_on"] = pd.to_datetime(control_target["created_on"])
    control_target["quantity"] = control_target["quantity"].str.split(
        pat="_", n=1, expand=True
    )
    control_target["interactivity"] = control_target.apply(
        lambda x: "Off"
        if x["origin"] in ["FailedReverse", "Reverse"]
        else x["quantity"],
        axis=1,
    )
    return control_target


def get_device_connectivity(dbcon, valid_real_devices: pd.DataFrame) -> pd.DataFrame:
    q = """
    SELECT
       Device.real_device_id,
       DeviceConnection.created_on,
       DeviceConnection.connection 
    FROM
       DeviceConnection 
       INNER JOIN
          Device 
          ON DeviceConnection.device_id = Device.device_id
    """
    device_connection = pd.DataFrame(queries.execute(dbcon, q))
    connectivity = pd.merge(
        device_connection.sort_values(["real_device_id", "created_on"]),
        valid_real_devices[["real_device_id"]],
        on="real_device_id",
    )
    connectivity["created_on"] = pd.to_datetime(
        connectivity["created_on"], errors="coerce"
    )
    return connectivity


def date_range(start: datetime, is_weekly_analysis: bool) -> List[datetime]:
    delta = 7 if is_weekly_analysis else 30.2
    timestamps = []
    timestamp = start
    while timestamp < datetime.utcnow():
        timestamps.append(timestamp)
        timestamp += timedelta(days=delta)
    return timestamps


def generate_fake_timestamp(x: pd.Series, is_weekly_analysis: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "real_device_id": x["real_device_id"],
            "created_on": date_range(x["created_on"], is_weekly_analysis),
        }
    )


def generate_fake_dates(
    valid_real_device: pd.DataFrame, is_weekly_analysis: bool
) -> pd.DataFrame:
    fake_dates = pd.concat(
        list(
            valid_real_device.apply(
                generate_fake_timestamp, args=(is_weekly_analysis,), axis=1
            )
        )
    )
    fake_dates["number"] = fake_dates.groupby("real_device_id").cumcount()
    return fake_dates


def insert_fake_dates(df: pd.DataFrame, fake_dates: pd.DataFrame) -> pd.DataFrame:
    df = df.append(fake_dates, sort=False).sort_values(["real_device_id", "created_on"])
    return df.groupby("real_device_id").apply(lambda x: x.ffill()).dropna()


def check_origin(quantity: str) -> bool:
    return bool(set(["Temperature", "Manual", "Climate", "Away"]) & set(quantity))


def generate_plots(
    df: pd.DataFrame,
    months: int,
    is_weekly_analysis: bool,
    daikin_devices: List[str],
    quantity: str,
) -> None:
    daikin = df[df["real_device_id"].isin(daikin_devices)]
    ambi = df[~df["real_device_id"].isin(daikin_devices)]

    limit = (months * 4) + 1 if is_weekly_analysis else months + 1
    daikin_stats = daikin.groupby("number").mean().reset_index().iloc[:limit]
    ambi_stats = ambi.groupby("number").mean().reset_index().iloc[:limit]

    plt.figure()
    plt.xlabel("week" if is_weekly_analysis else "month")
    plt.ylabel("% of original users")
    sns.lineplot(
        daikin_stats["number"], daikin_stats[quantity], label=f"daikin_{quantity}"
    )
    sns.lineplot(ambi_stats["number"], ambi_stats[quantity], label=f"ambi_{quantity}")
    plt.savefig(
        f"{quantity}_months={months}_is_weekly_analysis={is_weekly_analysis}.png"
    )


@click.command()
@click.option("--config", default="/Users/sherman/config.yml")
@click.option("--is_weekly_analysis", default=True, is_flag=True)
@click.option("--months", default=1)
def main(config, is_weekly_analysis, months):

    with open(config) as f:
        cnf = yaml.load(f)

    dbcon = dbconnection.DBConnection(**cnf["viewer"])

    start, end = datetime(2018, 1, 1), datetime.utcnow() - timedelta(days=months * 30)

    daikin_devices = get_daikin_devices(dbcon)
    valid_real_devices = get_valid_real_devices(dbcon, start, end)
    fake_dates = generate_fake_dates(valid_real_devices, is_weekly_analysis)

    control_target = get_control_target(dbcon, valid_real_devices)
    control_target = insert_fake_dates(control_target, fake_dates)
    interactivity = (
        control_target.groupby(["number", "real_device_id"])["interactivity"]
        .apply(check_origin)
        .reset_index()
    )
    generate_plots(
        interactivity,
        months,
        is_weekly_analysis,
        daikin_devices,
        quantity="interactivity",
    )

    device_connectivity = get_device_connectivity(dbcon, valid_real_devices)
    device_connectivity = insert_fake_dates(device_connectivity, fake_dates)
    connectivity = (
        device_connectivity.groupby(["number", "real_device_id"])["connection"]
        .any()
        .reset_index()
    )
    generate_plots(
        connectivity, months, is_weekly_analysis, daikin_devices, quantity="connection"
    )


if __name__ == "__main__":
    main()
