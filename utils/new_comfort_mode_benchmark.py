from datetime import timedelta

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mean_absolute_error

from skynet.user import comfort_model, sample
from skynet.user.store import UserSampleStore
from skynet.utils.analyze import convolve_with_gaussian
from skynet.utils.mongo import Client


def fit_predict(est, X_train, y_train, X_test):
    est.fit(X_train, y_train)
    return est.predict(X_test)


def benchmark(dataset, device_id, n_points):

    print("benchmarking {}".format(device_id))

    dataset = dataset.copy()

    # most common for this device
    user_id = dataset[dataset.device_id == device_id].user_id.value_counts().index[0]

    # drop feedback from user for other devices
    # drop feedback for this device from other users
    drop_idx = np.where(
        ((dataset.user_id == user_id) & (dataset.device_id != device_id))
        | ((dataset.user_id != user_id) & (dataset.device_id == device_id))
    )
    dataset = dataset.drop(dataset.index[drop_idx])

    # sort feedback by time
    idx = np.where(dataset.device_id == device_id)[0]
    idx = idx[np.argsort(dataset.iloc[idx].timestamp)]

    # filter feedback that is too close together
    filtered_idx = []
    drop = []
    for i, j in zip(idx[:-1], idx[1:]):
        dt = dataset.iloc[j]["timestamp"] - dataset.iloc[i]["timestamp"]
        if dt > timedelta(minutes=30):
            filtered_idx.append(i)
        else:
            drop.append(i)
    idx = np.array(filtered_idx)
    drop = np.array(drop)
    print("n device fb", len(dataset[dataset.device_id == device_id]))
    print("n user fb", len(dataset[dataset.user_id == user_id]))
    X, y = comfort_model.split(sample.prepare_dataset(dataset))

    errors = {}
    for i in range(min(n_points, len(idx))):

        # drop newer feedback from user
        drop_idx = idx[i:]
        drops = [set(drop)]
        newer = np.where(dataset.timestamp > dataset.iloc[idx[i]]["timestamp"])[0]
        drops.append(set(newer))
        drop_idx = np.array(list(set(drop_idx).union(*drops)))
        X_fit = X.drop(X.index[drop_idx])
        y_fit = y.drop(y.index[drop_idx])

        if len(X_fit) == 0:
            continue

        # compute error for following feedback from user/device
        pred_idx = idx[i : i + 50]
        y_pred = fit_predict(
            comfort_model.ComfortModel(), X_fit, y_fit, X.iloc[pred_idx]
        )
        errors[i] = mean_absolute_error(y_pred, y.iloc[pred_idx])
        print(i, errors[i])
    return pd.Series(errors, name=device_id)


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mongo", default="test")
@click.option(
    "--task",
    required=True,
    default="initial",
    type=click.Choice(["continuous", "initial"]),
)
@click.option("--device_id", default=None, help="Device ID for benchmark.")
def main(config, mongo, task, device_id):
    try:
        cnf = yaml.safe_load(open(config))
    except IOError:
        print("config.yml not found, using sample")
        cnf = yaml.safe_load(open("config.yml.sample"))

    mongo_cnf = cnf["mongo"][mongo]

    mongo_client = Client(**mongo_cnf)
    sample_store = UserSampleStore(mongo_client)

    def load_training_data():
        return pd.DataFrame(sample_store.get())

    errors = []

    dataset = load_training_data()
    diffs = (
        dataset.sort("timestamp")
        .groupby("device_id")["timestamp"]
        .diff()
        .fillna(timedelta(days=1))
    )
    keep = diffs[diffs > timedelta(minutes=30)].index
    print(keep)
    dataset = dataset.loc[keep]
    dataset = dataset.iloc[np.random.choice(range(len(dataset)), size=5000)]

    if device_id is None:
        v = dataset.device_id.value_counts()
        start = dataset.groupby("device_id")["timestamp"].min()
        start.sort()
        if task == "initial":
            n_points = 70
            n_dev = 7
            enough = start[v > n_points]
            print("n devices", len(enough))
            device_ids = list(enough.iloc[:n_dev].index)
        else:
            n_points = 20
            n_dev = 20
            enough = start[v > n_points]
            print("n devices", len(enough))
            device_ids = list(enough.iloc[-n_dev:].index)
    else:
        device_ids = [device_id]

    for device_id in device_ids:
        errors.append(benchmark(dataset, device_id, n_points))
    data = pd.concat(errors, 1)
    data = convolve_with_gaussian(data.bfill(), 1.0)

    ax = data.plot(legend=False, linewidth=0.5)
    sns.tsplot(
        ax=ax, data=data.ffill().values.T, linewidth=2, linestyle="--", condition="mean"
    )

    if task == "continuous":
        ax.legend_.remove()
    plt.xlabel("number of feedbacks given")
    plt.ylabel("mean absolute error")
    fname = "fig/new_comfort_bench_{}.png".format(task)
    plt.savefig(fname)
    print("saved figure to {}".format(fname))


if __name__ == "__main__":
    main()
