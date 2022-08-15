import os
from datetime import timedelta, datetime
from typing import Any, Dict, Optional
import skynet.utils.openblas_thread_hack  # noqa pylint:disable=unused-import

# pylint: disable=wrong-import-order
import click
import numpy as np
import pandas as pd
import yaml
from functools import partial
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.metrics import SCORERS, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

# pylint:enable=wrong-import-order

from skynet.user import comfort_model, sample
from skynet.user.store import UserSampleStore
from skynet.utils import sklearn_utils
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.misc import timeit
from skynet.utils.script_utils import (
    get_connections,
    get_sample_limit_according_to_number_of_months,
)
from skynet.utils.storage import get_storage
from utils.comfort_model_offline_metrics import hour


log = get_logger("skynet")

TRAIN_WITH_OUTLIERS = False
VALIDATION_SPLIT_INTERVAL = "6D"

FETCH_MONTH_BUFFER_OFFSET = 2
AVG_PARAMS = {
    14: {"average_interval": "24H", "rolling_window": "168H"},
    24: {"average_interval": "6D", "rolling_window": "24D"},
    90: {"average_interval": "24H", "rolling_window": "336H"},
    365: {"average_interval": "24H", "rolling_window": "336H"},
}
HOURS_OF_DAY = np.arange(0, 24)


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mongo", default="production")
@click.option(
    "--task",
    default="train",
    type=click.Choice(["train", "score", "test", "predict", "grid_search"]),
)
@click.option(
    "--validation_interval",
    default="24",
    type=click.Choice(str(a) for a in sorted(AVG_PARAMS)),
)
@click.option("--score_outliers", is_flag=True, default=False)
@click.option(
    "--scoring", type=click.Choice(SCORERS), default="neg_mean_absolute_error"
)
@click.option("--log_directory", default="log")
@click.option("--until", default=None)
@click.option(
    "--multiple_metrics",
    default=False,
    is_flag=True,
    help=(
        "Evaluate metrics: MAE per sample, weighted MAE, "
        "MAE per feedback type, MAE per hour of the day"
    ),
)
@click.option("--feature_to_evaluate", type=click.Choice(["device_id"]), default=None)
@click.option("--sample_limit", type=int, default=80000)
@click.option(
    "--month_limit",
    type=int,
    default=14,
    help=(
        "Limit the samples by taking last x months. Ignores sample_limit. "
        "It fetches data from current date minus month_limit till current date. "
        "It follows the same behavior irrespective of the flag --until."
    ),
)
@click.option("--version", type=int, default=None)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
@click.option("--samples_score_dir", default="data/samples_optimization")
@click.option("--n_jobs", type=int, default=1)
def main(
    config,
    mongo,
    task,
    validation_interval,
    score_outliers,
    scoring,
    log_directory,
    until,
    multiple_metrics,
    feature_to_evaluate,
    sample_limit,
    month_limit,
    version,
    storage,
    samples_score_dir,
    n_jobs,
):

    dir_path = f"./{samples_score_dir}/comfort"

    validation_interval = int(validation_interval)

    Model, model_store, sample_store = initialize_model_and_store(
        config, log_directory, mongo, storage
    )

    if task == "score":
        model = Model()
        model.set_params(estimator__filter__bypass=not score_outliers)
        task_score(
            sample_store,
            model,
            scoring,
            dir_path,
            month_limit,
            sample_limit,
            until,
            validation_interval,
            n_jobs,
            feature_to_evaluate,
            multiple_metrics,
        )

    elif task == "train":
        task_train(sample_store, model_store, until, sample_limit, month_limit, n_jobs)

    elif task == "grid_search":
        model = Model()
        model.set_params(estimator__filter__bypass=not score_outliers)
        task_grid_search(
            model,
            n_jobs,
            sample_limit,
            month_limit,
            sample_store,
            scoring,
            until,
            validation_interval,
        )

    elif task == "predict":
        task_predict(
            Model, model_store, sample_store, until, version, dir_path, multiple_metrics
        )


def initialize_model_and_store(
    config: str, log_directory: str, mongo: str, storage: str
):
    with open(config) as f:
        cnf = yaml.safe_load(f)
    init_logging_from_config(
        "comfort_model_script", cnf=cnf, log_directory=log_directory
    )
    model_store = get_storage(storage, directory="data/models", **cnf["model_store"])
    sample_store = UserSampleStore(get_connections(cnf, mongo=mongo).mongo)

    return comfort_model.ComfortModel, model_store, sample_store


def task_score(
    sample_store,
    model,
    scoring: str,
    dir_path: str,
    month_limit: int,
    sample_limit: int,
    until: Optional[str],
    validation_interval: int,
    n_jobs: int,
    feature_to_evaluate: str,
    multiple_metrics: bool,
) -> None:

    dataset, sample_limit = fetch_dataset_based_on_month_limit_and_validation_interval(
        month_limit, sample_limit, sample_store, until, validation_interval
    )

    X, y = comfort_model.split(sample.prepare_dataset(dataset))
    log.info(f"Sample limit {sample_limit}")

    timestamp = get_timestamp(X)

    os.makedirs(dir_path, exist_ok=True)

    if feature_to_evaluate:
        scorer = create_scorer_for_feature(
            feature=feature_to_evaluate,
            values=X[feature_to_evaluate].unique(),
            default_scorer=mean_absolute_error,
        )
        fname = (
            f"comfort_model_mae_score_per_{feature_to_evaluate}_for_"
            f"sample_number_{sample_limit}.pkl"
        )
    elif multiple_metrics:
        scorer = {
            **create_mae_scorer(),
            **create_scorer_for_feature(
                feature="target",
                values=comfort_model.FEEDBACKS,
                default_scorer=mean_absolute_error,
            ),
            **create_scorer_for_feature(
                feature="hour", values=HOURS_OF_DAY, default_scorer=mean_absolute_error
            ),
        }
        fname = (
            f"comfort_model_scorer_for_multiple_metrics_"
            f"for_sample_number_{sample_limit}.pkl"
        )
    else:
        scorer = scoring  # type: ignore
        fname = f"comfort_model_mae_scores_for_sample_number_{sample_limit}.pkl"

    scores = score(
        model, X, y, timestamp, validation_interval, sample_limit, scorer, n_jobs=n_jobs
    )
    save_score(scores, dir_path, fname, feature_to_evaluate, sample_limit)


def create_mae_scorer() -> Dict:
    return {
        "mae": mae_metric_evaluator,
        "weighted_mae": partial(mae_metric_evaluator, weighted=True),
    }


def mae_metric_evaluator(estimator, X, y, weighted=False) -> float:
    try:
        y_pred = estimator.predict(X)
        return mean_absolute_error(
            y,
            y_pred,
            sample_weight=compute_sample_weight("balanced", y) if weighted else None,
        )
    except ValueError as e:
        log.info(f"Not enough data points to predict. Exception message - {e}")
        return float("nan")


def individual_evaluator(estimator, X, y, feature: str, value, default_scorer) -> float:
    samples = X.copy()
    samples["hour"] = samples.apply(hour, axis=1)
    samples["target"] = y
    samples = samples.loc[samples[feature] == value]

    y = samples["target"].tolist()
    try:
        y_pred = estimator.predict(samples)
        return default_scorer(y, y_pred)
    except ValueError as e:
        log.info(f"Not enough data points to predict. Exception message - {e}")
        return float("nan")


def create_scorer_for_feature(
    feature: str, values, default_scorer, evaluator=individual_evaluator
) -> Dict:
    return {
        f"{feature}_{value}": partial(
            evaluator, feature=feature, value=value, default_scorer=default_scorer
        )
        for value in values
    }


@timeit()
def score(model, X, y, timestamps, validation_interval, sample_limit, scoring, n_jobs):

    scores = model.score(
        X,
        y,
        n_jobs=n_jobs,
        cv=get_cv(validation_interval, sample_limit=sample_limit),
        fit_params={"fit__sample_weight": comfort_model.get_sample_weight(y)},
        timestamps=timestamps,
        scoring=scoring,
    )
    return scores


def save_score(
    scores: Dict[str, np.array],
    dir_path: str,
    fname: str,
    feature_to_evaluate: Optional[str],
    sample_limit: int,
) -> None:
    if feature_to_evaluate:
        mae_test = []
        mae_train = []
        for key, value in scores.items():
            if "test" in key:
                mae_test.append(float(np.nanmean(value)))

            elif "train" in key:
                mae_train.append(float(np.nanmean(value)))

        train_mean = np.nanmean(mae_train)
        test_mean = np.nanmean(mae_test)
        scores_df = pd.DataFrame.from_dict(
            {
                "set": ["train", "test"],
                "mae": [train_mean, test_mean],
                "sample_limit": [sample_limit, sample_limit],
            }
        )
        scores_df = scores_df.set_index("set")
        scores_df.to_pickle(f"{dir_path}/{fname}")
    else:
        scores_df = pd.DataFrame(scores).agg(["mean", "std"])
        scores_df.loc["mean"] = np.abs(scores_df.loc["mean"])
        scores_df["sample_limit"] = sample_limit
        log.info(scores_df)
        scores_df.to_pickle(f"{dir_path}/{fname}")


def task_train(
    sample_store,
    model_store,
    until: Optional[str],
    sample_limit: int,
    month_limit: int,
    n_jobs: int,
) -> None:
    log.debug("training comfort model")
    dataset = load_samples(until, month_limit, sample_store, sample_limit)

    create_training_logs(dataset, month_limit)
    train_model(dataset, model_store, n_jobs=n_jobs)


def create_training_logs(dataset, month_limit):
    if month_limit is not None:
        log.info(
            f"Number of samples for previous {month_limit} months of data: "
            f"{len(dataset)}"
        )
    else:
        log.info(f"Number of samples: {len(dataset)}")
    log.info(
        f"Dataset timestamp range: {dataset.iloc[-1]['timestamp']} "
        f"to {dataset.iloc[0]['timestamp']}"
    )
    if is_training_set_older_than_one_day(dataset):
        log.warning(
            "Comfort model training data is outdated. Most recent timestamp is "
            f'{dataset.iloc[0]["timestamp"]}'
        )


def is_training_set_older_than_one_day(dataset):
    return (datetime.utcnow() - dataset.iloc[0]["timestamp"]) > timedelta(hours=24)


@timeit()
def train_model(dataset: pd.DataFrame, model_store, n_jobs: int) -> None:
    model = comfort_model.train(dataset, bypass=TRAIN_WITH_OUTLIERS, n_jobs=n_jobs)
    model_store.save(model.storage_key, model)


def task_grid_search(
    model,
    n_jobs: int,
    sample_limit: int,
    month_limit: int,
    sample_store,
    scoring: str,
    until: Optional[str],
    validation_interval: int,
) -> None:

    dataset, sample_limit = fetch_dataset_based_on_month_limit_and_validation_interval(
        month_limit, sample_limit, sample_store, until, validation_interval
    )

    X, y = comfort_model.split(sample.prepare_dataset(dataset))
    grid = GridSearchCV(
        model,
        comfort_model.estimator.get_params(),
        verbose=3,
        cv=get_cv(validation_interval, sample_limit=sample_limit),
        refit=False,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    grid.fit(
        X, y, groups=get_timestamp(X), sample_weight=comfort_model.get_sample_weight(y)
    )
    grid_scores = sklearn_utils.grid_search_score_dataframe(grid)
    print(grid_scores)
    grid_scores.to_csv("grid_scores_comfort.csv")


def task_predict(
    Model,
    model_store,
    sample_store,
    until: str,
    version: Optional[int],
    dir_path: str,
    multiple_metrics: bool,
) -> None:
    key = {"timestamp": {"$gt": parse(until)}}  # type: ignore
    d_pred = pd.DataFrame(sample_store.get(key, sort=[("timestamp", -1)]))
    X_test = sample.prepare_dataset(d_pred)
    if version is None:
        model_key = Model.get_storage_key()
    else:
        model_key = Model.get_storage_key(model_version=version)
    model = model_store.load(model_key)
    X_test["y_pred"] = model.predict(X_test)
    fname = "pred-{}.pkl".format("-".join(f"{k}={v}" for k, v in model_key.items()))
    X_test.to_pickle(fname)
    log.info(f"saved results to {fname}")

    if multiple_metrics:
        comfort_model.analyze_multiple_metrics(
            X_test, dir_path, baseline_model_version=None
        )


def fetch_dataset_based_on_month_limit_and_validation_interval(
    month_limit, sample_limit, sample_store, until, validation_interval
):
    if month_limit is not None:
        month_limit_including_validation_interval = (
            month_limit + validation_interval // 31 + FETCH_MONTH_BUFFER_OFFSET
        )
        dataset = load_samples(
            until, month_limit_including_validation_interval, sample_store, sample_limit
        )

        dataset = dataset.sort_values("timestamp")
        sample_limit = get_sample_limit_according_to_number_of_months(
            dataset, month_limit, validation_interval
        )
    else:
        dataset = load_samples(until, None, sample_store, 0)
        dataset = dataset.sort_values("timestamp")
    log.info(
        f"Range of timestamp for the dataset: {dataset['timestamp'].min()} to"
        f" {dataset['timestamp'].max()}"
    )
    return dataset, sample_limit


def load_samples(
    until: Optional[str], month_limit: Optional[int], sample_store, sample_limit: int
) -> pd.DataFrame:
    key: Dict[str, Any] = {"type": "user_feedback"}
    timestamp = {}
    if month_limit is not None:
        sample_limit = 0
        fetch_after_timestamp = (
            datetime.utcnow() - relativedelta(months=month_limit)
        ).isoformat()
        timestamp["$gt"] = parse(fetch_after_timestamp)
    if until is not None:
        timestamp["$lt"] = parse(until)

    if timestamp:
        key["timestamp"] = timestamp

    samples = sample_store.get_samples_sorted_by_timestamp_desc(
        key=key, sample_limit=sample_limit
    )
    return samples


def get_cv(validation_interval, sample_limit):
    return sklearn_utils.TimeSeriesFold(
        n_train=sample_limit,
        validation_interval=timedelta(days=validation_interval),
        validation_split_interval=VALIDATION_SPLIT_INTERVAL,
    )


def get_timestamp(X: pd.DataFrame) -> pd.Series:
    return X.reset_index()["timestamp"]


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
