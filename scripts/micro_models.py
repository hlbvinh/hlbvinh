import importlib
from functools import partial
from datetime import datetime, timedelta
from pathlib import Path

import skynet.utils.openblas_thread_hack  # noqa pylint:disable=unused-import

# pylint: disable=wrong-import-order
import click
import pandas as pd
import yaml
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from joblib import Memory
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from typing import List

# pylint: enable=wrong-import-order

from skynet.prediction import climate_model
from skynet.sample import climate_sample_store
from skynet.utils import sklearn_utils
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.storage import get_storage
from skynet.utils.script_utils import (
    get_connections,
    get_sample_limit_according_to_number_of_months,
)

from utils.micro_model_offline_metrics import (
    create_offline_metric_graphs_and_graph_characteristics,
)

log = get_logger("skynet")
VALIDATION_SPLIT_INTERVAL = "24H"
AVG_PARAMS = {
    7: {"average_interval": "24H", "rolling_window": "84H"},
    14: {"average_interval": "24H", "rolling_window": "168H"},
    90: {"average_interval": "24H", "rolling_window": "336H"},
    365: {"average_interval": "24H", "rolling_window": "336H"},
}
N_MONTHS = 3
FETCH_MONTH_BUFFER_OFFSET = 1

OVERALL_SCORE_FILENAME = "climate_model_mae_scores_for_month_number"
MULTIPLE_SCORER_FILENAME = "climate_model_scorer_for_multiple_metrics"
PER_COLUMN_SCORE_FILENAME = "climate_model_mae_score_per"


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--grid_module",
    default="skynet.prediction.estimators.climate_model",
    help="grid search pipeline and parameters to be used.",
)
@click.option("--mongo", default="production")
@click.option(
    "--task",
    default="train",
    type=click.Choice(["grid_search", "score", "train", "predict"]),
)
@click.option(
    "--until", default=None, help="fetch samples until timestamp YYYY-MM-DD:HH:MM"
)
@click.option("--cache", default=False, is_flag=True)
@click.option(
    "--multiple_metrics",
    default=False,
    is_flag=True,
    help="Evaluate metrics: MAE, max step size, min step size, range, inversions"
    " for climate model. Can be used with scoring to get numerical score of "
    "metrics. Can be used with prediction to get graphical representation of metrics.",
)
@click.option(
    "--mae_per_column", default=None, type=click.Choice(["mode", "appliance_id"])
)
@click.option(
    "--validation_interval",
    default="7",
    type=click.Choice(str(a) for a in sorted(AVG_PARAMS)),
)
@click.option(
    "--n_months",
    default=N_MONTHS,
    type=int,
    help="Fetch samples after current timestamp minus n_months.",
)
@click.option("--log_directory", default="log")
@click.option(
    "--prediction_for_months",
    default=1,
    type=int,
    help="Predict for given number of end months.",
)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
@click.option("--save_to_directory", default="./data/climate")
@click.option("--n_jobs", type=int, default=-1)
def main(
    config,
    grid_module,
    mongo,
    task,
    until,
    multiple_metrics,
    mae_per_column,
    cache,
    validation_interval,
    n_months,
    log_directory,
    prediction_for_months,
    storage,
    save_to_directory,
    n_jobs,
):
    """Computes any number of useful things"""
    Path(save_to_directory).mkdir(parents=True, exist_ok=True)
    validation_interval = int(validation_interval)
    until = parse(until) if until is not None else None
    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config("micro_model", cnf=cnf, log_directory=log_directory)

    sample_store = climate_sample_store.ClimateSampleStore(
        get_connections(cnf, mongo=mongo).mongo
    )
    mod = importlib.import_module(grid_module)
    estimator = mod.get_pipeline()

    if cache:
        maybe_cache = Memory(cachedir="cache", verbose=2).cache
    else:

        def maybe_cache(fun):
            return fun

    @maybe_cache
    def fetch(n_months):

        if task == "train":
            fetching_timestamp = datetime.utcnow() - relativedelta(months=n_months)
        elif task == "predict":
            fetching_timestamp = (
                datetime.utcnow()
                - relativedelta(months=n_months)
                - relativedelta(months=prediction_for_months)
                - relativedelta(months=FETCH_MONTH_BUFFER_OFFSET)
            )
        else:
            fetching_timestamp = (
                datetime.utcnow()
                - relativedelta(months=n_months)
                - relativedelta(days=validation_interval)
                - relativedelta(months=FETCH_MONTH_BUFFER_OFFSET)
            )

        log.info(f"Fetching range from {fetching_timestamp} to {datetime.utcnow()}")

        timestamp = {"$gt": fetching_timestamp}

        if until:
            timestamp["$lt"] = until

        key = {"timestamp": timestamp}

        log.info("fetching samples ...")
        feats, targs = climate_sample_store.get_climate_samples(
            sample_store, key=key, limit=0
        )
        log.info(f"fetched {len(feats)} samples")

        X, y = climate_model.make_static_climate_dataset(feats, targs)
        y = y[climate_model.QUANTITIES]
        X = climate_model.prepare_dataset(X)
        log.info(f"generated {len(X)} static samples")
        return X, y

    X, y = fetch(n_months)

    def get_cv(X, n_months, validation_interval):

        X = X.sort_values("timestamp")
        sample_limit_for_n_months = get_sample_limit_according_to_number_of_months(
            X, n_months, validation_interval
        )
        return sklearn_utils.TimeSeriesFold(
            n_train=sample_limit_for_n_months,
            validation_interval=timedelta(days=validation_interval),
            validation_split_interval=VALIDATION_SPLIT_INTERVAL,
        )

    if task == "train":
        model_store = get_storage(
            storage, **cnf["model_store"], directory="data/models"
        )
        m = climate_model.ClimateModel(estimator=estimator)
        m.fit(X, y)
        m.save(storage=model_store)

    elif task == "grid_search":
        m = climate_model.ClimateModel(estimator=estimator)
        grid_search = GridSearchCV(
            m,
            mod.get_params(),
            verbose=3,
            cv=get_cv(X, n_months, validation_interval),
            refit=False,
            n_jobs=n_jobs,
            scoring="mean_absolute_error",
        )
        grid_search.fit(X, y, groups=X["timestamp"])
        grid_scores = sklearn_utils.grid_search_score_dataframe(
            grid_search,
            VALIDATION_SPLIT_INTERVAL,
            AVG_PARAMS[validation_interval]["average_interval"],
            AVG_PARAMS[validation_interval]["rolling_window"],
        )
        print(grid_scores)
        grid_scores.to_csv(f"{save_to_directory}/grid_scores_climate.csv")

    elif task == "score":
        m = climate_model.ClimateModel(estimator=estimator)

        if mae_per_column:
            scorer = create_scorer_for_all_targets_and_all_possible_values_of_column(
                X,
                climate_model.QUANTITY_MAP,
                column=mae_per_column,
                default_scorer=mean_absolute_error,
            )
            fname = f"{PER_COLUMN_SCORE_FILENAME}_{mae_per_column}_for_month_number_{n_months}.csv"
        elif multiple_metrics:
            multiple_metrics = climate_model.get_metrics_to_evaluate()
            scorer = climate_model.create_average_scorer_for_different_metrics(
                quantity_map={
                    k: v
                    for k, v in climate_model.QUANTITY_MAP.items()
                    if k == "temperature"
                },
                metrics_to_evaluate=multiple_metrics,
            )
            fname = (
                f"{MULTIPLE_SCORER_FILENAME}_model_"
                f"number_{climate_model.CLIMATE_MODEL_VERSION}.csv"
            )
        else:
            scorer = climate_model.climate_scoring(mean_absolute_error)
            fname = f"{OVERALL_SCORE_FILENAME}_{n_months}.csv"

        # calculating training score takes time if done for multiple metrics
        scores = m.score(
            X,
            y,
            cv=get_cv(X, n_months, validation_interval),
            n_jobs=n_jobs,
            timestamps=X["timestamp"],
            scoring=scorer,
            return_train_score=not multiple_metrics,
        )
        print(scores)
        scores.to_csv(f"{save_to_directory}/{fname}")

    elif task == "predict":
        m = climate_model.ClimateModel(estimator=estimator)
        X_train, y_train, X_test, y_test = train_test_split(
            X, y, prediction_for_months, n_months
        )
        log.info(
            f"Range train: {X_train['timestamp'].min()} - {X_train['timestamp'].max()}"
        )
        log.info(
            f"Range test: {X_test['timestamp'].min()} - {X_test['timestamp'].max()}"
        )
        m.fit(X_train, y_train)
        X_test = append_output_error_to(X_test, output_error=m.predict(X_test) - y_test)
        if multiple_metrics:
            climate_model.analyze_multiple_metrics(
                m,
                X_train,
                X_test,
                y_test,
                save_to_directory,
                current_model_version=n_months,
                baseline_model_version=None,
            )
        create_offline_metric_graphs_and_graph_characteristics(
            X_test, save_to_directory
        )


def append_output_error_to(
    X_test: pd.DataFrame, output_error: pd.DataFrame = pd.DataFrame()
) -> pd.DataFrame:
    X_test["error_humidex"] = output_error["humidex"]
    X_test["error_humidity"] = output_error["humidity"]
    X_test["error_temperature"] = output_error["temperature"]

    return X_test


def train_test_split(
    X: pd.DataFrame, y: pd.DataFrame, prediction_for_months: int, n_months: int
) -> List[pd.DataFrame]:
    recent_timestamp = X["timestamp"].max()
    X_train = X.loc[
        (
            X["timestamp"]
            < recent_timestamp - relativedelta(months=prediction_for_months)
        )
        & (
            X["timestamp"]
            > recent_timestamp - relativedelta(months=n_months + prediction_for_months)
        )
    ]
    y_train = y.loc[X_train.index]
    X_test = X.loc[
        X["timestamp"] >= recent_timestamp - relativedelta(months=prediction_for_months)
    ]
    y_test = y.loc[X_test.index]
    return [X_train, y_train, X_test, y_test]


def create_scorer_for_all_targets_and_all_possible_values_of_column(
    features, QUANTITY_MAP, column="mode", default_scorer=mean_absolute_error
):
    scorer_dict = {}
    for target, index in QUANTITY_MAP.items():
        for value in features[column].unique():
            scorer_dict[f"{target}--{value}"] = partial(
                individual_scorer,
                target=target,
                value=value,
                column=column,
                index_of_target=index,
                default_scorer=default_scorer,
            )
    return scorer_dict


def individual_scorer(
    estimator, X, y, target, value, column, index_of_target, default_scorer
):

    X = X.loc[X[column] == value]
    y = y.loc[X.index.values]
    try:
        y_pred = estimator.predict(X)
        score_per_value_of_the_column = default_scorer(
            y[target], y_pred[:, index_of_target]
        )
    except ValueError as e:
        log.info(f"Not enough data points to predict. Exception message - {e}")
        score_per_value_of_the_column = float("nan")

    return score_per_value_of_the_column


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
