from collections import Counter
from datetime import timedelta
from functools import partial, update_wrapper
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error

from skynet.utils.misc import timeit
from utils.micro_model_offline_metrics import score_and_analyze

from ..sample.selection import INTERPOLATION_INDEX, extrapolate_target
from ..utils import thermo
from ..utils.compensation import ensure_compensated
from ..utils.log_util import get_logger
from . import model
from .estimators.climate_model import get_pipeline

log = get_logger(__name__)

MODEL_TYPE = "climate_model"
CLIMATE_MODEL_VERSION = 18
QUANTITIES = ["humidex", "humidity", "temperature"]
QUANTITY_MAP = {QUANTITIES[i]: i for i in range(len(QUANTITIES))}
MIN_TARGET_POINTS = 8
ANALYSIS_DURATION = 10

# drop columns before fitting (or use as index) if they are in the DataFrame
DROP_COLUMNS = ["timestamp", "appliance_state_id"]
FEATURE_COLUMNS = [
    "appliance_id",
    "mode",
    "mode_hist",
    "power_hist",
    "temperature_set",
    "temperature_set_last",
    "humidex",
    "humidity",
    "temperature_out",
    "temperature",
    "temperature_out_mean_day",
    "previous_temperatures",
]

TRAINING_INTERVAL_SECONDS = 3 * 3600
RELOAD_INTERVAL_SECONDS = TRAINING_INTERVAL_SECONDS / 10
TRAIN_USING_RELATIVE_TARGET = True


def _make_metric(x, y, metric, idx):
    return metric(getattr(x, "iloc", x)[:, idx], getattr(y, "iloc", y)[:, idx])


def climate_scoring(metric):
    scoring = {}
    for quantity, idx in QUANTITY_MAP.items():
        score_fun = partial(_make_metric, metric=metric, idx=idx)
        update_wrapper(score_fun, _make_metric)
        scoring[quantity] = make_scorer(score_fun)
    return scoring


def aggregate_scores(scores):
    df = pd.DataFrame(pd.DataFrame(scores).mean(), columns=["mean"])
    df[["set", "quantity"]] = pd.Series(df.index, index=df.index).str.split(
        "_", expand=True
    )
    df = df[df["set"].isin(["test", "train"])].pivot_table(
        values="mean", columns="quantity", index="set"
    )
    return df


class ClimateModel(model.Model):
    """Climate model."""

    default_estimator = get_pipeline()

    @classmethod
    def get_storage_key(
        cls, model_type=MODEL_TYPE, model_version=CLIMATE_MODEL_VERSION
    ):
        key = super().get_storage_key(model_type, model_version)
        return key

    def __init__(
        self, estimator=None, model_type=MODEL_TYPE, model_version=CLIMATE_MODEL_VERSION
    ):  # pylint: disable=useless-super-delegation
        """Model for Quantity Prediction.

        Parameters
        ----------
        quantity : str
            the quantity we're predicting with this model
        """
        super().__init__(estimator, model_type, model_version)

    @staticmethod
    def get_features(X):

        X = X.copy()

        for col in DROP_COLUMNS:
            if col in X:
                X.drop(col, axis=1, inplace=True)
        return X[FEATURE_COLUMNS]

    @ensure_compensated
    def fit(self, X, y, *args, **kwargs):
        return super().fit(self.get_features(X), y, *args, **kwargs)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = self.get_features(X)
        return super().predict(X)

    def score(self, X, y, *args, **kwargs):
        scores = super().score(self.get_features(X), y, *args, **kwargs)
        return aggregate_scores(scores)


def filter_targets_before_initial_time(X, y, initial_time):
    y_all = y.reset_index()
    delta_time = y_all["timestamp"].values - pd.to_datetime(
        X["timestamp"].loc[y_all["sample_id"]]
    )
    static_target_indices = np.where(delta_time >= timedelta(minutes=initial_time))
    y_long = y_all.iloc[static_target_indices]
    sample_ids = np.unique(y_long["sample_id"])
    return X.loc[sample_ids], y_long


def make_static_mode_dataset(X, y):
    X_static, y_long = filter_targets_before_initial_time(X, y, initial_time=15)
    y_static = y_long.groupby("sample_id").mean()
    y_static = y_static[QUANTITIES]

    # use differences for learning
    y_static -= X_static[y_static.columns]
    return X_static, y_static


def apply_parallel_extrapolation(df_grouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, df_grouped)
    return pd.concat(ret_list)


def get_nth_element(df):
    return df.groupby("sample_id")[QUANTITIES].nth(INTERPOLATION_INDEX)


def func_group_apply(df):
    return df.groupby("sample_id")[QUANTITIES].apply(extrapolate_target)


def chunk_generator(list_to_chunk, n):
    return list(
        zip(
            list(np.linspace(0, len(list_to_chunk), n + 1, dtype=int)),
            list(np.linspace(0, len(list_to_chunk), n + 1, dtype=int))[1:],
        )
    )


@timeit()
def make_static_climate_dataset(X, y):
    X, y = downcast_float(X), downcast_float(y)
    sensor_count_per_climate_sample = y.groupby("sample_id").size()
    y_indexed = get_sample_at_interpolation_index(y, sensor_count_per_climate_sample)
    y_extrapolated = get_extrapolated_sample(y, sensor_count_per_climate_sample)
    y_static = pd.concat([y_indexed, y_extrapolated])

    lens = len(y_static)
    y_static = y_static[
        y_static["humidity"].between(0, 100, inclusive=False)
        & y_static["temperature"].between(0, 50, inclusive=False)
    ]
    log.info(f"{lens/len(y)}% samples good enough for training")
    log.info(f"removed {(lens - len(y_static))/lens*100}% spurious samples")

    X_static = X.loc[y_static.index]
    # use differences for learning
    if TRAIN_USING_RELATIVE_TARGET:
        y_static -= X_static[y_static.columns]
    return X_static, y_static


def downcast_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def get_sample_at_interpolation_index(
    y: pd.DataFrame, sample_length: pd.Series
) -> pd.DataFrame:
    y_long_index = sample_length[sample_length > INTERPOLATION_INDEX]
    y_good_samples = subset_samples(y, y_long_index)
    return apply_parallel_extrapolation(y_good_samples, get_nth_element)


def get_extrapolated_sample(y: pd.DataFrame, sample_length: pd.Series) -> pd.DataFrame:
    to_extrapolate_index = sample_length[
        (sample_length > MIN_TARGET_POINTS) & (sample_length <= INTERPOLATION_INDEX)
    ]
    log.info(
        f"{len(to_extrapolate_index)/len(sample_length)*100}"
        "% samples need extrapolation"
    )
    log.info("computing static targets via linear fit (takes a few minutes)")
    y_good_samples = subset_samples(y, to_extrapolate_index)
    return apply_parallel_extrapolation(y_good_samples, func_group_apply)


def subset_samples(df: pd.DataFrame, on: pd.Series) -> List[pd.DataFrame]:
    result = []
    for start, end in chunk_generator(on.index, cpu_count()):
        result.append(df.loc[on.index[start:end]])
    log.info("Length of df per core: " f"{', '.join([str(len(i)) for i in result])}")
    return result


def prepare_dataset(X):
    """Overwrite non impactful set temperatures."""
    non_active_modes = ["off", "fan"]
    non_active_idx = X["mode"].isin(non_active_modes)
    X["temperature_set"] = thermo.fix_temperatures(X["temperature_set"])
    active_idx = ~non_active_idx
    mean = np.mean(X.loc[active_idx, "temperature_set"])
    std = np.std(X.loc[active_idx, "temperature_set"])
    log.info(
        "Overwriting non-impactful set temperature from " f"N({mean:.2f}, {std:.2f})"
    )
    X.loc[non_active_idx, "temperature_set"] = np.random.normal(
        loc=mean, scale=std, size=np.count_nonzero(non_active_idx)
    )
    return X


def create_average_scorer_for_different_metrics(quantity_map, metrics_to_evaluate):
    return {
        f"{target}--{name}": partial(
            score_creator,
            target=target,
            index_of_target=index,
            default_scorer=scoring_function,
        )
        for target, index in quantity_map.items()
        for name, (score_creator, scoring_function) in metrics_to_evaluate.items()
    }


def get_metrics_to_evaluate():
    return {
        "mae": (mae_metric_evaluator, mean_absolute_error),
        "noise": (derivative_evaluator, absolute_average),
        "inversions": (metrics_evaluator, average_inversions),
        "range": (metrics_evaluator, average_range),
        "minstepsize": (metrics_evaluator, average_min_step_size),
        "maxstepsize": (metrics_evaluator, average_max_step_size),
    }


def mae_metric_evaluator(estimator, X, y, target, index_of_target, default_scorer):
    try:
        y_pred = estimator.predict(X)
        score = default_scorer(y[target], y_pred[:, index_of_target])
    except ValueError as e:
        log.info(f"Not enough data points to predict. Exception message - {e}")
        score = float("nan")

    return score


def derivative_evaluator(
    estimator,
    X,
    y,  # pylint:disable=unused-argument
    target,
    index_of_target,
    default_scorer,
):

    if target == "humidex":
        epsilon = 0.2
    elif target == "temperature":
        epsilon = 0.1
    else:
        epsilon = 1

    try:
        y_pred = estimator.predict(X)[:, index_of_target]

        X[target] += epsilon
        y_pred_plus_epsilon = estimator.predict(X)[:, index_of_target]

        score = default_scorer((y_pred_plus_epsilon - y_pred) / epsilon)

    except Exception as e:
        log.error(
            f"Exception was raised. Check derivative evaluator and {default_scorer.__name__}- {e}"
        )
        score = float("nan")

    return score


def absolute_average(X: np.ndarray):
    return np.mean(np.abs(X))


def metrics_evaluator(
    estimator,
    X,
    y,  # pylint:disable=unused-argument
    target,  # pylint:disable=unused-argument
    index_of_target,
    default_scorer,
):
    possible_set_temperatures = pd.DataFrame(
        list(range(16, 32)), columns=["temperature_set"]
    )

    # Subset X for modes in which set temperature makes sense
    X = X.loc[
        (X["mode"] == "cool") | (X["mode"] == "heat") | (X["mode"] == "dry")
    ].copy()

    X.loc[:, "sample_key"] = X.index

    augmented_X = cartesian_product(X, possible_set_temperatures)

    try:
        augmented_X["prediction"] = estimator.predict(
            augmented_X.drop(columns=["sample_key"])
        )[:, index_of_target]

        score = default_scorer(augmented_X)

    except Exception as e:
        log.error(
            f"Exception was raised. Check metrics evaluator and {default_scorer.__name__}- {e}"
        )
        score = float("nan")

    return score


def cartesian_product(X, possible_set_temperatures):
    augmented_X = pd.merge(
        possible_set_temperatures.assign(outer_join_key=0),
        X.drop(columns=["temperature_set"]).assign(outer_join_key=0),
        how="outer",
        on="outer_join_key",
    ).drop(columns=["outer_join_key"])
    return augmented_X


def average_inversions(X):
    return (
        X.groupby("sample_key")["prediction"]
        .apply(lambda y: (y.shift(1) > y).mean())
        .mean()
    )


def average_range(X):
    return (
        X.groupby(["sample_key"])["prediction"]
        .apply(lambda x: x.max() - x.min())
        .mean()
    )


def average_min_step_size(X):
    return (
        X.groupby(["sample_key"])["prediction"]
        .apply(lambda y: y.sort_values().diff().min())
        .mean()
    )


def average_max_step_size(X):
    return (
        X.groupby(["sample_key"])["prediction"]
        .apply(lambda y: y.sort_values().diff().max())
        .mean()
    )


def analyze_multiple_metrics(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    path: str,
    current_model_version: int,
    baseline_model_version: Optional[int] = None,
) -> None:
    dataframes = {
        f"NEW_MODEL_{current_model_version}": get_dataframe_for_current_model(
            estimator, X_test, X_train, y_test, path, current_model_version
        )
    }
    baseline = get_baseline_dataframe(baseline_model_version, path)
    if baseline is not None:
        dataframes[f"BASELINE_{baseline_model_version}"] = baseline

    score_and_analyze(dataframes, path)


def get_dataframe_for_current_model(
    estimator,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_test: pd.DataFrame,
    path: str,
    current_model_version: int,
):
    try:
        X = pd.read_pickle(f"{path}/augmented_X_version_{current_model_version}.pkl")
    except FileNotFoundError:
        log.info(
            f"Dataframe doesn't exist for version number - {current_model_version}."
            f"Creating new dataframe."
        )
        X = create_augmented_dataframe(
            estimator, X_train, X_test, y_test, path, current_model_version
        )
    return X


def get_baseline_dataframe(
    model_version: Optional[int], path: str
) -> Optional[pd.DataFrame]:
    try:
        return (
            pd.read_pickle(f"{path}/augmented_X_version_{model_version}.pkl")
            if model_version
            else None
        )
    except FileNotFoundError:
        log.info(
            f"Dataframe doesn't exist for baseline version number - {model_version}."
        )
        return None


def create_augmented_dataframe(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    path: str,
    current_model_version: int,
) -> pd.DataFrame:
    X_test, y_test = limit_to_timestamp(
        X_test,
        y_test,
        timestamp=X_test["timestamp"].min() + timedelta(days=ANALYSIS_DURATION),
    )

    augmented_X = predict_for_boosted_set_temperatures(estimator, X_test)
    X_test["mae"] = absolute_mae(estimator.predict(X_test), y_test)
    X_test["number_of_samples"] = samples_per_appliance_id(
        appliance_id_df=X_train, indexing=X_test
    )
    X = append_features_to(augmented_X, features_from=X_test)

    X.to_pickle(f"{path}/augmented_X_version_{current_model_version}.pkl")
    return X


def limit_to_timestamp(
    X: pd.DataFrame, y: pd.DataFrame, timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = X.loc[X["timestamp"] < timestamp].copy()
    y = y.loc[X.index]
    return X, y


def predict_for_boosted_set_temperatures(estimator, X: pd.DataFrame):
    return metrics_evaluator(
        estimator, X, None, None, QUANTITY_MAP["temperature"], identity
    )


def absolute_mae(y_pred: pd.Series, y_test: np.array) -> pd.Series:
    return (y_test["temperature"] - y_pred[:, QUANTITY_MAP["temperature"]]).abs()


def samples_per_appliance_id(
    appliance_id_df: pd.DataFrame, indexing: pd.DataFrame
) -> np.array:
    c: Dict["str", int] = Counter(appliance_id_df.appliance_id)
    return np.array([c[id_] for id_ in indexing.appliance_id])


def append_features_to(X: pd.DataFrame, features_from: pd.DataFrame) -> pd.DataFrame:
    features_from.rename(
        columns={"temperature_set": "original_temperature_set"}, inplace=True
    )
    return pd.merge(
        X,
        features_from[
            [
                "appliance_id",
                "mode",
                "timestamp",
                "mae",
                "number_of_samples",
                "original_temperature_set",
            ]
        ],
        on=["appliance_id", "mode", "timestamp"],
        how="inner",
    )


def identity(X):
    return X
