"""Comfort Model. Learn user's likely comfort for given climate."""
from datetime import datetime, timedelta, tzinfo
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from utils import comfort_model_offline_metrics

from ..prediction import model
from ..prediction.estimators import comfort_model_estimator as estimator
from ..user.sample import (
    COMFORT_TIMESERIES_INTERVAL,
    USE_COMFORT_LSTM,
    WEATHER_AVERAGE_INTERVAL,
    prepare_non_feedback_features,
)
from ..utils import cache_util, misc
from ..utils.async_util import multi
from ..utils.compensation import COMPENSATE_COLUMN, ensure_compensated
from ..utils.config import MAX_COMFORT, MIN_COMFORT
from ..utils.database import result_format
from ..utils.log_util import get_logger
from ..utils.types import ComfortPrediction, Connections
from . import sample

log = get_logger(__name__)

COMFORT_MODEL_VERSION = 14
COMFORT_MODEL_VERSION += 1 if USE_COMFORT_LSTM else 0
MODEL_TYPE = "comfort_model"
FEEDBACKS = np.arange(-3.0, 4.0)

DECAY_TIME = timedelta(hours=4)

TRAINING_INTERVAL_SECONDS = 3600 * 3
RELOAD_INTERVAL_SECONDS = TRAINING_INTERVAL_SECONDS / 10


def train(dataset, bypass=True, **_):
    model = ComfortModel()
    model.set_params(estimator__filter__bypass=bypass)
    dataset = sample.prepare_dataset(dataset)
    model.fit(*split(dataset), sample_weight=get_sample_weight(dataset["feedback"]))

    if model.estimator.steps[0][0] == "filter":
        inlier_mask = model.estimator.steps[0][-1].inlier_mask
        log.debug(
            "Sample Filter take {} samples and "
            "return {} samples".format(len(dataset), np.sum(inlier_mask))
        )
    return model


def split(dataset):
    X = dataset[sample.COMFORT_FEATURES_TRAIN + [COMPENSATE_COLUMN]]
    # NaNs are from static samples, set those to feedback=0, i.e. comfortable
    y = dataset[sample.COMFORT_MODEL_TARGET].fillna(0)
    return X, y


def get_sample_weight(y: pd.Series) -> np.ndarray:
    return compute_sample_weight("balanced", y.values)


class ComfortModel(model.Model):

    default_estimator = estimator.get_pipeline()

    @classmethod
    def get_storage_key(
        cls, model_type=MODEL_TYPE, model_version=COMFORT_MODEL_VERSION
    ):
        return super().get_storage_key(model_type, model_version)

    def __init__(
        self, estimator=None, model_type=MODEL_TYPE, model_version=COMFORT_MODEL_VERSION
    ):  # pylint: disable=useless-super-delegation
        super().__init__(estimator, model_type, model_version)

    @ensure_compensated
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, fit__sample_weight=sample_weight)

    @misc.timeit()
    async def get_adjusted_comfort_prediction(
        self,
        connections: Connections,
        device_id: str,
        user_id: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        prediction = await self.get_predicted_comfort(
            connections, device_id, user_id, timestamp
        )
        log_msg = {"user_id": user_id, "prediction": prediction, "device_id": device_id}
        comfort_prediction = await cache_util.get_comfort_prediction(
            redis=connections.redis, key_arg=(device_id, user_id)
        )
        if comfort_prediction:
            prediction = get_adjusted_prediction(
                prediction, timestamp, comfort_prediction
            )
        else:
            log.info(
                "could not fetch adjustment data, skipping adjustment",
                extra={"data": {"user_id": user_id, "device_id": device_id}},
            )
        prediction = clip_feedback_prediction(prediction)
        log_msg["adjusted_prediction"] = prediction
        log.info("feedback prediction", extra={"data": log_msg})

        return {"created_on": timestamp, "comfort": prediction}

    async def get_predicted_comfort(
        self,
        connections: Connections,
        device_id: str,
        user_id: str,
        timestamp: datetime,
    ) -> float:

        features = await fetch_features_for_prediction(
            connections, device_id, user_id, timestamp
        )
        prediction = self.predict_one(features)

        return prediction

    def predict_one(self, x):
        return self.predict([x])[0]


def get_adjusted_prediction(
    prediction: float, timestamp: datetime, comfort_prediction: ComfortPrediction
) -> float:
    """Adjusts prediction based on latest feedback (the magnitude decay over time).

    Args:
        prediction:
        timestamp:
        comfort_prediction:

    Returns:

    """

    feedback_delta = (
        comfort_prediction.feedback - comfort_prediction.feedback_prediction
    )
    time_decay_delta = exponential_time_decay(
        comfort_prediction.feedback_timestamp, timestamp, DECAY_TIME, feedback_delta
    )
    return prediction + time_decay_delta


def exponential_time_decay(
    t0: datetime, t: datetime, tau: timedelta, coeff: float = 1
) -> float:
    elapsed = t - t0
    ratio = elapsed.total_seconds() / tau.total_seconds()
    return coeff * np.exp(-ratio)


def clip_feedback_prediction(
    prediction: float,
    min_comfort: float = MIN_COMFORT,
    max_comfort: float = MAX_COMFORT,
) -> float:
    return np.clip(prediction, min_comfort, max_comfort)


async def fetch_features_for_prediction(
    connections: Connections, device_id: str, user_id: str, timestamp: datetime
) -> Dict[str, Any]:
    timestamp = datetime.utcnow()
    data = await multi(
        {
            "sensors": cache_util.get_sensors_range(
                connections.redis,
                device_id,
                start=timestamp - COMFORT_TIMESERIES_INTERVAL,
                end=timestamp,
            ),
            "weather": cache_util.fetch_weather_from_device(
                connections,
                device_id,
                start=timestamp - WEATHER_AVERAGE_INTERVAL,
                end=timestamp,
            ),
            "timezone": get_timezone_wrap(connections, device_id),
        }
    )

    features = prepare_non_feedback_features(
        timestamp=timestamp, device_id=device_id, prediction=True, **data
    )
    features["user_id"] = user_id

    # TODO - Properly fix this for both sample filtering and comfort service
    for feature in sample.COMFORT_FEATURES_TRAIN:
        if feature not in features:
            features[feature] = np.nan

    return features


async def get_timezone_wrap(connections: Connections, device_id: str) -> tzinfo:
    timezone = await cache_util.fetch_timezone(connections, device_id)
    return result_format.parse_timezone(timezone)


def analyze_multiple_metrics(
    X_test: pd.DataFrame, path: str, baseline_model_version: Optional[int] = None
):
    save_dataframe(X_test, path)
    dataframes = {f"NEW_MODEL_{COMFORT_MODEL_VERSION}": X_test}

    baseline = get_baseline_dataframe(baseline_model_version, path)
    if baseline is not None:
        dataframes[f"BASELINE_{baseline_model_version}"] = baseline

    comfort_model_offline_metrics.generate_plots(dataframes, path)


def save_dataframe(X: pd.DataFrame, path: str):
    X.to_pickle(f"{path}/X_version_{COMFORT_MODEL_VERSION}.pkl")


def get_baseline_dataframe(
    model_version: Optional[int], path: str
) -> Optional[pd.DataFrame]:
    if not model_version:
        return None
    try:
        return pd.read_pickle(f"{path}/X_version_{model_version}.pkl")
    except FileNotFoundError:
        log.info(
            f"Dataframe doesn't exist for baseline version number - {model_version}."
        )
        return None
