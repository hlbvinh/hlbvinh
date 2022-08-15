from datetime import datetime, tzinfo
from operator import itemgetter
from typing import Dict, List, Optional

import numpy as np
from ambi_utils.zmq_micro_service.zmq_actor import AircomRequest, DealerActor

from ..prediction import prediction_service
from ..prediction.climate_model import QUANTITY_MAP, QUANTITIES
from ..sample import selection
from ..sample.feature_cache import RedisFeatureData
from ..user.comfort_model import get_adjusted_prediction
from ..user.sample import (
    COMFORT_TIMESERIES_FREQ,
    COMFORT_TIMESERIES_INTERVAL,
    TIMESERIES_FEATURES,
    USE_COMFORT_LSTM,
)
from ..utils.async_util import PREDICTION_TIMEOUT, multi, request_with_timeout
from ..utils.progressive_rollout import is_in
from ..utils.cache_util import get_comfort_prediction
from ..utils.types import Connections, Feedback, UserFeatures, UserIdSet
from . import util

ClimatePrediction = List[float]
ClimatePredictions = List[ClimatePrediction]

# FOCUS [0, 1) controls the importance of recency in the feedback weights calculation
# When FOCUS = 0, all feedbacks have the same weight
# When FOCUS -> 1, only the most recent feedback matters
FOCUS = 0
# INTENSITY (0, 1] controls the initial magnitude of the feedback weights
# When INTENSITY = 1, the initial weight of the most recent feedback is at its maximum value
# When INTENSITY -> 0, the values of the feedback weights will move closer to each other
INTENSITY = 1


class Comfort(util.LogMixin):
    """Determine comfort score of user(s) and also other related stuffs (e.g. feedbacks)."""

    def __init__(
        self,
        device_id: str,
        timezone: tzinfo,
        prediction_clients: Dict[str, DealerActor],
        feature_data: RedisFeatureData,
        nearby_users: UserIdSet,
        latest_feedbacks: List[Feedback],
        connections: Connections,
        experiments=[],
    ):
        self.device_id = device_id
        self.timezone = timezone
        self.prediction_clients = prediction_clients
        self.feature_data = feature_data
        self.nearby_users = nearby_users
        self.latest_feedbacks = latest_feedbacks
        self.connections = connections
        self.is_in = is_in("average_humidity", experiments)

    async def predict(self) -> float:
        """Predicts comfort score of a user.

        Returns:

        """
        return (await self._predict([self.user_features()]))[0]

    async def _predict(self, features: List) -> List[float]:
        """Predicts comfort score.

        Prediction can be made in single request, we get List[feature] -> List[score].

        Args:
            features:

        Returns:

        """
        req = prediction_service.get_comfort_model_request(features)
        return (await self.prediction_request("comfort_model", req))["prediction"]

    def prediction_request(self, model: str, req: AircomRequest):
        return request_with_timeout(
            PREDICTION_TIMEOUT, self.prediction_clients[model], req
        )

    async def predict_adjusted(self) -> float:
        """Predicts comfort scores for nearby users and then averaging the scores.

        Returns:

        """
        predictions = await self._predict_for_nearby_users()
        adjusted_predictions = await self._adjust_predictions(
            predictions, self.nearby_users_ids
        )
        self._log(predictions, adjusted_predictions)
        return self.multi_user_predictions_average(adjusted_predictions)

    async def _predict_for_nearby_users(self) -> List[float]:
        return await self._predict(
            [self.single_user_features(f) for f in self.nearby_users_feedbacks]
        )

    async def _adjust_predictions(
        self, predictions: List[float], user_ids: List[str]
    ) -> List[float]:
        """Fetches recent feedbacks and adjusts prediction results.

        Args:
            predictions:
            user_ids:

        Returns:

        """
        last_feedback_prediction = await multi(
            {
                user_id: get_comfort_prediction(
                    redis=self.connections.redis, key_arg=(self.device_id, user_id)
                )
                for user_id in set(user_ids)
            }
        )
        return [
            get_adjusted_prediction(
                prediction, datetime.utcnow(), last_feedback_prediction[user_id]
            )
            if last_feedback_prediction[user_id]
            else prediction
            for prediction, user_id in zip(predictions, user_ids)
        ]

    def multi_user_predictions_average(self, predictions: List[float]) -> float:
        """Averages score based on how recent feedbacks is given.

        Args:
            predictions:

        Returns:

        """
        return np.average(
            predictions, weights=_get_weights(self.nearby_users_feedbacks)
        )

    async def from_climate_predictions(
        self, predictions: ClimatePredictions
    ) -> List[float]:
        comfort_predictions = await self._predict(
            [
                self._user_features_from_climate_prediction(p, f)
                for p in predictions
                for f in self.nearby_users_feedbacks
            ]
        )
        adjusted_predictions = await self._adjust_predictions(
            comfort_predictions, self.nearby_users_ids * len(predictions)
        )
        self._log_climate(predictions, comfort_predictions, adjusted_predictions)
        return self._average_over_nearby_users(adjusted_predictions)

    def single_user_features(
        self, feedback: Feedback, timestamp: Optional[datetime] = None
    ) -> UserFeatures:
        return {
            "user_id": feedback.get("user_id"),
            "device_id": self.device_id,
            **self.feature_data.get_user_features(
                self.timezone, timestamp if timestamp else datetime.utcnow(), self.is_in
            ),
        }

    def _user_features_from_climate_prediction(
        self, prediction: ClimatePrediction, feedback
    ):
        features = self.single_user_features(
            feedback, datetime.utcnow() + selection.STATIC_INTERPOLATION
        )
        features.update(dict(zip(QUANTITIES, prediction)))
        if USE_COMFORT_LSTM:
            features.update(
                dict(zip(TIMESERIES_FEATURES, fitted_timeseries(features, prediction)))
            )
        return features

    def _average_over_nearby_users(self, predictions):
        return list(
            map(
                self.multi_user_predictions_average,
                zip(*[iter(predictions)] * len(self.nearby_users_feedbacks)),
            )
        )

    @property
    def nearby_users_feedbacks(self) -> List[Feedback]:
        feedbacks = [
            f for f in self.latest_feedbacks if f["user_id"] in self.nearby_users
        ]

        return feedbacks or [self.latest_feedback]

    @property
    def nearby_users_ids(self):
        return [f.get("user_id") for f in self.nearby_users_feedbacks]

    @property
    def latest_feedback(self) -> Feedback:
        if not self.latest_feedbacks:
            return {}
        return sorted(self.latest_feedbacks, key=itemgetter("created_on"))[-1]

    def user_features(self, timestamp: Optional[datetime] = None) -> UserFeatures:
        return self.single_user_features(self.latest_feedback, timestamp)

    def log_user_features(self) -> None:
        features = self.feature_data.get_user_features(
            self.timezone, datetime.utcnow(), self.is_in
        )
        self.log("comfort features", comfort_features=features)

    def log_multi_user(self, predictions: List[float]) -> None:
        average_prediction = self.multi_user_predictions_average(predictions)
        prediction_for_user_ids = "; ".join(
            [
                f"{nearby_user_feedback.get('user_id', 'NO FEEDBACK')}, prediction "
                f"{user_prediction}, discomfort {user_prediction - average_prediction}"
                for nearby_user_feedback, user_prediction in zip(
                    self.nearby_users_feedbacks, predictions
                )
            ]
        )
        self.log(f"multi_user_comfort {prediction_for_user_ids}")

    def _log(self, predictions, adjusted_predictions):
        self.log_user_features()
        self.log_multi_user(adjusted_predictions)
        self.log(
            "comfort adjustment",
            prediction=self.multi_user_predictions_average(predictions),
            adjusted_prediction=self.multi_user_predictions_average(
                adjusted_predictions
            ),
        )

    def _log_climate(
        self, climate_predictions, comfort_predictions, adjusted_predictions
    ):
        self.log(
            "comfort adjustments",
            climate_prediction=climate_predictions,
            prediction=self._average_over_nearby_users(comfort_predictions),
            adjusted_prediction=self._average_over_nearby_users(adjusted_predictions),
        )


def _get_weights(nearby_users_feedbacks: List[Feedback]) -> List[float]:
    if nearby_users_feedbacks == [{}]:
        return [1]
    times = [
        _seconds_since_last_feedback(feedback) for feedback in nearby_users_feedbacks
    ]
    return [
        1 / (_map_intensity_interval(INTENSITY) + time ** _map_focus_interval(FOCUS))
        for time in times
    ]


def _map_focus_interval(parameter: float) -> float:
    return parameter / (1 - parameter)


def _map_intensity_interval(parameter: float) -> float:
    return (1 - parameter) / parameter


def _seconds_since_last_feedback(feedback: Feedback) -> float:
    # avoid division by zero or indeterminate form by adding a small non-zero value
    return (datetime.utcnow() - feedback["created_on"]).total_seconds() + np.finfo(
        float
    ).eps


def fitted_timeseries(
    features: UserFeatures, prediction: ClimatePrediction
) -> List[List[float]]:
    # exponential decay more accurately models the sensor data but simply fitting
    # a first-degree polynomial was found to give comparable results in experiment
    return [
        fit_timeseries(features[p][-1], prediction[QUANTITY_MAP[q]])
        for p, q in (
            ("previous_temperatures", "temperature"),
            ("previous_humidities", "humidity"),
        )
    ]


def fit_timeseries(last: float, pred: float) -> List[float]:
    m, b = np.polyfit(
        [-selection.STATIC_INTERPOLATION / COMFORT_TIMESERIES_FREQ, 0], [last, pred], 1
    )
    return [
        m * x + b
        for x in range(-COMFORT_TIMESERIES_INTERVAL // COMFORT_TIMESERIES_FREQ, 1)
    ]
