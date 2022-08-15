from datetime import datetime
from typing import List, Optional

import numpy as np

from ..utils import cache_util
from ..utils.log_util import get_logger
from ..utils.types import ApplianceState, BasicDeployment, Connections, ControlTarget
from . import adjust, penalty, setting_selection, target, util
from .feedback_adjustment import FeedbackAdjustment
from .maintain import RedisMaintain
from .prediction import Prediction
from .target import Target

log = get_logger(__name__)


class Tracking(util.LogMixin):
    def __init__(
        self,
        state: ApplianceState,
        device_id: str,
        connections: Connections,
        prediction: Prediction,
        target: Target,
        feedback_adjustment: FeedbackAdjustment,
        control_target: ControlTarget,
        experiments=[],
    ):
        self.state = state
        self.device_id = device_id
        self.connections = connections
        self.prediction = prediction
        self.penalty_factor = 1.0
        self.target = target
        self.feedback_adjustment = feedback_adjustment
        self.control_target = control_target
        self._deviation_trackers = adjust.create_redis_trackers(
            connections.redis, device_id, log_fun=self.log
        )
        self.maintain = RedisMaintain(
            connections.redis,
            self.device_id,
            self.state["temperature"],
            log_fun=self.log,
            experiments=experiments,
        )

    async def get_best_setting(self) -> Optional[BasicDeployment]:
        """Gets best deployment setting based on predicted states (and its comfort scores).

        Returns:
            optional deployment settings - None means keeping the current setting
        """

        states, predictions = await self.prediction.get_predictions()

        best_setting = self.select_best_setting(predictions, states)
        await self.log_target_error()

        if not util.state_update_required(
            self.state, best_setting, self.target.scaled_target_delta
        ):
            self.log(
                f"adjustment from {self.state['temperature']} to {best_setting['temperature']}"
                f" looks wrong for target delta {self.target.target_delta}"
            )
            return None

        return BasicDeployment(**best_setting)

    async def log_target_error(self) -> None:
        control_state = self.target.control_state
        control_state.update(
            {
                "minute_ac_on": await self.minute_ac_on(),
                "minutes_since_last_state": (
                    datetime.utcnow() - self.state["created_on"]
                ).total_seconds()
                // 60,
            }
        )
        self.log("control_state", control_state=control_state)

    def select_best_setting(
        self, predictions: List[float], states: List[ApplianceState]
    ) -> ApplianceState:

        preds = np.array(predictions)
        deviations = abs(preds - self.target.target_value)

        if setting_selection.previous_mode_with_inactive_temperatures(
            self.current_mode, states[0]["mode"]
        ):
            penalized_deviations = deviations
        else:
            penalized_deviations = self.penalise_deviation_from_current_temperature(
                preds, deviations, states
            )

        climate_predictions = {
            s["temperature"]: {"prediction": pred, "delta": d, "penalized": pen}
            for s, pred, d, pen in zip(
                states, preds, preds - self.target.target_value, penalized_deviations
            )
        }
        self.log(
            "penalty",
            climate_predictions=climate_predictions,
            quantity=self.target.quantity,
        )

        best_setting = states[np.argmin(penalized_deviations)].copy()

        best_setting[
            "temperature"
        ] = self.deviation_tracker.get_set_temperature_with_offset(
            self.target.scaled_target_delta,
            best_setting["temperature"],
            best_setting["mode"],
            [s["temperature"] for s in states],
        )

        best_setting["temperature"] = self.maintain.maintain_temperature(
            self.target,
            self.current_mode,
            best_setting["mode"],
            best_setting["temperature"],
            [s["temperature"] for s in states],
        )
        best_setting["temperature"] = self.feedback_adjustment.override_temperature(
            best_setting["temperature"], [s["temperature"] for s in states]
        )
        return best_setting

    def penalise_deviation_from_current_temperature(
        self,
        predictions: np.ndarray,
        deviations: np.ndarray,
        states: List[ApplianceState],
    ) -> np.ndarray:
        assert self.target.quantity

        error_factor = penalty.error_factor(abs(self.target.scaled_target_delta))
        predicted_error_factor = penalty.error_factor(
            penalty.current_state_deviation(deviations, states, self.state),
            target.TARGET_DELTA_SCALING_FACTOR[self.target.quantity],
        )
        time_factor = penalty.time_factor(datetime.utcnow(), self.last_deployment)
        factors = (
            self.penalty_factor * error_factor * predicted_error_factor * time_factor
        )

        deviation_penalties = factors * penalty.penalize_deviations(
            predictions, states, self.current_set_temp(states)
        )
        penalized_deviations = deviations + deviation_penalties

        climate_predictions = {
            s["temperature"]: {
                "prediction": pred,
                "prediction_delta": d,
                "penalty": pen,
                "penalized": dev,
            }
            for s, pred, d, pen, dev in zip(
                states,
                predictions,
                abs(predictions - self.target.target_value),
                deviation_penalties,
                penalized_deviations,
            )
        }
        current_set_temp = self.current_set_temp(states)
        pred_set_temp = states[np.argmin(penalized_deviations)]["temperature"]
        if adjust.is_numeric([current_set_temp, pred_set_temp]):
            current_set_temp = float(current_set_temp)  # type: ignore
            pred_set_temp = float(pred_set_temp)
        self.log(
            "penalty factors",
            error_factor=error_factor,
            predicted_error_factor=predicted_error_factor,
            time_factor=time_factor,
            factors=factors,
            current_set_temperature=current_set_temp,
            predicted_set_temperature=pred_set_temp,
            quantity=self.target.quantity,
            climate_predictions=climate_predictions,
        )
        return penalized_deviations

    async def minute_ac_on(self) -> int:
        last_off_state_timestamp = await cache_util.get_last_off_state_timestamp(
            redis=self.connections.redis, key_arg=self.device_id
        )
        if last_off_state_timestamp:
            return (datetime.utcnow() - last_off_state_timestamp).total_seconds() // 60
        return 0

    def current_set_temp(self, states):
        return self.deviation_tracker.remove_offset(
            self.state["temperature"], [s["temperature"] for s in states]
        )

    @property
    def current_mode(self) -> str:
        return self.state["mode"]

    @property
    def last_deployment(self) -> datetime:
        return self.state["created_on"]

    @property
    def deviation_tracker(self):
        return self._deviation_trackers[self.target.quantity]
