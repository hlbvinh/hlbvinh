from datetime import datetime
from typing import List, Optional

import numpy as np

from skynet.utils.cache_util import _key

from ..prediction.mode_model_util import COOL_MODES, HEAT_MODES
from ..utils import event_parsing
from ..utils.progressive_rollout import Experiment
from ..utils.types import AutomatedDemandResponse, ControlTarget, ModePrefKey, Sensors
from .comfort import Comfort
from .managed_manual import ManagedManual
from .util import is_active

TARGET_DELTA_SCALING_FACTOR = {
    "humidex": 2.5,
    "comfort": 0.9,
    "temperature": 1.2,
    "humidity": 5.0,
    "set_temperature": 1.2,
}
SIGNAL_LEVEL_PENALTY = {0: 0.0, 2: 2.0, 3: 1.0, 4: 3.0}
ADR_OFFSET = {
    "comfort": {0: 0.0, 2: 0.7, 3: 0.4, 4: 1.0},
    "set_temperature": {0: 0.0, 2: 2.0, 3: 1.0, 4: 3.0},
}
for i in np.arange(0.1, 1.5, 0.1):
    ADR_OFFSET["comfort"][str(i)] = i  # type: ignore
OFF_SIGNAL_LEVEL = 1
AI_OPTIMISATION_SIGNAL_LEVEL = 5


class Target:
    def __init__(
        self,
        device_id: str,
        sensors: Sensors,
        comfort: Comfort,
        managed_manual: ManagedManual,
        control_target: ControlTarget,
        automated_demand_response: Optional[AutomatedDemandResponse],
        experiments: List[Experiment] = [],
        redis=None,
    ):
        (
            self.control_mode,
            self.quantity,
            self.threshold_type,
            self._threshold,
            self._target_value,
        ) = event_parsing.extract_control_target(control_target)
        self.control_target = control_target
        self.sensors = sensors
        self.comfort = comfort
        self.device_id = device_id
        self.experiments = experiments
        self.redis = redis
        self.adr = ADRPenalty(
            automated_demand_response,
            self.control_mode,
            self.quantity,
            self.threshold_type,
            self.device_id,
            self.redis,
        )
        self.managed_manual = managed_manual
        self.mode_pref_key = ModePrefKey(
            self.control_mode, self.quantity, self.threshold_type
        )
        self._current_value: float

    @property
    def target_value(self) -> Optional[float]:
        if self.quantity == "comfort":
            return 0 + self.adr_penalty
        if self._target_value is None:
            return None
        return self._target_value + self.adr_penalty

    @property
    def threshold(self) -> Optional[float]:
        if self.control_mode == "away":
            assert self._threshold is not None
            return self._threshold + self.adr_penalty
        return self._threshold

    @property
    def current_value(self) -> float:
        assert self.quantity
        if self.quantity == "comfort":
            assert self._current_value
            return self._current_value
        if self.quantity == "set_temperature":
            return self.sensors["temperature"]
        return self.sensors[self.quantity]

    @property
    def target_delta(self) -> float:
        assert self.target_value is not None
        return self.target_value - self.current_value

    @property
    def scaled_target_delta(self):
        return self.target_delta / TARGET_DELTA_SCALING_FACTOR[self.quantity]

    @property
    def scaled_mode_model_target_delta(self) -> float:
        if self.quantity == "comfort":
            return self.mode_model_target_delta / TARGET_DELTA_SCALING_FACTOR["humidex"]
        return (
            self.mode_model_target_delta
            / TARGET_DELTA_SCALING_FACTOR[self.quantity]  # type: ignore
        )

    @property
    def mode_model_target_delta(self) -> float:
        if self.quantity == "comfort":
            return (
                -self.current_value
                * TARGET_DELTA_SCALING_FACTOR["humidex"]
                / TARGET_DELTA_SCALING_FACTOR["comfort"]
            )
        # we want to prevent the adr penalty to affect the mode selection
        assert self._target_value is not None
        return self._target_value - self.current_value

    @property
    def mode_model_quantity(self) -> str:
        assert self.quantity is not None
        if self.quantity == "comfort":
            return "humidex"
        return self.quantity

    @property
    def climate_model_quantity(self) -> Optional[str]:
        if self.quantity == "comfort":
            return None
        return self.quantity

    async def from_climate_predictions(self, predictions) -> List[float]:
        """Predicts comfort scores based on predicted states.

        Args:
            predictions:

        Returns:

        """
        if self.quantity == "comfort":
            return await self.comfort.from_climate_predictions(predictions)
        return predictions

    async def update_value(self) -> None:
        # HACK we need to use await for the new adr offset, we put the query
        # here so that we don't have to modify the signature of the @property
        # in the ADRPenalty class
        await self.adr.ai_optimisation()

        if self.quantity == "comfort":
            self._current_value = await self.comfort.predict_adjusted()
        elif self.control_mode == "managed_manual":
            self._target_value = await self.managed_manual.get_target_value()

    @property
    def turn_off_appliance(self) -> bool:
        return self.adr.turn_off_appliance

    @property
    def adr_penalty(self) -> float:
        return self.adr.penalty

    def set_predicted_mode(self, mode: str) -> None:
        self.adr.set_predicted_mode(mode)

    @property
    def control_state(self):
        return {
            "control_target": self.control_target["quantity"],
            "quantity": self.quantity,
            "current": self.current_value,
            "target": self.target_value,
            "adr_penalty": self.adr_penalty,
            "delta": -self.target_delta,
            "error": abs(self.target_delta),
            "minute_since_last_control_target": self.minute_since_last_control_target,
        }

    @property
    def minute_since_last_control_target(self) -> int:
        return (
            datetime.utcnow() - self.control_target["created_on"]
        ).total_seconds() // 60


class ADRPenalty:
    def __init__(
        self,
        automated_demand_response: Optional[AutomatedDemandResponse],
        control_mode: str,
        quantity: Optional[str],
        threshold_type: Optional[str] = None,
        device_id=None,
        redis=None,
    ):
        self.automated_demand_response = automated_demand_response
        self.control_mode = control_mode
        self.quantity = quantity
        self.threshold_type = threshold_type
        self.predicted_mode: str
        self.device_id = device_id
        self.redis = redis
        self.ai_optimisation_level: Optional[float] = None

    @property
    def penalty(self) -> float:
        if self.is_active and self.is_mode_supported:
            if self.control_mode == "away":
                if self.threshold_type == "upper":
                    return self.offset
                if self.threshold_type == "lower":
                    return -self.offset
            elif self.control_mode in ["comfort", "temperature", "managed_manual"]:
                if self.predicted_mode in COOL_MODES:
                    return self.offset
                if self.predicted_mode in HEAT_MODES:
                    return -self.offset
        return 0

    @property
    def turn_off_appliance(self) -> bool:
        if self.is_active:
            assert self.automated_demand_response
            return self.automated_demand_response.signal_level == OFF_SIGNAL_LEVEL
        return False

    @property
    def is_active(self) -> bool:
        return is_active(self.automated_demand_response)

    @property
    def is_mode_supported(self) -> bool:
        return self.control_mode in ["comfort", "temperature", "away", "managed_manual"]

    async def ai_optimisation(self):
        if (
            self.automated_demand_response
            and self.automated_demand_response.signal_level
            == AI_OPTIMISATION_SIGNAL_LEVEL
        ):
            self.ai_optimisation_level = await get_adr_offset(
                self.redis, self.device_id
            )

    @property
    def offset(self) -> float:
        assert self.automated_demand_response
        assert self.quantity

        if self.ai_optimisation_level is not None:
            if self.quantity == "comfort":
                # following last year simulation of comfort/set_temperature adr
                # offset mapping we had:
                # | comfort-offset | setpoint-offset |
                # |            1.4 |         4.45469 |
                # |            1.3 |         4.11668 |
                # |            1.2 |         3.76107 |
                # |            1.1 |         3.40282 |
                # |            1.0 |         3.03273 |
                # |            0.9 |         2.67673 |
                # |            0.8 |         2.32594 |
                # |            0.7 |         1.97629 |
                # |            0.6 |         1.64608 |
                # |            0.5 |         1.31436 |
                # |            0.4 |        0.995312 |
                # |            0.3 |        0.698743 |
                # |            0.2 |        0.423719 |
                # |            0.1 |        0.186985 |
                # applying linear regression gives the following mapping
                return 0.29762258 * self.ai_optimisation_level + 0.09072029
            if self.quantity == "set_temperature":
                return self.ai_optimisation_level
            return self.ai_optimisation_level * self.scaling_factor

        try:
            return ADR_OFFSET[self.quantity][
                self.automated_demand_response.signal_level
            ]
        except KeyError:
            return self.level * self.scaling_factor

    @property
    def level(self) -> float:
        assert self.automated_demand_response
        return SIGNAL_LEVEL_PENALTY[self.automated_demand_response.signal_level]

    @property
    def scaling_factor(self) -> float:
        assert self.quantity
        return TARGET_DELTA_SCALING_FACTOR[self.quantity]

    def set_predicted_mode(self, mode: str) -> None:
        self.predicted_mode = mode


async def get_adr_offset(redis, device_id):
    # control_worker is already locking per device_id when processing an event,
    # so we don't need to worry to much about race conditions
    offset = await redis.hget(_key("adr:offset"), device_id)
    if offset is None:
        offset = await redis.lpop(_key("adr:sobol"))
        await redis.hmset(_key("adr:offset"), {device_id: offset})
    return float(offset)
