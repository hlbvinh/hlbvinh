import functools
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from ..utils import thermo
from ..utils.types import ApplianceState, StateTemperature

DEFAULT_PENALTY = 0.5
L1_FACTOR = 0.45
L2_FACTOR = 0.18
TIME_TAU = timedelta(minutes=60)
DEFAULT_PREDICTED_DEVIATION = 0.0

STATE_UPDATE_FACTOR = 0.2
FEEDBACK_UPDATE_FACTOR = 0.1
ADR_UPDATE_FACTOR = 0.1


def control_target_change_penalty_factor(
    old_mode: Optional[str], new_mode: str
) -> float:
    tracking_modes = ["comfort", "temperature"]
    if (old_mode is None) or (old_mode in tracking_modes) != (
        new_mode in tracking_modes
    ):
        return 0
    return STATE_UPDATE_FACTOR


def time_factor(t, t_0, tau=TIME_TAU):
    if not isinstance(t, datetime) or not isinstance(t_0, datetime):
        return 1.0
    return np.exp(-(t - t_0).total_seconds() / tau.total_seconds())


def error_factor(deviation, beta=1):
    return np.exp(-deviation / beta)


def penalty(temperature_set: float, current_temperature_set: float, factor: float, fun):
    if temperature_set == current_temperature_set:
        return 0.0
    if any(np.isnan([temperature_set, current_temperature_set])):
        return DEFAULT_PENALTY
    return factor * fun(temperature_set, current_temperature_set)


l1 = functools.partial(penalty, fun=lambda x, y: abs(x - y))
l2 = functools.partial(penalty, fun=lambda x, y: (x - y) ** 2.0)


def penalize_deviations(
    predictions: np.ndarray,
    states: List[ApplianceState],
    current_temperature_set: StateTemperature,
) -> np.ndarray:

    current_tempset = thermo.fix_temperature(current_temperature_set)
    tempsets = [thermo.fix_temperature(s["temperature"]) for s in states]

    if np.isnan(current_tempset):

        mean_state_temp = np.mean(
            [tempset for tempset in tempsets if not np.isnan(tempset)]
        )
        current_tempset = thermo.fix_temperature(mean_state_temp)

    penalties = [
        l1(tempset, current_tempset, L1_FACTOR * np.std(predictions))
        + l2(tempset, current_tempset, L2_FACTOR * np.std(predictions))
        for tempset in tempsets
    ]
    return np.array(penalties)


def current_state_deviation(
    deviations: np.ndarray, states: List[ApplianceState], current_state: ApplianceState
) -> float:

    for state, deviation in zip(states, deviations):
        if state["mode"] == current_state["mode"] and state["temperature"] == str(
            current_state["temperature"]
        ):
            return deviation
    return DEFAULT_PREDICTED_DEVIATION
