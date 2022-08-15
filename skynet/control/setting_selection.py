from typing import List

import numpy as np

from ..utils.types import ApplianceState

MODES_WITH_INACTIVE_TEMPSET = ["fan"]


def select_away_mode_setting(
    predictions: List[float],
    states: List[ApplianceState],
    threshold_type: str,
    quantity,
) -> ApplianceState:

    # we used to pick the set point solely based on the climate model
    # predictions and the threshold. it's safe to assume that if the quantity
    # is temperature we can try to hardcode picking the lowest set point in
    # away_upper and highest set point in away_lower.

    # HACK we assume that the climate model gives us states all with the same
    # mode picked by the mode model
    mode = states[0]["mode"]
    if quantity == "temperature":
        try:
            if threshold_type == "upper" and mode in ("cool", "dry"):
                return min(states, key=lambda s: float(s["temperature"]))
            if threshold_type == "lower" and mode == "heat":
                return max(states, key=lambda s: float(s["temperature"]))
        except (TypeError, ValueError):
            pass

    # if we're not able to use the hardcoded results, the trust the AI to make
    # an educated guess.
    if threshold_type == "upper":
        return states[np.argmin(predictions)]
    if threshold_type == "lower":
        return states[np.argmax(predictions)]
    raise ValueError("don't understand threshold type {threshold_type}")


def previous_mode_with_inactive_temperatures(current_mode, best_mode):
    return (
        current_mode in MODES_WITH_INACTIVE_TEMPSET
        and best_mode not in MODES_WITH_INACTIVE_TEMPSET
    )
