from typing import Callable, List

from ..utils.types import IRFeature, ModeSelection
from .mode_model_util import COOL_MODES, HEAT_MODES

# analysed the distribution of temperature for manual users using heat and
# cool modes; the quantiles give us some upper and lower bounds on when users
# would not use heat or cool.
MAX_TEMPERATURES_FOR_HEATING_PER_QUANTILE = {
    "0.999": {"temperature": 28.48, "temperature_out": 27.40},
    "0.995": {"temperature": 27.04, "temperature_out": 22.68},
    "0.99": {"temperature": 26.40, "temperature_out": 20.87},
}
MIN_TEMPERATURES_FOR_COOLING_PER_QUANTILE = {
    "0.999": {"temperature": 14.84, "temperature_out": -0.81},
    "0.995": {"temperature": 17.18, "temperature_out": 5.55},
    "0.99": {"temperature": 17.94, "temperature_out": 8.66},
}
HEAT_COOL_TRANSITION_QUANTILE = "0.99"
MAX_TEMPERATURES_FOR_HEATING = MAX_TEMPERATURES_FOR_HEATING_PER_QUANTILE[
    HEAT_COOL_TRANSITION_QUANTILE
]
MIN_TEMPERATURES_FOR_COOLING = MIN_TEMPERATURES_FOR_COOLING_PER_QUANTILE[
    HEAT_COOL_TRANSITION_QUANTILE
]

TOO_COLD_FOR_DRY_MODE = 1.5
TOO_EARLY_TO_SWITCH_TO_DRY = 0.0

PREVENT_FUNCTIONS: List[Callable] = []


def filter_mode_selection(
    mode_selection: ModeSelection,
    temperature: float,
    temperature_out: float,
    current_mode: str,
    ir_feature: IRFeature,
    scaled_target_delta: float,
) -> ModeSelection:
    kwargs = {
        "temperature": temperature,
        "temperature_out": temperature_out,
        "current_mode": current_mode,
        "ir_feature": ir_feature,
        "scaled_target_delta": scaled_target_delta,
    }
    for filtering in PREVENT_FUNCTIONS:
        mode_selection = filtering(mode_selection, **kwargs)

    return mode_selection


def mode_selection_filtering(fun):
    PREVENT_FUNCTIONS.append(fun)
    return fun


@mode_selection_filtering
def prevent_using_auto_mode(mode_selection: ModeSelection, **_kwargs) -> ModeSelection:
    mode_selection_without_auto = [m for m in mode_selection if m != "auto"]
    if mode_selection_without_auto:
        return mode_selection_without_auto
    return mode_selection


@mode_selection_filtering
def prevent_bad_heat_cool_selection(
    mode_selection: ModeSelection,
    *,
    temperature: float,
    temperature_out: float,
    **_kwargs
) -> ModeSelection:
    temperatures = {"temperature": temperature, "temperature_out": temperature_out}
    heating = [m for m in mode_selection if m in HEAT_MODES]
    cooling = [m for m in mode_selection if m in COOL_MODES]

    if (
        any(temperatures[q] < MIN_TEMPERATURES_FOR_COOLING[q] for q in temperatures)
        and heating
    ):
        return heating

    if (
        any(temperatures[q] > MAX_TEMPERATURES_FOR_HEATING[q] for q in temperatures)
        and cooling
    ):
        return cooling
    return mode_selection


@mode_selection_filtering
def prevent_dry_mode_from_cooling_too_much(
    mode_selection: ModeSelection,
    *,
    current_mode: str,
    ir_feature: IRFeature,
    scaled_target_delta: float,
    **_kwargs
) -> ModeSelection:
    currently_using_dry_mode = current_mode == "dry"
    is_too_cold_to_keep_using_dry_mode = (
        currently_using_dry_mode and scaled_target_delta >= TOO_COLD_FOR_DRY_MODE
    )
    is_too_early_to_try_dry_mode = (
        not currently_using_dry_mode
        and scaled_target_delta >= TOO_EARLY_TO_SWITCH_TO_DRY
    )
    does_dry_mode_have_a_single_setting = dry_mode_has_a_single_setting(ir_feature)
    not_only_dry_mode_selected = mode_selection != ["dry"]

    if (
        is_too_early_to_try_dry_mode
        or is_too_cold_to_keep_using_dry_mode
        or does_dry_mode_have_a_single_setting
    ) and not_only_dry_mode_selected:
        mode_selection_without_dry = [m for m in mode_selection if m != "dry"]
        return mode_selection_without_dry

    return mode_selection


@mode_selection_filtering
def prevent_fan_mode_when_too_far_from_target(
    mode_selection: ModeSelection, *, scaled_target_delta: float, **_kwargs
) -> ModeSelection:
    # Apply logic such that we only deal with fan mode
    # when target values is close to 0.
    # also when target value becomes too hot, too cold, this logic will force
    # a mode change to other mode
    away_from_target = abs(scaled_target_delta) > 0.5
    without_fan = [mode for mode in mode_selection if mode != "fan"]
    if away_from_target and len(without_fan) == 1:
        return without_fan
    return mode_selection


def dry_mode_has_a_single_setting(ir_feature: IRFeature) -> bool:
    if "dry" not in ir_feature:
        return True
    if "temperature" not in ir_feature["dry"]:
        return True
    if len(ir_feature["dry"]["temperature"]["value"]) <= 1:
        return True
    return False
