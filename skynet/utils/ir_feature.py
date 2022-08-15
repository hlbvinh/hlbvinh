from typing import List, Optional

from .enums import Power
from .types import ApplianceState, DeploymentSettings, IRFeature

APPLIANCE_PROPERTIES = [
    "power",
    "mode",
    "temperature",
    "fan",
    "louver",
    "swing",
    "ventilation",
]
AMBI_CONTROL_BUTTONS = ["power", "mode", "temp_up", "temp_down", "fan", "ventilation"]

FAN_SETTING_ORDER = [
    "low",
    "med-low",
    "med",
    "medium",
    "med-high",
    "mid",
    "auto",
    "auto-auto",
    "quiet",
    "soft",
    "night",
    "natural",
    "oscillate",
    "high",
    "very-high",
]
DEFAULT_IRFEATURE = {
    "cool": {
        "fan": {"ftype": "select_option", "value": ["auto"]},
        "temperature": {"ftype": "select_option", "value": ["25"]},
    }
}


class IRFeatureError(ValueError):
    pass


class NoButtonError(IRFeatureError):
    pass


def button_press(old_state, new_state):
    """Determine most likely button press from old_state to new_state.

    Parameters
    ----------
    old_state: dict
        old AC state

    new_state: dict
        new AC state

    Returns
    -------
    str
        button press
    """
    if new_state == old_state:
        raise NoButtonError("no button press for identical states")

    na = "na"
    new_temp = str(new_state.get("temperature", 0))
    old_temp = str(old_state.get("temperature", 0))
    new_pow = new_state.get("power", na)
    old_pow = old_state.get("power", na)

    if new_pow != old_pow:
        button = "power"
    elif new_state.get("mode", na) != old_state.get("mode", na):
        button = "mode"
    elif new_state.get("fan", na) != old_state.get("fan", na):
        button = "fan"
    elif new_temp < old_temp:
        button = "temp_down"
    elif new_temp > old_temp:
        button = "temp_up"
    elif new_state.get("swing", na) != old_state.get("swing", na):
        button = "swing"
    elif new_state.get("louver", na) != old_state.get("louver", na):
        button = "louver"
    elif new_state.get("ventilation", na) != old_state.get("ventilation", na):
        button = "ventilation"
    else:
        raise NoButtonError(
            "no button press from " "{} to {} ".format(old_state, new_state)
        )

    return button


def best_possible_deployment(
    current_state: ApplianceState,
    last_on_state: ApplianceState,
    settings,
    ir_feature: IRFeature,
    device_id: str,
) -> DeploymentSettings:
    """Polish deployment setting before deploying to the device.

    Deployment setting is determined based on
    1. "setting" - which is determined by custom logic / ML prediction,
        not necessarily possible to deploy
    2. current and last_on state - to provide reasonable setting based on history

    Args:
        current_state:
        last_on_state:
        settings:
        ir_feature:
        device_id:

    Returns:
        Possible DeploymentSettings

    """
    signal = dict()
    signal["device_id"] = device_id
    mode = settings.get("mode", "cool")
    power = settings.get("power", Power.OFF)

    if mode == "heat":
        temperature = settings.get("temperature", 21)
    else:
        temperature = settings.get("temperature", 24)

    if mode not in ir_feature:
        mode = "cool"

    feat = ir_feature[mode]

    signal["temperature"] = property_value(
        "temperature", current_state, settings, feat, default=temperature
    )
    signal["mode"] = mode
    signal["power"] = power

    # Get fan from last on state if it's not in settings (for dry mode)
    # In other words, preserve if not explicitly chosen.
    signal["fan"] = property_value("fan", last_on_state, settings, feat, default="auto")

    # get louver and swing from last_on_state to preserve it
    signal["louver"] = property_value("louver", last_on_state, last_on_state, feat)
    signal["swing"] = property_value("swing", last_on_state, last_on_state, feat)
    # HACK we used to double check whether ventilation was supported in the ir
    # features, it seems that during the first 30 min of the setup we don't
    # always get the right ir profile from the event. It's better to simply let
    # ventilation pass and check on BE whether or not ventilation setting is
    # actually supported.
    signal["ventilation"] = settings.get("ventilation", None) or current_state.get(
        "ventilation", None
    )

    signal["button"] = button_press(current_state, signal)

    return DeploymentSettings(**signal)


def property_value(name, current_state, settings, feature, default=None):
    """Get the value of a AC state property."""
    value = None
    if name in feature:
        vals = feature[name]
        if vals.get("ftype") in ["select_option", "radio"]:
            for state in [settings, current_state]:
                val = state.get(name)

                # IR Feature uses strings for temperatures
                if str(val) in vals["value"]:
                    value = val
                    break

            # really wanted to use that for ... else once in my life
            else:
                value = default if default in vals["value"] else vals["value"][-1]
    else:
        value = None

    # HACK
    # TODO: temperature is no longer required, don't send one if not necessary
    if name == "temperature" and value is None:
        value = 24

    return value


def get_lowest_fan_settings(fan_settings: List[str]) -> Optional[str]:
    available_settings = [s for s in FAN_SETTING_ORDER if s in fan_settings]
    if available_settings:
        return available_settings[0]

    return None


def is_default_irfeature(ir_feature):
    return ir_feature == DEFAULT_IRFEATURE
