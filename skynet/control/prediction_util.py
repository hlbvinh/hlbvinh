from typing import List, Optional

from ..utils import data, thermo
from ..utils.enums import Power
from ..utils.types import ApplianceState, IRFeature, StateTemperature

INACTIVE_TEMP_OFFSET = 2.0
MIN_TEMP = 14
MAX_TEMP = 34


def get_states_to_predict(
    mode: str,
    ir_feature: IRFeature,
    current_temperature: float,
    current_tempset: StateTemperature,
) -> List[ApplianceState]:
    default_state = get_default_state(
        mode, ir_feature, current_temperature, current_tempset
    )
    if default_state:
        return [default_state]
    return [
        {"power": Power.ON, "mode": mode, "temperature": temperature}
        for temperature in ir_feature[mode]["temperature"]["value"]
        if not is_string_temperature(temperature)
    ]


def get_default_state(
    mode: str,
    ir_feature: IRFeature,
    current_temperature: float,
    current_tempset: StateTemperature,
) -> Optional[ApplianceState]:
    """Gets possible states for predicting future state.

    Args:
        mode:
        ir_feature:
        current_temperature:
        current_tempset:

    Returns:

    """

    if has_no_set_temperature(mode, ir_feature):
        return {"mode": mode, "power": Power.ON, "temperature": None}

    temperatures = ir_feature[mode]["temperature"]["value"]

    if has_only_string_temperature(temperatures):
        return {
            "mode": mode,
            "power": Power.ON,
            "temperature": get_default_string_values(temperatures),
        }
    if mode == "fan" and has_room_temperature(temperatures):
        new_tempset = tempset_to_prevent_fan_mode_switching_ac_off(
            current_temperature, current_tempset, temperatures
        )
        return {"mode": mode, "power": Power.ON, "temperature": new_tempset}
    return None


def has_no_set_temperature(mode: str, ir_feature: IRFeature):
    try:
        values = ir_feature[mode]["temperature"]["value"]
        assert values
    except (TypeError, AssertionError, KeyError):
        return True
    else:
        return False


def has_only_string_temperature(temperatures: List[str]) -> bool:
    return all([is_string_temperature(temperature) for temperature in temperatures])


def get_default_string_values(temperatures: List[str]):
    for s in sorted(temperatures):
        if s.lower() in ["auto", "blank"]:
            return s

    med_string = [s for s in temperatures if "d" in s]
    if med_string:
        return sorted(med_string)[0]
    return temperatures[len(temperatures) // 2]


def has_room_temperature(temperatures: List[str]) -> bool:
    return all([is_room_temperature(temperature) for temperature in temperatures])


def tempset_to_prevent_fan_mode_switching_ac_off(
    current_temperature: float,
    current_tempset: StateTemperature,
    temperatures: List[str],
) -> str:
    """some acs (hitachi circulation mode) would stop the fan if the set point in
    fan mode is higher than the current temperature, we prevent that by trying to
    deploy a set point lower than the current temperature while minimising the
    number of deployments"""
    new_tempset = find_nearest(current_temperature - INACTIVE_TEMP_OFFSET, temperatures)
    if is_room_temperature(current_tempset) and str(current_tempset) in temperatures:
        if float(current_tempset) < float(new_tempset):  # type: ignore
            return str(current_tempset)
    return new_tempset


def find_nearest(temperature: float, temperatures: List[str]) -> str:
    return min(temperatures, key=lambda t: abs(float(t) - temperature))


def is_string_temperature(temperature: StateTemperature) -> bool:
    return not data.is_int(temperature) and not data.is_float(temperature)


def is_room_temperature(temperature: StateTemperature) -> bool:
    try:
        temp = float(temperature)  # type: ignore
    except (ValueError, TypeError):
        return False
    else:
        if temp > thermo.MAX_CELSIUS:
            temp = thermo.celsius_from_fahrenheit(temp)
        return MIN_TEMP <= temp < MAX_TEMP
