import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import dateutil
import voluptuous
from voluptuous import REMOVE_EXTRA, All, Coerce, In, Invalid, Range, Required, Schema

from . import parse
from .compensation import COMPENSATE_COLUMN, compensate_sensors
from .enums import NearbyUser
from .log_util import get_logger
from .types import (
    AutomatedDemandResponse,
    ControlTarget,
    ModePref,
    ModePrefKey,
    NearbyUserAction,
)
from ..prediction.mode_config import MODES
from ..prediction.mode_model_util import MULTIMODES
from ..utils.config import MAX_COMFORT, MIN_COMFORT

log = get_logger(__name__)

COMFORT_MODES = set(["climate", "comfort"])

# this default value is based on the average humidity for all users in the last
# two weeks, see https://app.scalyr.com/y/VqWzT4iNPy5
AVERAGE_HUMIDITY = 61.9322


def parse_host_port(address: str) -> Tuple[str, str]:
    host, port = address.split("//")[1].split(":")
    return (host, port)


def to_naive_datetime(datetime_or_str):
    if isinstance(datetime_or_str, str):
        datetime_or_str = dateutil.parser.parse(datetime_or_str)
    return datetime_or_str.replace(tzinfo=None)


CONTROL_TARGET_SCHEMA = Schema(
    {
        Required("quantity"): str,
        Required("value"): voluptuous.Any(None, Coerce(float)),
        Required("origin"): str,
        Required("created_on"): datetime,
    },
    extra=REMOVE_EXTRA,
)


def parse_control_target(msg):
    parsed = parse.lower_dict(msg)
    return CONTROL_TARGET_SCHEMA(parsed)


SENSORS_COMMON = {
    voluptuous.Optional("LU"): {Required("infrared_spectrum"): Coerce(float)},
    voluptuous.Optional("CO"): float,
    Required("created_on"): datetime,
}

SENSORS_LEGACY_SCHEMA = Schema(
    {Required("TP"): Coerce(float), Required("HM"): Coerce(float), **SENSORS_COMMON},
    extra=REMOVE_EXTRA,
)

SENSORS_REFINED_SCHEMA = Schema(
    {
        Required("TP_refined"): Coerce(float),
        Required("HM_refined"): Coerce(float),
        **SENSORS_COMMON,
    },
    extra=REMOVE_EXTRA,
)


def _parse_sensors_common(msg):
    sensors = {"created_on": msg["created_on"]}
    if "CO" in msg:
        sensors.update({"co2": msg["CO"]})
    if "LU" in msg:
        sensors.update({"luminosity": msg["LU"]["infrared_spectrum"]})
    return sensors


def _parse_sensors_refined(msg):
    return {
        "temperature": msg["TP_refined"],
        "humidity": msg["HM_refined"],
        COMPENSATE_COLUMN: True,
    }


def _parse_sensors_legacy(msg):
    return {"temperature": msg["TP"], "humidity": msg["HM"], COMPENSATE_COLUMN: False}


def _parse_sensors(msg):

    try:
        msg = SENSORS_REFINED_SCHEMA(msg)
        parse_fun = _parse_sensors_refined
    except Invalid:
        msg = SENSORS_LEGACY_SCHEMA(msg)
        parse_fun = _parse_sensors_legacy

    return {**parse_fun(msg), **_parse_sensors_common(msg)}


def parse_sensor_event(msg):
    sensors = _parse_sensors(msg)
    if sensors[COMPENSATE_COLUMN] is not True:
        sensors = compensate_sensors(sensors)

    # HACK for some daikin cloud devices the humidity can be missing, in order
    # to prevent having to train separate models without humidity devired
    # features, we simply put a default humidity value. using just nan with
    # models gives bad result as humidex cannot be computed, for simplicity
    # it's better just to use a default value, rather than trying to fix that
    # in the pipeline imputer.
    if math.isnan(sensors["humidity"]):
        sensors["humidity"] = AVERAGE_HUMIDITY
    return sensors


FEEDBACK_SCHEMA = Schema(
    {
        Required("feedback"): All(
            Coerce(float), Range(min=MIN_COMFORT, max=MAX_COMFORT)
        ),
        Required("user_id"): str,
        Required("created_on"): datetime,
    },
    extra=REMOVE_EXTRA,
)


def parse_feedback_event(msg):
    feedback = parse.lower_dict(msg)
    return FEEDBACK_SCHEMA(feedback)


def parse_irprofile_event(msg):
    return msg["irfeature"]


def parse_quantity_field(field: str) -> ModePrefKey:
    quantity = None
    threshold_type = None

    if field in COMFORT_MODES:
        control_mode = "comfort"
        quantity = "comfort"
    elif "away" in field:
        control_mode, quantity, threshold_type = field.split("_")
    elif field == "temperature":
        control_mode = quantity = field
    elif field == "managed_manual":
        control_mode = field
        quantity = "set_temperature"
    elif field in ["manual", "off"]:
        control_mode = field
    else:
        log.error(f"unknown control mode quantity {field}, " "default to manual mode")
        control_mode = "manual"
    return ModePrefKey(control_mode, quantity, threshold_type)


def extract_control_target(
    control_target: ControlTarget,
) -> Tuple[str, Optional[str], Optional[str], Optional[float], Optional[float]]:
    """Extracts control information and constrains from control_target.

    control_mode: (comfort, away, temperature, etc.)
    quantity: (humidex, temperature, humidity, etc.)
    threshold_type: (upper, lower)
    threshold: threshold value
    target_value: target value of that quantity

    Args:
        control_target:

    Returns:

    """

    control_mode, quantity, threshold_type = parse_quantity_field(
        control_target["quantity"]
    )

    if control_mode == "away":
        threshold = control_target["value"]

        # by using mode model in away mode, we need a target to make
        # prediction, lets assume the goal of away mode is to save energy
        # then theoretically we want to set target to threshold.
        # We may need to keep reviewing whether this is the best choice.
        # 1/20/2017 Dominic
        target_value = threshold
    elif control_mode in ["manual", "off"]:
        target_value = None
        threshold = None
    else:
        target_value = control_target["value"]
        threshold = None

    return control_mode, quantity, threshold_type, threshold, target_value


def parse_mode_prefs(
    msg: Dict[str, Any], multimodes: List[str] = MULTIMODES
) -> ModePref:
    msg = parse.lower_dict(msg)
    key = parse_quantity_field(msg["quantity"])
    modes = [m for m in multimodes if msg[m]]
    return ModePref(key=key, modes=modes)


MODE_FEEDBACK_SCHEMA = Schema(
    {
        Required("mode_feedback"): In(MODES),
        Required("device_id"): str,
        Required("created_on"): datetime,
    },
    extra=REMOVE_EXTRA,
)


def parse_mode_feedback_event(msg: Dict[str, Any]):
    mode_feedback = parse.lower_dict(msg)
    return MODE_FEEDBACK_SCHEMA(mode_feedback)


NEARBY_USER_ACTION_SCHEMA = Schema(
    {
        Required("action"): In(
            [
                NearbyUser.DEVICE_CLEAR.value,
                NearbyUser.USER_IN.value,
                NearbyUser.USER_OUT.value,
            ]
        ),
        Required("user_id"): voluptuous.Any(None, str),
    },
    extra=REMOVE_EXTRA,
)


def parse_nearby_user_event(msg: Dict[str, Any]) -> NearbyUserAction:
    nearby_user_action = NEARBY_USER_ACTION_SCHEMA(parse.lower_dict(msg))
    return NearbyUserAction(
        action=NearbyUser(nearby_user_action["action"]),
        user_id=nearby_user_action["user_id"],
    )


AUTOMATED_DEMAND_RESPONSE_SCHEMA = Schema(
    {
        Required("action"): In(["start", "stop"]),
        Required("signal_level"): All(int, Range(min=0)),
        Required("create_time"): datetime,
        voluptuous.Optional("group_name", default=""): str,
    },
    extra=REMOVE_EXTRA,
)


def parse_automated_demand_response(msg: Dict[str, Any]) -> AutomatedDemandResponse:
    adr = AUTOMATED_DEMAND_RESPONSE_SCHEMA(parse.lower_dict(msg))
    adr["created_on"] = adr["create_time"]
    del adr["create_time"]
    return AutomatedDemandResponse(**adr)


DK_VENTILATION_OPTION_SCHEMA = Schema(
    {Required("value"): In(["on", "off"])}, extra=REMOVE_EXTRA
)
