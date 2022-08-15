from datetime import datetime, timedelta
from typing import Any, Awaitable, Dict, Iterator, List, Optional

from aredis import StrictRedis as Redis

from ..utils import cache_util
from ..utils.async_util import multi
from ..utils.ir_feature import (
    AMBI_CONTROL_BUTTONS,
    NoButtonError,
    best_possible_deployment,
    get_lowest_fan_settings,
)
from ..utils.log_util import get_logger
from ..utils.misc import elapsed
from ..utils.types import (
    ApplianceState,
    AutomatedDemandResponse,
    BasicDeployment,
    Connections,
    DeploymentSettings,
    IRFeature,
    ModePref,
    StateTemperature,
)

log = get_logger(__name__)

N_INVALID_LOG = 3
BETTER_SET_TEMP_OFFSET = 0.5

AWAY_MODE_WINDOWS = {"temperature": 2.0, "humidity": 7.0}

# We create controller upon receiving sensor event if update interval
# seconds have passed. If the sensor event seconds and update interval
# are almost equal then we will not create a controller if we receive
# a sensor event just before the update interval. The frequency
# of sensor event is stochastic and on average is equal to interval
# seconds.
AMBI_SENSOR_INTERVAL_SECONDS = 30
DAIKIN_SENSOR_INTERVAL_SECONDS = 60
UPDATE_INTERVAL_SECONDS = max(
    AMBI_SENSOR_INTERVAL_SECONDS, DAIKIN_SENSOR_INTERVAL_SECONDS
)
UPDATE_INTERVAL = timedelta(seconds=UPDATE_INTERVAL_SECONDS)
SIGNAL_INTERVAL = timedelta(minutes=3)

TRACKING_QUANTITIES = ["temperature", "comfort"]
TRIGGER_MANAGED_MANUAL_DURING_ADR_WHEN_IN_MANUAL = True
ADR_MAX_DURATION = timedelta(hours=4)
ACTIVE_CONTROL_MODES = (
    "comfort",
    "temperature",
    "away",
    "managed_manual",
    "off",  # needs to create a controller in off mode to turn the appliance off
)


def group_by(key, rows):
    ret = {}
    for row in rows:
        row_copy = row.copy()
        row_key = row_copy.pop(key)
        ret[row_key] = row_copy
    return ret


def needs_active_control(
    control_mode: str, automated_demand_response: Optional[AutomatedDemandResponse]
) -> bool:
    return control_mode in ACTIVE_CONTROL_MODES or needs_managed_manual(
        control_mode, automated_demand_response
    )


def needs_managed_manual(
    control_mode: str, automated_demand_response: Optional[AutomatedDemandResponse]
):
    return (
        control_mode == "manual"
        and is_active(automated_demand_response)
        and TRIGGER_MANAGED_MANUAL_DURING_ADR_WHEN_IN_MANUAL
    )


def is_active(automated_demand_response: Optional[AutomatedDemandResponse]) -> bool:
    if automated_demand_response:
        return automated_demand_response.action == "start" and not elapsed(
            automated_demand_response.created_on, ADR_MAX_DURATION
        )
    return False


def needs_feedback_control(control_mode: str) -> bool:
    return control_mode == "comfort"


def needs_sensors_update(
    control_mode: str,
    automated_demand_response: Optional[AutomatedDemandResponse],
    last_state_update: Optional[datetime],
    last_deployment: Optional[datetime],
) -> bool:
    """Checks sensor update conditions.

    Args:
        control_mode:
        automated_demand_response:
        last_state_update:
        last_deployment:

    Returns:

    """
    return all(
        [
            needs_active_control(control_mode, automated_demand_response),
            elapsed(last_state_update, UPDATE_INTERVAL),
            elapsed(last_deployment, SIGNAL_INTERVAL),
        ]
    )


def needs_climate_model_metric(control_mode: str) -> bool:
    return control_mode in ["off", "manual"]


def away_mode_conditions(
    quantity: str, threshold_type: str, threshold: float, current: float
) -> str:
    """Determine the room status.

    Args:
        quantity:
        threshold_type:
        threshold:
        current:

    Returns:
        str: either good, bad ot close

    """

    if threshold_type == "lower":
        if current <= threshold:
            state = "bad"
        elif current >= threshold + AWAY_MODE_WINDOWS[quantity]:
            state = "good"
        else:
            state = "close"
    elif threshold_type == "upper":
        if current >= threshold:
            state = "bad"
        elif current <= threshold - AWAY_MODE_WINDOWS[quantity]:
            state = "good"
        else:
            state = "close"
    else:
        raise ValueError("threshold type {} not supported".format(threshold_type))

    return state


def away_mode_action(
    is_on: bool, condition: str, timed_out: bool, new: bool
) -> Optional[str]:
    # FIXME: better naming (e.g. what timed out, mode or deployment)

    action = None
    # stop condition triggered when:
    # 1. if the condition is "good"
    # 2. last deployment is timed out AND control target is not newly recreated
    #   (just to make sure last deployment is responding last control target?)
    # 3.
    stop_conditions = [
        condition == "good",
        timed_out and not new,
        new and condition == "close",
    ]
    if is_on and any(stop_conditions):
        action = "off"
    elif condition == "bad" and (not is_on or new):
        action = "update"
    return action


def state_update_required(
    state: ApplianceState, signal: ApplianceState, target_delta: float
) -> bool:
    """Check if proposed signal needs to be deployed."""
    if state["power"] != signal["power"]:
        return True

    if state["mode"] != signal["mode"]:
        return True

    if has_same_temperature(signal["temperature"], state["temperature"]):
        return False

    return likely_better(signal["temperature"], state["temperature"], target_delta)


def has_same_temperature(
    new_temperature: Optional[str], current_temperature: StateTemperature
) -> bool:
    if str(new_temperature) == str(current_temperature):
        return True

    try:
        new_t = float(new_temperature)  # type: ignore
        old_t = float(current_temperature)  # type: ignore
    except (TypeError, ValueError):
        return False

    return new_t == old_t


def likely_better(
    new_temperature: Optional[str],
    old_temperature: StateTemperature,
    target_delta: float,
) -> bool:
    try:
        new_t = float(new_temperature)  # type: ignore
        old_t = float(old_temperature)  # type: ignore
    except (TypeError, ValueError):
        return True

    if target_delta < -BETTER_SET_TEMP_OFFSET and new_t > old_t:
        return False
    if target_delta > BETTER_SET_TEMP_OFFSET and new_t < old_t:
        return False
    return True


def needs_mode_pref_update(
    last_mode_pref: Optional[ModePref], new_mode_pref: ModePref
) -> bool:
    if not last_mode_pref:
        return True

    key_equal = last_mode_pref.key == new_mode_pref.key
    modes_changed = last_mode_pref.modes != new_mode_pref.modes
    if key_equal and modes_changed:
        return True

    return False


def chunks(ls: List, n: int) -> Iterator[List]:
    for i in range(0, len(ls), n):
        yield ls[i : i + n]


async def get_fan_setting(
    redis: Redis, appliance_id: str, mode: str, ir_feature: IRFeature
) -> Optional[str]:

    fan_setting = await cache_util.get_fan_redis(redis, appliance_id, mode)
    if not fan_setting:
        try:
            fan_settings = ir_feature[mode]["fan"]["value"]
        except KeyError:
            fan_settings = []
        fan_setting = get_lowest_fan_settings(fan_settings)
    return fan_setting


async def adjust_deployment_settings(
    connections: Connections,
    basic_deployment: BasicDeployment,
    device_id: str,
    appliance_id: str,
    state: ApplianceState,
    ir_feature: IRFeature,
) -> DeploymentSettings:
    """Fetches current and last appliance state to make a better deployment decision.

    Args:
        connections:
        basic_deployment:
        device_id:
        appliance_id:
        state:
        ir_feature:

    Returns:

    """

    coroutines: Dict[str, Awaitable[Any]] = {
        "last_on_state": cache_util.fetch_on_state(connections, device_id)
    }
    if basic_deployment.mode is not None:
        coroutines["fan"] = get_fan_setting(
            connections.redis, appliance_id, basic_deployment.mode, ir_feature
        )
    data = await multi(coroutines)
    settings = basic_deployment._asdict()

    if "fan" in data:
        settings["fan"] = data["fan"]

    deployment_settings = best_possible_deployment(
        state, data["last_on_state"], settings, ir_feature, device_id
    )

    if deployment_settings.button not in AMBI_CONTROL_BUTTONS:
        raise NoButtonError(
            f"button {deployment_settings.button} not controlled by skynet"
        )

    return deployment_settings


async def handle_invalid_sensors(redis, device_id, exc, msg):
    count = await cache_util.incr_invalid_sensors_count(redis, device_id)
    logged = await cache_util.get_invalid_sensors_logged(redis, device_id)
    if not logged and count >= N_INVALID_LOG:
        log_method = log.error
        await cache_util.set_invalid_sensors_logged(redis, device_id)
    else:
        log_method = log.info

    log_method(
        f"invalid event service data {exc}",
        extra={"data": {"device_id": device_id, "msg": msg}},
    )


class LogMixin:
    device_id: str

    def log(self, msg="", *, level="info", **kwargs):
        """Log 'msg' with device information."""
        extra = {"data": kwargs}
        extra["data"]["device_id"] = self.device_id
        try:
            meth = getattr(log, level)
        except AttributeError:
            raise ValueError("unknown logging level {}".format(level))

        meth(msg, extra=extra)
