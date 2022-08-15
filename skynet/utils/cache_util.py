"""Provides extra functionalities related to CRUD on cache.

"""

import asyncio
import itertools
import math
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from ambi_utils.zmq_micro_service import msg_util
from aredis import StrictRedis as Redis

from skynet.sample.selection import TIMESERIES_INTERVAL
from . import event_parsing, events, json, thermo
from .async_util import multi
from .database import queries
from .database.cassandra_queries import query_feedback_humidex
from .db_service import db_service_util
from .enums import NearbyUser, Power, VentilationState
from .log_util import get_logger
from .misc import elapsed
from .redis_util import redis_retry
from .types import (
    ApplianceID,
    ApplianceState,
    AutomatedDemandResponse,
    ComfortPrediction,
    Connections,
    ControlTarget,
    DeviceID,
    Feedback,
    IRFeature,
    ModeFeedback,
    ModePref,
    ModePrefKey,
    NearbyUserAction,
    Sensors,
    Timezone,
    UserIdSet,
    Weather,
)

log = get_logger(__name__)
WEATHER_INTERVAL = timedelta(hours=1)
WEATHER_CACHE_INTERVAL = timedelta(days=1)
ELAPSED_CACHE_INTERVAL = timedelta(minutes=40)
IR_FEATURE_CACHE_SECOND = 30 * 60
COMFORT_STATS_CACHE_SECOND = 24 * 60 * 60
CONTROLLER_CACHE_SECOND = 15 * 60
CONTROLLER_REFETCH_SECOND = 10 * 60
COMFORT_PREDICTION_CACHE_SECOND = 24 * 60 * 60
FEEDBACK_HUMIDEX_CACHE_SECOND = 3600 * 24 * 30
EMPTY_SET_TOKEN = ""


@redis_retry
async def set_redis(
    *,
    redis: Redis,
    key_arg: str,
    value: Any,
    key_fun: Callable[[str], str],
    redis_encode: Callable[[Any], Any],
    expiry: float = None,
) -> None:
    key = key_fun(key_arg)
    await redis.set(key, redis_encode(value), ex=expiry)


@redis_retry
async def get_redis(
    *,
    redis: Redis,
    key_arg: str,
    key_fun: Callable[[str], str],
    redis_decode: Callable[[Any], Any],
    default_value: Union[Any, str] = "raise",
) -> Any:
    """

    Args:
        redis:
        key_arg:
        key_fun:
        redis_decode:
        default_value:

    Returns:
        payload: Decoded data stored in Redis with certain key

    Raises:
        LookupError: When provided key does not exist

    """
    key = key_fun(key_arg)
    stored = await redis.get(key)
    if stored is None:
        if default_value == "raise":
            raise LookupError(f"no value stored for {key}")
        return default_value
    return redis_decode(stored)


set_json = partial(set_redis, redis_encode=json.dumps)
get_json = partial(get_redis, redis_decode=json.loads)


def _key(key: str) -> str:
    return "skynet:{}".format(key)


def ir_feature_key(appliance_id: str) -> str:
    return _key(f"ir_feature:appliance_id:{appliance_id}")


set_ir_feature = partial(
    set_json, key_fun=ir_feature_key, expiry=IR_FEATURE_CACHE_SECOND
)
get_ir_feature = partial(get_json, key_fun=ir_feature_key)


def _feedback_humidex_key(device_id: str, created_on: datetime) -> str:
    return _key(
        "feedback_humidex:device_id:{}:created_on:{}".format(
            device_id, created_on.isoformat()
        )
    )


async def _set_feedback_humidex(redis: Redis, key: str, value: Optional[float]) -> None:
    await redis.set(key, json.dumps(value), ex=FEEDBACK_HUMIDEX_CACHE_SECOND)


@redis_retry
async def fetch_feedback_humidex(
    connections: Connections, device_id: str, created_on: datetime
) -> Optional[float]:

    key = _feedback_humidex_key(device_id, created_on)
    saved = await connections.redis.get(key)

    # isnan check due to compensation bug. Remove when no longer any NaN
    # values in redis.f

    if saved is not None:
        value = json.loads(saved)
    else:
        value = None

    if value is None or math.isnan(value):
        log.debug("fetching feedback humidex from db")
        value = await query_feedback_humidex(connections.session, device_id, created_on)
        await _set_feedback_humidex(connections.redis, key, value)
    return value


def _datetime_to_score(timestamp: datetime) -> float:
    return timestamp.timestamp()


def _weather_key(device_id: str) -> str:
    return _key("weather:v2:device_id:{}".format(device_id))


@redis_retry
async def get_weather_redis(
    redis: Redis, device_id: str, start: datetime, end: datetime
) -> List[Weather]:
    key = _weather_key(device_id)
    cached = await redis.zrangebyscore(
        key, _datetime_to_score(start), _datetime_to_score(end)
    )
    if cached:
        weather = [json.loads(item) for item in cached]

        # exclude end if we just fetched it to make this match MySQL queries
        if weather[-1]["timestamp"] == end:
            weather = weather[:-1]
        return weather
    return []


@redis_retry
async def set_weather_redis(
    redis: Redis, device_id: str, weather: List[Weather]
) -> None:
    key = _weather_key(device_id)
    items = [json.dumps(w) for w in weather]
    scores = [_datetime_to_score(w["timestamp"]) for w in weather]
    args = itertools.chain.from_iterable(zip(scores, items))
    async with await redis.pipeline(transaction=True) as pipe:
        await pipe.zadd(key, *args)
        await clean_weather_redis(pipe, key)
        await pipe.execute()


@redis_retry
async def clean_weather_redis(redis: Redis, key: str) -> None:
    remove_until = datetime.utcnow() - WEATHER_CACHE_INTERVAL
    await redis.zremrangebyscore(key, "-inf", _datetime_to_score(remove_until))


async def fetch_weather_from_device(
    connections: Connections, device_id: str, start: datetime, end: datetime
) -> List[Weather]:
    cached = await get_weather_redis(connections.redis, device_id, start, end)

    if cached:
        fetch_start = cached[-1]["timestamp"] + WEATHER_INTERVAL
        if fetch_start > end:
            return cached

    else:
        fetch_start = start

    rows = await connections.pool.execute(
        *queries.query_weather_from_device(device_id, fetch_start, end)
    )

    # fetch weather from device returns data in descending time order
    # revert it
    new_weather = list(rows)[::-1]

    if new_weather:
        await set_weather_redis(connections.redis, device_id, new_weather)

    return cached + new_weather


def _online_device_key() -> str:
    return _key("active_controller:v2")


@redis_retry
async def set_online_device(redis: Redis, device_id: str) -> None:
    """Add device ID into online_devices list.

    An event from particular device is received. That mean

    Args:
        redis:
        device_id:

    Returns:

    """
    key = _online_device_key()
    score = _datetime_to_score(datetime.utcnow())
    await redis.zadd(key, score, device_id)


@redis_retry
async def get_online_devices(redis: Redis) -> List[str]:
    """Queries online devices from redis.

    Args:
        redis:

    Returns:
        List[str]: list of device IDs

    """
    key = _online_device_key()
    await clean_online_devices(redis)  # refresh online devices list first
    cached = await redis.zrange(key, 0, -1)
    return cached


@redis_retry
async def clean_online_devices(
    redis: Redis,
) -> None:  # FIXME: naming is not very actuate
    """Remove expired device IDs.

    Expiration is defined as (now - last_active_time) <= CONTROLLER_CACHE_SECOND. (to be verified)

    Args:
        redis:

    Returns:

    """
    key = _online_device_key()
    remove_until = _datetime_to_score(datetime.utcnow()) - CONTROLLER_CACHE_SECOND
    await redis.zremrangebyscore(key, "-inf", remove_until)


def create_events_for_online_devices(topic: str, prevent_overflow=True):
    """Insert device IDs into related topic (Not so sure).

    Args:
        topic:
        prevent_overflow:

    Returns:

    """

    @redis_retry
    async def wrapper(redis: Redis) -> None:
        key = _event_queue_key(topic)
        device_ids = await get_online_devices(redis)
        should_push = not prevent_overflow or await redis.llen(key) == 0
        if device_ids and should_push:
            events_ = [
                msg_util.encode(dict(device_id=device_id)) for device_id in device_ids
            ]
            await redis.lpush(key, *events_)

    return wrapper


def _fan_key(appliance_id: str) -> str:
    return _key("appliance_state:{}:fan".format(appliance_id))


@redis_retry
async def get_fan_redis(
    redis: Redis, appliance_id: str, control_mode: str
) -> Optional[str]:
    key = _fan_key(appliance_id)
    fan_setting = await redis.hget(key, control_mode)

    return fan_setting


@redis_retry
async def set_fan_redis(
    redis: Redis, appliance_id: str, control_mode: str, fan_setting: str
) -> None:
    key = _fan_key(appliance_id)
    await redis.hset(key, control_mode, fan_setting)


def _comfort_factors_key(user_id: str, device_id: str) -> str:
    return _key(f"analytics:{user_id}:{device_id}:comfort_factors")


@redis_retry
async def get_comfort_factors_redis(
    redis: Redis, user_id: str, device_id: str
) -> Optional[str]:
    key = _comfort_factors_key(user_id, device_id)
    saved = await redis.get(key)
    if saved is not None:
        saved = json.loads(saved)

    return saved


@redis_retry
async def set_comfort_factors_redis(
    redis: Redis, comfort_factors: Dict[str, Any], user_id: str, device_id: str
) -> None:
    key = _comfort_factors_key(user_id, device_id)
    await redis.set(key, json.dumps(comfort_factors), ex=COMFORT_STATS_CACHE_SECOND)


def _sensors_key(device_id: str) -> str:
    return _key(f"sensors:v3:{device_id}")


@redis_retry
async def get_sensors_range(
    redis: Redis, device_id: str, start: datetime, end: datetime
) -> List[Sensors]:
    """Queries historical sensor data.

    Args:
        redis:
        device_id:
        start:
        end:

    Returns:

    """
    key = _sensors_key(device_id)
    cached = await redis.zrangebyscore(
        key, _datetime_to_score(start), _datetime_to_score(end)
    )
    if cached:
        sensors = [json.loads(item) for item in cached]
        return sensors
    return []


@redis_retry
async def get_sensors(redis: Redis, key_arg: str) -> Sensors:
    """Queries the latest sensor readings.

    Args:
        redis:
        key_arg:

    Returns:

    """
    device_id = key_arg
    key = _sensors_key(device_id)
    cached = await redis.zrevrange(key, 0, 0)
    if cached:
        sensors = json.loads(cached[0])
        if not elapsed(sensors["created_on"], ELAPSED_CACHE_INTERVAL):
            return sensors

        raise LookupError(f"outdated sensors for {key}")

    raise LookupError(f"no value stored for {key}")


@redis_retry
async def set_sensors(redis: Redis, key_arg: str, value: Sensors) -> None:
    """Puts sensor data into Redis, also resets metadata and cleans historical data.

    Args:
        redis:
        key_arg:
        value:

    Returns:

    """
    device_id, sensors = key_arg, value
    sensors["humidex"] = thermo.humidex(sensors["temperature"], sensors["humidity"])
    key = _sensors_key(device_id)
    score = _datetime_to_score(sensors["created_on"])
    async with await redis.pipeline(transaction=True) as pipe:
        await pipe.zadd(key, score, json.dumps(sensors))
        await reset_invalid_sensors(pipe, device_id)
        await clean_sensors(pipe, key)
        await pipe.execute()


@redis_retry
async def clean_sensors(redis: Redis, key: str) -> None:
    """Remove expired historical sensor readings.

    Args:
        redis:
        key:

    Returns:

    """
    remove_until = datetime.utcnow() - TIMESERIES_INTERVAL
    await redis.zremrangebyscore(key, "-inf", _datetime_to_score(remove_until))


def _invalid_sensors_count_key(device_id: str) -> str:
    return _key(f"sensors:missing:count:{device_id}")


def _invalid_sensors_logged_key(device_id: str) -> str:
    return _key(f"sensors:missing:logged:{device_id}")


@redis_retry
async def incr_invalid_sensors_count(redis: Redis, device_id: str) -> int:
    key = _invalid_sensors_count_key(device_id)
    return await redis.incr(key)


@redis_retry
async def get_invalid_sensors_count(redis: Redis, device_id: str) -> int:
    key = _invalid_sensors_count_key(device_id)
    return int(await redis.get(key) or 0)


@redis_retry
async def reset_invalid_sensors(redis: Redis, device_id: str) -> None:
    """Removes device ID from invalid-sensors set.

    Args:
        redis:
        device_id:

    Returns:

    """
    for key in [
        _invalid_sensors_count_key(device_id),
        _invalid_sensors_logged_key(device_id),
    ]:
        await redis.delete(key)


@redis_retry
async def get_invalid_sensors_logged(redis: Redis, device_id: str) -> bool:
    key = _invalid_sensors_logged_key(device_id)
    return bool(await redis.get(key))


@redis_retry
async def set_invalid_sensors_logged(redis: Redis, device_id: str) -> None:
    key = _invalid_sensors_logged_key(device_id)
    await redis.set(key, 1)


def _appliance_state_key(device_id: str) -> str:
    return _key(f"appliance_state:{device_id}")


def _on_state_key(device_id: str) -> str:
    return _key(f"on_state:{device_id}")


def _control_target_key(device_id: str) -> str:
    return _key(f"control_target:{device_id}")


def _comfort_prediction_key(tuple_id: Tuple[str, str]) -> str:
    device_id, user_id = tuple_id
    return _key(f"comfort_prediction:{device_id}:{user_id}")


def _timezone_key(device_id: str) -> str:
    return _key(f"timezone:v2:{device_id}")


def _last_deployment_key(device_id: str) -> str:
    return _key(f"last_deployment:{device_id}")


get_last_deployment = partial(
    get_json, key_fun=_last_deployment_key, default_value=None
)
set_last_deployment = partial(set_json, key_fun=_last_deployment_key)


def _last_on_ventilation_key(device_id: str) -> str:
    return _key(f"last_on_ventilation:{device_id}")


get_last_on_ventilation = partial(
    get_json, key_fun=_last_on_ventilation_key, default_value=None
)
set_last_on_ventilation = partial(set_json, key_fun=_last_on_ventilation_key)

get_appliance_state = partial(get_json, key_fun=_appliance_state_key)
set_appliance_state = partial(
    set_json, key_fun=_appliance_state_key, expiry=CONTROLLER_REFETCH_SECOND
)


get_on_state = partial(get_json, key_fun=_on_state_key)
set_on_state = partial(
    set_json, key_fun=_on_state_key, expiry=CONTROLLER_REFETCH_SECOND
)


def _last_off_state_timestamp_key(device_id: str) -> str:
    return _key(f"last_off_state_timestamp:{device_id}")


get_last_off_state_timestamp = partial(
    get_json, key_fun=_last_off_state_timestamp_key, default_value=None
)
set_last_off_state_timestamp = partial(set_json, key_fun=_last_off_state_timestamp_key)


async def update_appliance_state(
    connections: Connections, device_id: str, state_to_set: ApplianceState
) -> ApplianceState:
    try:
        current_state = await fetch_appliance_state(connections, device_id)
    except LookupError:
        return state_to_set

    else:
        current_state.update(state_to_set)
        return current_state


async def update_last_on_ventilation(connections, device_id, new_state):
    if new_state.get("ventilation") == VentilationState.ON:
        await set_last_on_ventilation(
            redis=connections.redis, key_arg=device_id, value=new_state["created_on"]
        )


async def set_appliance_state_all(
    connections: Connections, key_arg: str, value: ApplianceState
) -> None:
    device_id = key_arg
    await update_last_on_ventilation(connections, device_id, value)
    state = await update_appliance_state(connections, device_id, value)
    async with await connections.redis.pipeline(transaction=True) as pipe:
        if "fan" in state:
            await set_fan_redis(
                pipe, state["appliance_id"], state["mode"], state["fan"]
            )
        await set_last_deployment(
            redis=pipe, key_arg=device_id, value=state["created_on"]
        )
        await set_appliance_state(redis=pipe, key_arg=device_id, value=state)
        if state["power"] == Power.ON:
            await set_on_state(redis=pipe, key_arg=device_id, value=state)
        if state["power"] == Power.OFF:
            await set_last_off_state_timestamp(
                redis=pipe, key_arg=device_id, value=state["created_on"]
            )
        await pipe.execute()


get_control_target = partial(get_json, key_fun=_control_target_key)
set_control_target = partial(
    set_json, key_fun=_control_target_key, expiry=CONTROLLER_REFETCH_SECOND
)


def _last_control_mode_key(device_id: str) -> str:
    return _key(f"last_control_mode:{device_id}")


get_last_control_mode = partial(
    get_json, key_fun=_last_control_mode_key, default_value=None
)
set_last_control_mode = partial(set_json, key_fun=_last_control_mode_key)


async def set_control_target_all(
    connections: Connections, key_arg: str, value: ControlTarget
) -> None:
    device_id = key_arg
    last_control_mode = await fetch_control_mode(connections, device_id)
    async with await connections.redis.pipeline(transaction=True) as pipe:
        await set_control_target(redis=pipe, key_arg=device_id, value=value)
        await set_last_control_mode(
            redis=pipe, key_arg=device_id, value=last_control_mode
        )
        await pipe.execute()


def _nearby_users_key(device_id: str) -> str:
    return _key(f"nearby_users:{device_id}")


@redis_retry
async def get_nearby_users(redis: Redis, key_arg: str) -> UserIdSet:
    device_id = key_arg
    key = _nearby_users_key(device_id)
    current_members = await redis.smembers(key)
    if current_members == set():
        raise LookupError(f"No nearby users stored for this device id {device_id}")
    current_members.discard(EMPTY_SET_TOKEN)
    return current_members


@redis_retry
async def set_nearby_users(redis: Redis, key_arg: str, value: NearbyUserAction) -> None:
    device_id = key_arg
    key = _nearby_users_key(device_id)
    need_to_set_expiry = not await is_the_key_set_for_nearby_users(redis, device_id)
    log.info(
        f"multi_user_comfort: Received event {value.action} for "
        f"{device_id} with user_id {value.user_id}"
    )

    async with await redis.pipeline(transaction=True) as pipe:
        if value.action == NearbyUser.DEVICE_CLEAR:
            await pipe.delete(key)
            await pipe.sadd(key, EMPTY_SET_TOKEN)
        elif value.action == NearbyUser.USER_IN:
            await pipe.sadd(key, value.user_id)
        elif value.action == NearbyUser.USER_OUT:
            await pipe.sadd(key, EMPTY_SET_TOKEN)
            await pipe.srem(key, value.user_id)
        if need_to_set_expiry:
            await pipe.expire(key, CONTROLLER_REFETCH_SECOND)
        await pipe.execute()


async def is_the_key_set_for_nearby_users(redis: Redis, device_id: str) -> bool:
    try:
        await get_nearby_users(redis, device_id)
    except LookupError:
        return False
    return True


def _latest_feedbacks_key(device_id: str) -> str:
    return _key(f"latest_feedbacks:{device_id}")


@redis_retry
async def get_latest_feedbacks(redis: Redis, key_arg: str) -> List[Feedback]:
    device_id = key_arg
    key = _latest_feedbacks_key(device_id)
    latest_feedbacks = await redis.hgetall(key)
    if latest_feedbacks == {}:
        raise LookupError(f"no value stored for {key_arg}")
    return [json.loads(feedback) for feedback in latest_feedbacks.values()]


@redis_retry
async def set_latest_feedback(redis: Redis, key_arg: str, value: Feedback) -> None:
    device_id, feedback = key_arg, value
    key = _latest_feedbacks_key(device_id)
    need_to_set_expiry = not await is_key_set_for_latest_feedbacks(redis, device_id)

    async with await redis.pipeline(transaction=True) as pipe:
        await pipe.hset(key, feedback["user_id"], json.dumps(feedback))
        if need_to_set_expiry:
            await pipe.expire(key, CONTROLLER_REFETCH_SECOND)
        await pipe.execute()


async def is_key_set_for_latest_feedbacks(redis: Redis, device_id: str) -> bool:
    try:
        await get_latest_feedbacks(redis, device_id)
    except LookupError:
        return False
    return True


def _last_state_update_key(device_id: str) -> str:
    return _key(f"last_state_update:{device_id}")


# last update time
get_last_state_update = partial(
    get_json, key_fun=_last_state_update_key, default_value=None
)
set_last_state_update = partial(set_json, key_fun=_last_state_update_key)


async def fetch_control_mode(connections: Connections, key_arg: str) -> str:
    """Fetches control mode of device.

    Args:
        connections:
        key_arg:

    Returns:

    """
    control_target = await fetch_control_target(connections, key_arg)
    return event_parsing.extract_control_target(control_target)[0]


async def fetch_mode_preference_key(
    connections: Connections, key_arg: str
) -> ModePrefKey:
    control_target = await fetch_control_target(connections, key_arg)
    return ModePrefKey(*event_parsing.extract_control_target(control_target)[:3])


def _deviation_tracker_key(key: Tuple[str, str]) -> str:
    device_id, quantity = key
    return _key(f"deviation_tracker:v3:{device_id}:{quantity}")


get_deviation_tracker = partial(get_json, key_fun=_deviation_tracker_key)
set_deviation_tracker = partial(set_json, key_fun=_deviation_tracker_key)


def _maintain_tracker_key(key: str) -> str:
    device_id = key
    return _key(f"maintain:v3:{device_id}")


get_maintain_tracker = partial(
    get_json,
    key_fun=_maintain_tracker_key,
    default_value={
        "long_term_errors": [],
        "timestamps": [],
        "start": datetime.utcnow(),
        "errors": [],
        "seconds": [],
    },
)
set_maintain_tracker = partial(set_json, key_fun=_maintain_tracker_key)


def namedtuple_encode(n) -> str:
    return json.dumps(n._asdict() if n is not None else None)


def namedtuple_decode(namedtuple):
    def decode(s):
        d = json.loads(s)
        if d is None:
            return None
        return namedtuple(**d)

    return decode


get_comfort_prediction = partial(
    get_redis,
    redis_decode=namedtuple_decode(ComfortPrediction),
    key_fun=_comfort_prediction_key,
    default_value=None,
)
set_comfort_prediction = partial(
    set_redis,
    redis_encode=namedtuple_encode,
    key_fun=_comfort_prediction_key,
    expiry=COMFORT_PREDICTION_CACHE_SECOND,
)


get_timezone = partial(get_json, key_fun=_timezone_key)
set_timezone = partial(
    set_json, key_fun=_timezone_key, expiry=CONTROLLER_REFETCH_SECOND
)


def _mode_preferences_key(device_id: str) -> str:
    return _key(f"mode_preference:v2:{device_id}")


def _last_mode_preferences_key(device_id: str) -> str:
    return _key(f"last_mode_preference:v2:{device_id}")


def _mode_preferences_hash_key(mode_pref_key: ModePrefKey) -> str:
    return ":".join([k for k in mode_pref_key if k is not None])


def _mode_preferences_from_hash_key(hash_key: str) -> ModePrefKey:
    return ModePrefKey(*hash_key.split(":"))


def _mode_feedback_key(device_id: str) -> str:
    return _key(f"mode_feedback:{device_id}")


get_mode_feedback = partial(get_json, key_fun=_mode_feedback_key)
set_mode_feedback = partial(
    set_json, key_fun=_mode_feedback_key, expiry=CONTROLLER_REFETCH_SECOND
)


def _automated_demand_response_key(device_id: str) -> str:
    return _key(f"automated_demand_response:v2:{device_id}")


get_automated_demand_response: Callable[
    ..., Optional[AutomatedDemandResponse]
] = partial(
    get_redis,
    redis_decode=namedtuple_decode(AutomatedDemandResponse),
    key_fun=_automated_demand_response_key,
    default_value=None,
)
set_automated_demand_response = partial(
    set_redis,
    redis_encode=namedtuple_encode,
    key_fun=_automated_demand_response_key,
    expiry=None,  # XXX put expiry back once we can fetch from the db
)


@redis_retry
async def get_mode_preferences(redis: Redis, key_arg: str) -> List[ModePref]:
    device_id = key_arg
    redis_key = _mode_preferences_key(device_id)
    stored = await redis.hgetall(redis_key)
    # returns {} if not found
    if not stored:
        raise LookupError(f"no mode_preferences in redis for {device_id}")
    return [
        ModePref(_mode_preferences_from_hash_key(key), json.loads(value))
        for key, value in stored.items()
    ]


@redis_retry
async def set_mode_preference(redis: Redis, key_arg: str, value: ModePref) -> None:
    device_id = key_arg
    key = _mode_preferences_key(device_id)
    hash_key = _mode_preferences_hash_key(value.key)
    await redis.hset(key, hash_key, json.dumps(value.modes))
    await redis.expire(key, CONTROLLER_REFETCH_SECOND)


@redis_retry
async def set_last_mode_preference(redis: Redis, key_arg: str, value: ModePref) -> None:
    device_id = key_arg
    key = _last_mode_preferences_key(device_id)
    hash_key = _mode_preferences_hash_key(value.key)
    await redis.hset(key, hash_key, json.dumps(value.modes))


@redis_retry
async def fetch_mode_preference(
    connections: Connections, device_id: str, key: ModePrefKey
) -> Optional[ModePref]:
    mode_prefs = await fetch_mode_preferences(connections, device_id)
    for mode_pref in mode_prefs:
        if mode_pref.key == key:
            return mode_pref
    return None


@redis_retry
async def set_mode_preference_all(
    connections: Connections, key_arg: str, value: ModePref
) -> None:
    device_id = key_arg
    current_mode_pref = await fetch_mode_preference(connections, device_id, value.key)
    async with await connections.redis.pipeline(transaction=True) as pipe:
        if current_mode_pref:
            await set_last_mode_preference(
                redis=pipe, key_arg=device_id, value=current_mode_pref
            )
        await set_mode_preference(redis=pipe, key_arg=device_id, value=value)
        await pipe.execute()


@redis_retry
async def get_last_mode_preference(
    redis: Redis, device_id: str, key: ModePrefKey
) -> Optional[ModePref]:
    key_ = _last_mode_preferences_key(device_id)
    hash_key = _mode_preferences_hash_key(key)
    stored = await redis.hget(key_, hash_key)
    if not stored:
        return None

    return ModePref(key, json.loads(stored))


def identity(x, *_):
    return x


def make_cache(
    getter: Any,
    setter: Any,
    db_getter: Any,
    db_decode: Callable = identity,
    redis_decode: Callable = identity,
    redis_encode: Callable = identity,
    multiple_return: bool = False,
) -> Callable:
    """Fetches data from DB (or caches from Redis).

    Payload are also cached in Redis once queried instead of querying DB
    for every repeated requests.

    Args:
        getter:
        setter:
        db_getter:
        db_decode:
        redis_decode:
        redis_encode:
        multiple_return:

    Returns:

    """

    async def fetch(
        connections: Connections, key_arg: str, refetch: bool = False
    ) -> Dict[str, Any]:
        """Fetches data from db (or Redis if cached data is okay).

        Args:
            connections:
            key_arg:
            refetch:

        Returns:

        """
        if not refetch:
            try:
                return redis_decode(
                    await getter(redis=connections.redis, key_arg=key_arg)
                )
            except LookupError as exc:
                log.debug(exc)
                log.debug(
                    "redis cache miss: %s: %s",
                    getattr(
                        getter, "func", getter
                    ).__name__,  # FIXME: there is no "func" (?)
                    key_arg,
                )
        try:
            response = await db_getter(connections=connections, key_arg=key_arg)
        except (TypeError, asyncio.TimeoutError) as exc:
            raise LookupError from exc
        parsed = db_decode(response, key_arg)

        await multi(
            [
                setter(
                    redis=connections.redis, key_arg=key_arg, value=redis_encode(parse)
                )
                for parse in parsed
            ]
        )
        return parsed if multiple_return else parsed[0]

    return fetch


def query_decoder(
    rows: List,
    key_arg: str,
    query_name: str,
    row_decoder: Callable = identity,
    missing: Any = "raise",
    process_multiple_rows: bool = False,
) -> List:
    if rows:
        from_db = rows if process_multiple_rows else [rows[0]]
        parsed = [row_decoder(x) for x in from_db]
    else:
        if missing == "raise":
            raise LookupError("no data in mysql for " f"{query_name} for {key_arg}")
        parsed = [] if process_multiple_rows else [missing()]
    return parsed


def ir_feature_decoder(
    resp: Optional[IRFeature], _key_arg: str
) -> List[Optional[IRFeature]]:
    return [resp]


async def mysql_query_getter(
    connections: Connections, key_arg: str, query: Callable
) -> List[Dict]:
    return await connections.pool.execute(*query(device_id=key_arg))


async def db_service_ir_feature_getter(
    connections: Connections, key_arg: str
) -> IRFeature:
    return await db_service_util.fetch_ir_feature_from_db_service(
        db_service_msger=connections.db_service_msger, appliance_id=key_arg
    )


async def db_service_nearby_users_getter(
    connections: Connections, key_arg: str
) -> UserIdSet:
    return await db_service_util.fetch_nearby_users_from_db_service(
        db_service_msger=connections.db_service_msger, device_id=key_arg
    )


fetch_appliance_state: Callable[
    [Connections, DeviceID], Awaitable[ApplianceState]
] = make_cache(
    get_appliance_state,
    set_appliance_state,
    db_getter=partial(mysql_query_getter, query=queries.query_last_appliance_state),
    db_decode=partial(query_decoder, query_name="appliance state"),
)


fetch_on_state: Callable[
    [Connections, DeviceID], Awaitable[ApplianceState]
] = make_cache(
    get_on_state,
    set_on_state,
    db_getter=partial(mysql_query_getter, query=queries.query_last_on_appliance_state),
    db_decode=partial(query_decoder, query_name="on_state", missing=dict),
)


fetch_control_target: Callable[
    [Connections, DeviceID], Awaitable[ControlTarget]
] = make_cache(
    get_control_target,
    set_control_target,
    db_getter=partial(mysql_query_getter, query=queries.query_last_control_target),
    db_decode=partial(query_decoder, query_name="control_target"),
)


fetch_timezone: Callable[[Connections, DeviceID], Awaitable[Timezone]] = make_cache(
    get_timezone,
    set_timezone,
    db_getter=partial(mysql_query_getter, query=queries.query_device_timezone),
    db_decode=partial(query_decoder, query_name="timezone"),
)


fetch_mode_feedback: Callable[
    [Connections, DeviceID], Awaitable[ModeFeedback]
] = make_cache(
    get_mode_feedback,
    set_mode_feedback,
    db_getter=partial(mysql_query_getter, query=queries.query_last_mode_feedback),
    db_decode=partial(query_decoder, query_name="mode_feedback", missing=dict),
)


fetch_latest_feedbacks: Callable[
    [Connections, DeviceID], Awaitable[List[Feedback]]
] = make_cache(
    get_latest_feedbacks,
    set_latest_feedback,
    db_getter=partial(mysql_query_getter, query=queries.query_latest_feedbacks),
    db_decode=partial(
        query_decoder,
        query_name="latest_feedbacks",
        missing=list,
        process_multiple_rows=True,
    ),
    multiple_return=True,
)

fetch_ir_feature: Callable[
    [Connections, ApplianceID], Awaitable[Optional[IRFeature]]
] = make_cache(
    get_ir_feature,
    set_ir_feature,
    db_getter=db_service_ir_feature_getter,
    db_decode=ir_feature_decoder,
)


def add_nearby_user_encoding(value) -> NearbyUserAction:
    return NearbyUserAction(action=NearbyUser.USER_IN, user_id=value)


fetch_nearby_users: Callable[
    [Connections, DeviceID], Awaitable[UserIdSet]
] = make_cache(
    get_nearby_users,
    set_nearby_users,
    db_getter=db_service_nearby_users_getter,
    multiple_return=True,
    redis_encode=add_nearby_user_encoding,
)


# query result (from "DeviceModePreference")
# [{'auto': 1, 'cool': 1, 'created_on': datetime.datetime(2016, 11, 23, 9, 45, 7),
#   'dry': 0, 'fan': 1, 'heat': 1, 'quantity': 'Away_Humidity_Upper'},
#  {'auto': 0, 'cool': 0, 'created_on': datetime.datetime(2017, 1, 12, 5, 42, 18),
#   'dry': 1, 'fan': 1, 'heat': 0, 'quantity': 'Climate'},
#  {'auto': 1, 'cool': 1, 'created_on': datetime.datetime(2016, 11, 23, 9, 45, 39),
#   'dry': 1,'fan': 1, 'heat': 1, 'quantity': 'Temperature'}]
fetch_mode_preferences: Callable[
    [Connections, DeviceID], Awaitable[List[ModePref]]
] = make_cache(
    get_mode_preferences,
    set_mode_preference,
    db_getter=partial(mysql_query_getter, query=queries.query_last_mode_preferences),
    db_decode=partial(
        query_decoder,
        query_name="mode_preference",
        process_multiple_rows=True,
        missing=list,
        row_decoder=event_parsing.parse_mode_prefs,
    ),
    multiple_return=True,
)


def _event_queue_key(topic: str) -> str:
    return _key(f"event_queue:{topic}")


def decode_event(data: bytes) -> Any:
    event = msg_util.decode(data)
    timestamp_fields = ["created_on", "create_time"]
    for field in set(timestamp_fields).intersection(event):
        event[field] = event_parsing.to_naive_datetime(event[field])
    return event


@redis_retry
async def enqueue_event(redis: Redis, msg: List[bytes]) -> None:
    topic, data = msg
    key = _event_queue_key(topic.decode())
    await redis.lpush(key, data)


EVENT_KEYS = {_event_queue_key(topic): topic for topic in events.EVENT_REGISTRY}
PRIORITIZE_EVENT_TOPICS = sorted(
    events.EVENT_REGISTRY, key=lambda topic: events.EVENT_REGISTRY[topic].priority.value
)


@redis_retry
async def pick_event(redis: Redis) -> Tuple[str, Dict[str, Any]]:
    """Queries an event from Redis based on event priority.

    Args:
        redis:

    Returns:
        msg: topic name (str), event data (dict)

    """
    key, data = await redis.brpop(
        [_event_queue_key(topic) for topic in PRIORITIZE_EVENT_TOPICS]
    )

    return EVENT_KEYS[key], decode_event(data)


async def trim_queue(redis: Redis, topic, start, stop):
    key = _event_queue_key(topic)
    await redis.ltrim(key, start, stop)
