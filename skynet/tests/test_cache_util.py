import inspect
import math
from collections import namedtuple
from datetime import datetime, timedelta
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, List

import numpy as np
import pytest
from ambi_utils.zmq_micro_service import msg_util
from asynctest import mock
from freezegun import freeze_time

from ..control.util import AMBI_SENSOR_INTERVAL_SECONDS
from ..utils import cache_util, event_parsing, events
from ..utils.database import queries
from ..utils.enums import NearbyUser, Power
from ..utils.ir_feature import is_default_irfeature
from ..utils.log_util import get_logger
from ..utils.types import Connections, ModePref, ModePrefKey, NearbyUserAction
from .test_irfeature import get_irfeature

log = get_logger(__name__)


@pytest.fixture(
    params=[True, False], ids=["default ir feature", "not default ir feature"]
)
def ir_feature(request):
    return get_irfeature(request.param)


@pytest.fixture
def db_service_msger_with_irfeature(ir_feature, get_db_service_msger):
    response = {
        "data": ir_feature["data"],
        "appliance_id": ir_feature["appliance_id"],
        "result_code": ir_feature["result_code"],
    }
    return get_db_service_msger(code=200, response=response)


@pytest.fixture(
    params=[
        (500, {"reason": "db service not available"}),
        (200, {"bad": "bad ir feature"}),
    ],
    ids=["db service error", "malformed ir feature"],
)
def db_service_msger_with_exception(request, get_db_service_msger):
    return get_db_service_msger(code=request.param[0], response=request.param[1])


@pytest.fixture
def ir_feature_connections(rediscon, db_service_msger_with_irfeature):
    return Connections(redis=rediscon, db_service_msger=db_service_msger_with_irfeature)


@pytest.fixture
def ir_feature_connections_with_exception(rediscon, db_service_msger_with_exception):
    return Connections(redis=rediscon, db_service_msger=db_service_msger_with_exception)


@pytest.mark.asyncio
async def test_ir_feature_cache_exception(
    ir_feature_connections_with_exception, appliance_id
):
    with pytest.raises(LookupError):
        await cache_util.fetch_ir_feature(
            ir_feature_connections_with_exception, appliance_id, refetch=True
        )


@pytest.mark.parametrize("refetch", [True, False])
@pytest.mark.asyncio
async def test_ir_feature_cache_through_db_service(
    ir_feature_connections, ir_feature, appliance_id, refetch
):
    # when no ir feature is stored in redis or we ask for a refetch, make sure
    # we fetch it from the http client
    with pytest.raises(LookupError):
        await cache_util.get_ir_feature(
            redis=ir_feature_connections.redis, key_arg=appliance_id
        )

    result = await cache_util.fetch_ir_feature(
        ir_feature_connections, appliance_id, refetch
    )
    if is_default_irfeature(ir_feature.get("data")):
        assert result is None
    else:
        assert result == ir_feature.get("data")

    # and store it in redis afterwards
    redis_result = await cache_util.get_ir_feature(
        redis=ir_feature_connections.redis, key_arg=appliance_id
    )

    if is_default_irfeature(ir_feature.get("data")):
        assert result is None
    else:
        assert redis_result == ir_feature.get("data")


@pytest.fixture
async def connections_with_irfeature(rediscon, ir_feature, appliance_id):
    await cache_util.set_ir_feature(
        redis=rediscon, key_arg=appliance_id, value=ir_feature["data"]
    )
    yield Connections(redis=rediscon)


@pytest.mark.asyncio
async def test_ir_feature_cache_through_redis(
    ir_feature, connections_with_irfeature, appliance_id
):
    # when the ir feature is already in redis we should fetch it from here
    assert ir_feature["data"] == await cache_util.fetch_ir_feature(
        connections_with_irfeature, appliance_id
    )


@pytest.fixture
def connections_feedback_humidex(rediscon, cassandra_session):
    return Connections(redis=rediscon, session=cassandra_session)


@pytest.mark.asyncio
async def test_feedback_humidex_cache(
    connections_feedback_humidex, device_id, device_intervals
):
    t = device_intervals[0]["end"]
    fbh = await cache_util.fetch_feedback_humidex(
        connections_feedback_humidex, device_id, t
    )
    assert isinstance(fbh, float)
    fbh2 = await cache_util.fetch_feedback_humidex(
        connections_feedback_humidex, device_id, t
    )
    assert fbh == fbh2

    # check with no data
    t = device_intervals[0]["start"] - timedelta(days=1)
    fbh = await cache_util.fetch_feedback_humidex(
        connections_feedback_humidex, device_id, t
    )
    assert fbh is None


@pytest.mark.parametrize("bad_feedback_humidex", [np.nan, None])
@pytest.mark.asyncio
async def test_feedback_humidex_cache_regression_nan_refetch(
    connections_feedback_humidex, device_id, device_intervals, bad_feedback_humidex
):
    """Refetch from cassandra if bad NaN/None feedback_humidex in redis."""
    t = device_intervals[0]["end"]
    key = cache_util._feedback_humidex_key(device_id, t)

    await cache_util._set_feedback_humidex(
        connections_feedback_humidex.redis, key, bad_feedback_humidex
    )

    fbh = await cache_util.fetch_feedback_humidex(
        connections_feedback_humidex, device_id, t
    )

    assert isinstance(fbh, float)
    assert math.isfinite(fbh)


@pytest.fixture
def connections_weather(pool, rediscon):
    return Connections(pool=pool, redis=rediscon)


@pytest.mark.skip("somehow freezegun is messing things up")
@pytest.mark.flaky(reruns=2, reruns_delay=1)
@pytest.mark.asyncio
async def test_weather_cache(db, connections_weather, device_id):
    async def _fetch_cached(start, end):
        return await cache_util.fetch_weather_from_device(
            connections_weather, device_id, start, end
        )

    def _fetch_db(start, end):
        # reverse to sort in ascending time order
        return list(
            queries.execute(
                db, *queries.query_weather_from_device(device_id, start, end)
            )
        )[::-1]

    async def _fetch_redis(start, end):
        return await cache_util.get_weather_redis(
            connections_weather.redis, device_id, start, end
        )

    start = datetime.fromtimestamp(0)

    all_weather = _fetch_db(start, datetime.utcnow())

    weather_start = min(w["timestamp"] for w in all_weather)

    async def _compare(start, end):
        from_db = _fetch_db(start, end)
        # XXX quick hack as we clean data with respect to utcnow, to still be
        # able to use those tests
        from_db = _fetch_db(start, end)
        with freeze_time(end - timedelta(hours=1)):
            from_cache = await _fetch_cached(start, end)
        from_redis = await _fetch_redis(start, end)
        assert len(from_db) == len(from_cache)
        assert from_db == from_cache
        assert len(from_db) == len(from_redis)
        assert from_db == from_redis
        return from_db

    end = weather_start

    # set to first timestamp, to fetch nothing
    resp = await _compare(start, end)
    assert not resp

    # set end just after first timestamp, this should fetch one row
    end = weather_start + timedelta(seconds=1)
    resp = await _compare(start, end)
    assert len(resp) == 1

    # fetch one day of data
    end = weather_start + timedelta(days=1)
    resp = await _compare(start, end)
    assert len(resp) == 24

    # fetch one more day
    start = end
    end = start + timedelta(days=1)
    resp = await _compare(start, end)
    assert len(resp) == 24

    # check if expired data is removed
    all_cached = await _fetch_redis(datetime(2015, 1, 1), datetime(2099, 1, 1))
    assert len(resp) == len(all_cached)
    assert resp == all_cached

    # all data should be cached and end boundary not included
    end = start + timedelta(minutes=30)
    resp = await _compare(start, end)
    assert len(resp) == 1

    # all data should be cached, hitting end boundary
    end = start + timedelta(hours=1)
    resp = await _compare(start, end)
    assert len(resp) == 1


@pytest.mark.asyncio
async def test_fan_setting(rediscon):
    assert await cache_util.get_fan_redis(rediscon, "appliance_id", "cool") is None
    await cache_util.set_fan_redis(rediscon, "appliance_id", "cool", "low")
    await cache_util.set_fan_redis(rediscon, "appliance_id", "heat", "med")
    assert await cache_util.get_fan_redis(rediscon, "appliance_id", "cool") == "low"
    assert await cache_util.get_fan_redis(rediscon, "appliance_id", "heat") == "med"


@pytest.mark.asyncio
async def test_comfort_factors(rediscon):
    assert (
        await cache_util.get_comfort_factors_redis(rediscon, "user", "appliance")
        is None
    )
    data = [{"type": "Luminosity ", "value": 0.7, "data": 10}]
    await cache_util.set_comfort_factors_redis(rediscon, data, "user", "appliance")
    assert (
        await cache_util.get_comfort_factors_redis(rediscon, "user", "appliance")
        == data
    )


@pytest.mark.asyncio
async def test_sensors(rediscon, device_id, sensors):
    with pytest.raises(LookupError):
        await cache_util.get_sensors(redis=rediscon, key_arg=device_id)
    await cache_util.set_sensors(rediscon, device_id, sensors)
    assert "humidex" in sensors
    assert await cache_util.get_sensors(redis=rediscon, key_arg=device_id) == sensors


@pytest.mark.asyncio
async def test_elapsed_sensors(rediscon, device_id, sensors):
    await cache_util.set_sensors(rediscon, device_id, sensors)
    # make sure that the sensors was not removed already by the cleaning
    # cleaned
    await cache_util.get_sensors(redis=rediscon, key_arg=device_id)
    # sensors should now expire
    with mock.patch(
        "skynet.utils.cache_util.ELAPSED_CACHE_INTERVAL", timedelta(milliseconds=1)
    ):
        with pytest.raises(LookupError):
            await cache_util.get_sensors(redis=rediscon, key_arg=device_id)


@pytest.fixture
def sensors_range():
    sensors = [
        {
            "created_on": datetime.utcnow()
            - timedelta(seconds=i * AMBI_SENSOR_INTERVAL_SECONDS),
            "temperature": 10.1,
            "humidity": 10.2,
            "luminosity": 10.0,
        }
        for i in range(60)
    ]
    return sensors


@pytest.mark.asyncio
async def test_get_sensors_range(rediscon, device_id, sensors_range):
    for sensors in sensors_range:
        await cache_util.set_sensors(rediscon, device_id, sensors)
    end = datetime.utcnow()
    start = end - timedelta(minutes=20)

    sensors = await cache_util.get_sensors_range(rediscon, device_id, start, end)
    assert len(sensors) == 40


@pytest.mark.asyncio
async def test_invalid_sensors(rediscon, device_id):

    funs = []
    for name, fun in [
        ("get_count", cache_util.get_invalid_sensors_count),
        ("incr", cache_util.incr_invalid_sensors_count),
        ("get_logged", cache_util.get_invalid_sensors_logged),
        ("set_logged", cache_util.set_invalid_sensors_logged),
        ("reset", cache_util.reset_invalid_sensors),
    ]:
        funs.append([name, partial(fun, redis=rediscon, device_id=device_id)])

    funs = namedtuple("funs", [n for n, _ in funs])(*(f for _, f in funs))

    assert await funs.get_count() == 0
    assert await funs.get_logged() is False

    for res in [1, 2]:
        await funs.incr()
        assert await funs.get_count() == res
        assert await funs.get_logged() is False

    await funs.set_logged()
    assert await funs.get_logged() is True

    await funs.reset()
    assert await funs.get_count() == 0
    assert await funs.get_logged() is False


# XXX getter/setter should be Awaitable, but this causes problems with
# mypy as of mypy 0.521, try again with new release
# with mypy 0.560 we can replace Any by partial but not Awaitable yet


def _case(
    test_id: str,
    getter: Callable,
    setter: Callable,
    kwargs: List[Dict[str, Any]],
    stored: List[Any],
):
    return pytest.param(getter, setter, kwargs, stored, id=test_id)


DEVICE_ID = "device_id"
CREATED_ON = datetime(2015, 1, 1)

TEST_CASES = [
    _case(
        test_id="sensor",
        getter=cache_util.get_sensors,
        setter=cache_util.set_sensors,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[
            {
                "value": dict(
                    temperature=20.0,
                    humidity=50.0,
                    luminosity=100.0,
                    created_on=datetime.utcnow(),
                )
            }
        ],
    ),
    _case(
        test_id="appliance_state",
        getter=cache_util.get_appliance_state,
        setter=cache_util.set_appliance_state,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[
            {
                "value": dict(
                    appliance_id="my_appliance",
                    power=Power.OFF,
                    temperature=23,
                    louver="auto",
                    swing=None,
                    ventilation=None,
                    created_on=CREATED_ON,
                )
            }
        ],
    ),
    _case(
        test_id="on_state",
        getter=cache_util.get_on_state,
        setter=cache_util.set_on_state,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[
            {
                "value": dict(
                    appliance_id="my_appliance",
                    power=Power.ON,
                    temperature=23,
                    louver="auto",
                    swing=None,
                    ventilation=None,
                    created_on=CREATED_ON,
                )
            }
        ],
    ),
    _case(
        test_id="control_target",
        getter=cache_util.get_control_target,
        setter=cache_util.set_control_target,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[
            {
                "value": dict(
                    device_id=DEVICE_ID,
                    quantity="comfort",
                    value=None,
                    created_on=CREATED_ON,
                )
            }
        ],
    ),
    _case(
        test_id="last_state_update",
        getter=cache_util.get_last_state_update,
        setter=cache_util.set_last_state_update,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[{"value": CREATED_ON}],
    ),
    _case(
        test_id="timezone",
        getter=cache_util.get_timezone,
        setter=cache_util.set_timezone,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[{"value": dict(timezone="some_tz")}],
    ),
    _case(
        test_id="mode_feedback",
        getter=cache_util.get_mode_feedback,
        setter=cache_util.set_mode_feedback,
        kwargs=[dict(key_arg=DEVICE_ID)],
        stored=[{"value": dict(feedback="cool", created_on=CREATED_ON)}],
    ),
]


@pytest.mark.parametrize("getter,setter,kwargs,stored", TEST_CASES)
@pytest.mark.asyncio
async def test_get_set(rediscon, getter, setter, kwargs, stored):

    for kw, values in zip(kwargs, stored):
        _get = partial(getter, redis=rediscon, **kw)
        _set = partial(setter, redis=rediscon, **kw)

        params = inspect.signature(_get).parameters

        if "default_value" in params:
            default_value = params["default_value"].default
            if default_value == "raise":
                with pytest.raises(LookupError):
                    assert await _get()
            else:
                assert await _get() == default_value
        else:
            with pytest.raises(LookupError):
                assert await _get()

        await _set(**values)
        key = next(iter(values))
        assert await _get(**kw) == values[key]


FETCH_FUNCS = [
    pytest.param(cache_util.fetch_appliance_state, "raise", id="appliance_state"),
    pytest.param(cache_util.fetch_control_target, "raise", id="control_target"),
    pytest.param(cache_util.fetch_mode_preferences, [], id="mode_preferences"),
    pytest.param(cache_util.fetch_on_state, {}, id="on_state"),
    pytest.param(cache_util.fetch_timezone, "raise", id="timezone"),
    pytest.param(cache_util.fetch_mode_feedback, {}, id="mode_feedback"),
]


@pytest.fixture
def connections_cache(
    pool, rediscon, device_mode_preference_db_pool, db_service_msger
):  # pylint: disable=unused-argument
    return Connections(pool=pool, redis=rediscon, db_service_msger=db_service_msger)


@pytest.mark.flaky(reruns=3, reruns_delay=0.1)
@pytest.mark.parametrize("fetch_func, missing", FETCH_FUNCS)
@pytest.mark.asyncio
async def test_redis_db_cache(fetch_func, missing, device_id, connections_cache):
    # fetch from mysql and cache in redis
    from_db = await fetch_func(connections=connections_cache, key_arg=device_id)

    # fetch from redis only
    from_redis = await fetch_func(connections=connections_cache, key_arg=device_id)

    assert from_db == from_redis

    if missing == "raise":
        with pytest.raises(LookupError):
            await fetch_func(connections=connections_cache, key_arg="bad-device-id")
    else:
        assert (
            await fetch_func(connections=connections_cache, key_arg="bad-device-id")
            == missing
        )


@pytest.fixture
def connections(rediscon, pool):
    return Connections(redis=rediscon, pool=pool)


@pytest.mark.asyncio
async def test_set_apppliance_state_all(connections, rediscon, device_id, state):
    state["power"] = Power.OFF
    with mock.patch(
        "skynet.utils.cache_util.fetch_appliance_state", side_effect=LookupError
    ):
        await cache_util.set_appliance_state_all(
            connections=connections, key_arg=device_id, value=state
        )
    assert (
        await cache_util.get_appliance_state(redis=rediscon, key_arg=device_id) == state
    )
    with pytest.raises(LookupError):
        await cache_util.get_on_state(redis=rediscon, key_arg=device_id)

    state["power"] = Power.ON
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state
    )
    assert (
        await cache_util.get_appliance_state(redis=rediscon, key_arg=device_id) == state
    )
    assert await cache_util.get_on_state(redis=rediscon, key_arg=device_id) == state


@pytest.mark.asyncio
async def test_set_apliance_state_all_fan(connections, rediscon, device_id, state):
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state
    )
    assert (
        await cache_util.get_fan_redis(rediscon, state["appliance_id"], state["mode"])
        == state["fan"]
    )


@pytest.mark.asyncio
async def test_set_appliance_state_all_last_deployment(
    connections, rediscon, device_id, state
):
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state
    )
    assert (
        await cache_util.get_last_deployment(redis=rediscon, key_arg=device_id)
        == state["created_on"]
    )


@pytest.fixture
def off_state(state):
    s = state.copy()
    s["power"] = Power.OFF
    return s


@pytest.mark.asyncio
async def test_set_appliance_state_all_should_store_last_off_state_timestamp(
    connections, rediscon, device_id, state, off_state
):
    assert (
        await cache_util.get_last_off_state_timestamp(redis=rediscon, key_arg=device_id)
        is None
    )
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state
    )
    assert (
        await cache_util.get_last_off_state_timestamp(redis=rediscon, key_arg=device_id)
        is None
    )
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=off_state
    )
    assert (
        await cache_util.get_last_off_state_timestamp(redis=rediscon, key_arg=device_id)
        == off_state["created_on"]
    )


@pytest.mark.asyncio
async def test_update_appliance_state(connections, rediscon, device_id, state):
    # with nothing in redis and the database, we should just return the state
    # itself
    with mock.patch(
        "skynet.utils.cache_util.fetch_appliance_state", side_effect=LookupError
    ):
        assert (
            await cache_util.update_appliance_state(connections, device_id, state)
            == state
        )

    # we first put the states in redis, and then update with an incomplete
    # state, the complete state should still be in redis
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state
    )
    state2 = state.copy()
    del state2["fan"]
    await cache_util.set_appliance_state_all(
        connections=connections, key_arg=device_id, value=state2
    )
    assert "fan" in await cache_util.get_appliance_state(
        redis=rediscon, key_arg=device_id
    )


@pytest.mark.asyncio
async def test_update_last_on_ventilation(connections, rediscon, device_id, state):
    ventilation = [None, "off", "on", "off"]
    result = [None, None, state["created_on"], state["created_on"]]
    for v, r in zip(ventilation, result):
        state["ventilation"] = v
        await cache_util.set_appliance_state_all(connections, device_id, state)
        assert (
            await cache_util.get_last_on_ventilation(redis=rediscon, key_arg=device_id)
            == r
        )


@pytest.fixture
def old_device_id():
    return "old"


@pytest.mark.asyncio
async def test_active_controller(rediscon, device_id, old_device_id):
    active_controllers = await cache_util.get_online_devices(rediscon)
    assert active_controllers == []

    with freeze_time(
        datetime.utcnow() - timedelta(seconds=cache_util.CONTROLLER_CACHE_SECOND)
    ):
        await cache_util.set_online_device(rediscon, old_device_id)
    await cache_util.set_online_device(rediscon, device_id)

    active_controllers = await cache_util.get_online_devices(rediscon)
    assert device_id in active_controllers
    assert old_device_id not in active_controllers


@pytest.fixture
def event_msg(sensors):
    sensors["created_on"] = sensors["created_on"].replace(microsecond=0)
    return [events.SensorEvent.topic.encode(), msg_util.encode(sensors)]


@pytest.mark.asyncio
async def test_event_queue(rediscon, event_msg, sensors):
    await cache_util.enqueue_event(rediscon, event_msg)
    topic, data = await cache_util.pick_event(rediscon)
    assert topic == events.SensorEvent.topic
    assert data == sensors


@pytest.fixture
async def active_controller_redis(rediscon, device_id):
    await cache_util.set_online_device(rediscon, device_id)
    yield rediscon


@pytest.mark.parametrize("topic", [events.ComfortPredictionEvent.topic])
@pytest.mark.asyncio
async def test_create_events_for_online_devices(
    active_controller_redis, device_id, topic
):
    await cache_util.create_events_for_online_devices(topic)(active_controller_redis)
    topic_, data = await cache_util.pick_event(active_controller_redis)
    assert topic_ == topic
    assert data == dict(device_id=device_id)


@pytest.fixture
def control_mode(control_target):
    return event_parsing.extract_control_target(control_target)[0]


@pytest.fixture
@pytest.mark.asyncio
async def control_target_redis(rediscon, device_id, control_target):
    await cache_util.set_control_target(
        redis=rediscon, key_arg=device_id, value=control_target
    )
    yield rediscon


@pytest.fixture
def control_target_connections(control_target_redis, pool):
    return Connections(pool=pool, redis=control_target_redis)


@pytest.mark.asyncio
async def test_fetch_control_mode(control_target_connections, device_id, control_mode):
    assert (
        await cache_util.fetch_control_mode(control_target_connections, device_id)
        == control_mode
    )


@pytest.mark.asyncio
async def test_fetch_control_mode_no_target(connections, device_id):
    # when nothing is cached we should be able to fetch from db
    assert await cache_util.fetch_control_mode(connections, device_id) is not None


@pytest.mark.asyncio
async def test_set_control_target_all(
    control_target_connections, control_target_redis, device_id, control_mode
):
    # by default with no control target the mode should be None
    assert (
        await cache_util.get_last_control_mode(
            redis=control_target_redis, key_arg=device_id
        )
        is None
    )
    # adding a new control target should set the last mode
    await cache_util.set_control_target_all(
        connections=control_target_connections, key_arg=device_id, value={}
    )
    assert (
        await cache_util.get_last_control_mode(
            redis=control_target_redis, key_arg=device_id
        )
        == control_mode
    )


@pytest.fixture
def mode_pref_key():
    return ModePrefKey("comfort", None, None)


@pytest.fixture
def mode_pref(mode_pref_key):
    return ModePref(mode_pref_key, ["cool"])


@pytest.mark.asyncio
async def test_last_mode_preference(rediscon, device_id, mode_pref_key, mode_pref):
    # we return None on a cache miss
    assert (
        await cache_util.get_last_mode_preference(rediscon, device_id, mode_pref_key)
        is None
    )

    # we should be able to get the mode preference back once set
    await cache_util.set_last_mode_preference(rediscon, device_id, mode_pref)
    assert (
        await cache_util.get_last_mode_preference(rediscon, device_id, mode_pref_key)
        == mode_pref
    )


@pytest.mark.asyncio
async def test_set_mode_preference_all(
    connections, rediscon, device_id, mode_pref_key, mode_pref
):
    await cache_util.set_mode_preference_all(connections, device_id, mode_pref)
    # last modepref should still be None and the new modepref should be set
    assert (
        await cache_util.get_last_mode_preference(rediscon, device_id, mode_pref_key)
        is None
    )
    assert (
        await cache_util.fetch_mode_preference(connections, device_id, mode_pref_key)
        == mode_pref
    )

    new_mode_pref = ModePref(mode_pref_key, ["cool", "heat"])
    await cache_util.set_mode_preference_all(connections, device_id, new_mode_pref)

    # last modepref should now be set
    assert (
        await cache_util.get_last_mode_preference(rediscon, device_id, mode_pref_key)
        == mode_pref
    )
    assert (
        await cache_util.fetch_mode_preference(connections, device_id, mode_pref_key)
        == new_mode_pref
    )


@pytest.mark.asyncio
async def test_get_nearby_users_with_non_existant_key(rediscon, device_id):
    with pytest.raises(LookupError):
        await cache_util.get_nearby_users(rediscon, device_id)


test_data = [
    pytest.param(
        [NearbyUserAction(action=NearbyUser.DEVICE_CLEAR, user_id=None)],
        set(),
        id="get nearby user gives an empty set upon reset event",
    ),
    pytest.param(
        [NearbyUserAction(action=NearbyUser.USER_IN, user_id="user0")],
        {"user0"},
        id="get nearby user gives single user_id upon single add event",
    ),
    pytest.param(
        [
            NearbyUserAction(action=NearbyUser.USER_IN, user_id="user0"),
            NearbyUserAction(action=NearbyUser.USER_OUT, user_id="user0"),
        ],
        set(),
        id="get nearby user gives empty set upon removing the only user_id present in nearby users",
    ),
    pytest.param(
        [NearbyUserAction(action=NearbyUser.USER_OUT, user_id="user0")],
        set(),
        id="get nearby user gives empty set upon removing the user_id from empty nearby users",
    ),
    pytest.param(
        [
            NearbyUserAction(action=NearbyUser.DEVICE_CLEAR, user_id=None),
            NearbyUserAction(action=NearbyUser.USER_OUT, user_id="user0"),
        ],
        set(),
        id="get nearby user gives empty set upon reset event "
        "followed by removing the user_id from nearby users",
    ),
]


@pytest.mark.parametrize("events, expected", test_data)
@pytest.mark.asyncio
async def test_set_get_nearby_users(rediscon, device_id, events, expected):
    for event in events:
        await cache_util.set_nearby_users(rediscon, key_arg=device_id, value=event)
    actual = await cache_util.get_nearby_users(rediscon, device_id)
    assert actual == expected


@pytest.fixture
def nearby_user_connections(rediscon):
    return Connections(redis=rediscon)


@pytest.mark.parametrize("events, expected", test_data)
@pytest.mark.asyncio
async def test_fetch_nearby_users(
    rediscon, device_id, nearby_user_connections, events, expected
):
    for event in events:
        await cache_util.set_nearby_users(rediscon, key_arg=device_id, value=event)
    actual = await cache_util.fetch_nearby_users(nearby_user_connections, device_id)
    assert actual == expected


@pytest.fixture
def db_service_response(user_id):
    return (
        200,
        {
            "data": {
                "device": {
                    "device_nearby_user_status": [
                        {"user_id": user_id},
                        {"user_id": user_id},
                    ]
                }
            },
            "errors": None,
        },
    )


@pytest.fixture
def db_service(db_service_response):
    db_service_messenger = mock.Mock()
    db_service_messenger.ask = mock.CoroutineMock(return_value=db_service_response)
    return db_service_messenger


@pytest.fixture
def nearby_user_connections_with_db_service(rediscon, db_service):
    return Connections(redis=rediscon, db_service_msger=db_service)


@pytest.mark.asyncio
async def test_fetch_nearby_users_from_db_service(
    rediscon, device_id, user_id, nearby_user_connections_with_db_service
):
    user_ids = await cache_util.fetch_nearby_users(
        nearby_user_connections_with_db_service, device_id
    )
    user_id_using_get_nearby_users = await cache_util.get_nearby_users(
        rediscon, device_id
    )
    assert user_ids == {user_id}
    assert user_id_using_get_nearby_users == {user_id}


@pytest.mark.asyncio
async def test_get_latest_feedbacks_with_no_key(rediscon, device_id):
    with pytest.raises(LookupError):
        await cache_util.get_latest_feedbacks(rediscon, device_id)


@pytest.fixture
def multiple_feedbacks(feedback):
    feedback_first_user, feedback_second_user = feedback.copy(), feedback.copy()
    feedback_first_user["user_id"] = "user1"
    feedback_second_user["user_id"] = "user2"
    return [feedback_first_user, feedback_second_user]


@pytest.fixture
def feedbacks_dict(feedback, multiple_feedbacks):
    return {"multiple_feedback": multiple_feedbacks, "list_single_feedback": [feedback]}


test_feedbacks_data = [
    pytest.param(
        "list_single_feedback",
        id="Test add a single feedback returns the same feedback",
    ),
    pytest.param(
        "multiple_feedback",
        id="Test adding two feedbacks returns the two added feedbacks",
    ),
]


# https://stackoverflow.com/a/42400786/5576491
@pytest.mark.parametrize("type_feedback", test_feedbacks_data)
@pytest.mark.asyncio
async def test_add_feedbacks(rediscon, device_id, feedbacks_dict, type_feedback):
    for feedback in feedbacks_dict[type_feedback]:
        await cache_util.set_latest_feedback(rediscon, device_id, feedback)

    latest_feedbacks = await cache_util.get_latest_feedbacks(rediscon, device_id)

    assert sorted(latest_feedbacks, key=itemgetter("user_id")) == sorted(
        feedbacks_dict[type_feedback], key=itemgetter("user_id")
    )


@pytest.fixture
def latest_feedbacks_pool(feedback):
    pool = mock.Mock()
    pool.execute = mock.CoroutineMock(return_value=[feedback])
    return pool


@pytest.fixture
def latest_feedbacks_connections(rediscon, latest_feedbacks_pool):
    return Connections(redis=rediscon, pool=latest_feedbacks_pool)


@pytest.mark.asyncio
async def test_fetch_latest_feedbacks(
    rediscon, device_id, feedback, latest_feedbacks_connections
):
    await cache_util.set_latest_feedback(rediscon, device_id, feedback)
    latest_feedbacks = await cache_util.fetch_latest_feedbacks(
        latest_feedbacks_connections, device_id
    )

    assert latest_feedbacks == [feedback]


@pytest.mark.asyncio
async def test_fetch_latest_feedbacks_from_mysql(
    rediscon, device_id, feedback, latest_feedbacks_connections
):
    latest_feedbacks = await cache_util.fetch_latest_feedbacks(
        latest_feedbacks_connections, device_id
    )
    latest_feedbacks_from_redis = await cache_util.get_latest_feedbacks(
        rediscon, device_id
    )
    assert latest_feedbacks == [feedback]
    assert latest_feedbacks == latest_feedbacks_from_redis


@pytest.fixture
def same_user_feedbacks(feedback):
    first_feedback, second_feedback = feedback.copy(), feedback.copy()
    first_feedback["feedback"] = 1
    second_feedback["feedback"] = 2
    return [first_feedback, second_feedback]


@pytest.mark.asyncio
async def test_update_feedback(rediscon, device_id, same_user_feedbacks, user_id):
    for feedback in same_user_feedbacks:
        await cache_util.set_latest_feedback(rediscon, device_id, feedback)

    latest_feedbacks = await cache_util.get_latest_feedbacks(rediscon, device_id)
    feedback_wth_user_id = [
        feedback for feedback in latest_feedbacks if feedback["user_id"] == user_id
    ][0]
    assert feedback_wth_user_id == same_user_feedbacks[-1]


@pytest.fixture
def low_priority_topic():
    return events.ComfortPredictionEvent.topic


@pytest.fixture
def high_priority_topic():
    return events.FeedbackEvent.topic


@pytest.mark.asyncio
async def test_pick_event_priority(
    active_controller_redis, low_priority_topic, high_priority_topic
):
    await cache_util.create_events_for_online_devices(low_priority_topic)(
        active_controller_redis
    )

    await cache_util.create_events_for_online_devices(high_priority_topic)(
        active_controller_redis
    )

    topic, _ = await cache_util.pick_event(active_controller_redis)
    assert topic == high_priority_topic
