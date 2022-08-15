import pytest

from ...control import util
from ...utils import cache_util
from ...utils.enums import Power
from ...utils.types import BasicDeployment, Connections, ModePref, ModePrefKey


def test_likely_better():
    assert util.likely_better(1, 10, -5) is True
    assert util.likely_better("1", "10", -5) is True
    assert util.likely_better(1, "10", -5) is True
    assert util.likely_better(1, 10, 5) is False
    assert util.likely_better(10, 1, -5) is False
    assert util.likely_better("abc", "10", -5) is True


def test_has_same_temperature():
    assert util.has_same_temperature(1, 1) is True
    assert util.has_same_temperature("abc", "abc") is True
    assert util.has_same_temperature("1", 1) is True
    assert util.has_same_temperature("1", 1) is True
    assert util.has_same_temperature("+1", 1) is True
    assert util.has_same_temperature("+1", "-1") is False
    assert util.has_same_temperature(None, 1) is False
    assert util.has_same_temperature(None, None) is True


def test_away_mode_conditions():
    runs = [
        ("temperature", "upper", 10, 20, "bad"),
        ("temperature", "upper", 20, 10, "good"),
        ("temperature", "lower", 10, 20, "good"),
        ("temperature", "lower", 20, 10, "bad"),
        ("temperature", "lower", 19, 20, "close"),
        ("temperature", "upper", 20, 19, "close"),
        ("humidity", "upper", 70, 65, "close"),
        ("temperature", "upper", 25, 24, "close"),
    ]
    for quantity, threshold_type, threshold, current, result in runs:
        assert (
            util.away_mode_conditions(quantity, threshold_type, threshold, current)
            == result
        )

    with pytest.raises(ValueError):
        util.away_mode_conditions("temperature", "blah", 0, 0)


def test_away_mode_action():
    # is_on, condition, timed_out, new
    # (is_on, condition, timed_out, new)
    runs = [
        (False, "good", False, None),
        (False, "good", True, None),
        (False, "close", False, None),
        (False, "close", True, None),
        (False, "bad", False, "update"),
        (False, "bad", True, "update"),
        (True, "good", False, "off"),
        (True, "good", True, "off"),
        (True, "close", False, None),
        (True, "close", True, "off"),
        (True, "bad", False, None),
        (True, "bad", True, "off"),
    ]
    for *args, action in runs:
        # pylint:disable=no-value-for-parameter
        assert util.away_mode_action(*args, new=False) == action
        # pylint:enable=no-value-for-parameter

    # new control target and conditions close
    assert util.away_mode_action(True, "close", False, new=True) == "off"

    # new control target and AC already on, update to best setting for away
    # mode
    assert util.away_mode_action(True, "bad", False, new=True) == "update"

    # when switching to away mode or changing away mode threshold or type
    # it should update when condition is bad
    assert util.away_mode_action(True, "bad", True, new=True) == "update"


def test_needs_mode_pref_update():
    runs = [
        {
            "current_mode_pref": ("comfort", "humidex", None, ["cool"]),
            "new_mode_pref": ("comfort", "humidex", None, ["cool"]),
            "return": False,
        },
        {
            "current_mode_pref": ("comfort", "humidex", None, ["cool"]),
            "new_mode_pref": ("comfort", "humidex", None, ["cool", "dry"]),
            "return": True,
        },
        {
            "current_mode_pref": ("away", "temperature", "upper", ["cool"]),
            "new_mode_pref": ("away", "temperature", "upper", ["cool", "dry"]),
            "return": True,
        },
        {
            "current_mode_pref": ("away", "temperature", "upper", ["cool"]),
            "new_mode_pref": ("away", "temperature", "lower", ["cool"]),
            "return": False,
        },
        {
            "current_mode_pref": ("away", "temperature", "lower", ["cool"]),
            "new_mode_pref": ("away", "temperature", "lower", ["cool"]),
            "return": False,
        },
    ]
    for run in runs:

        *key, val = run["current_mode_pref"]
        current = ModePref(ModePrefKey(*key), val)

        *key, val = run["new_mode_pref"]
        new = ModePref(ModePrefKey(*key), val)
        assert util.needs_mode_pref_update(current, new) == run["return"]


def test_groupby():
    records = []
    assert {} == util.group_by("a", records)
    records = [{"a": 1, "b": 1}]
    assert {1: {"b": 1}} == util.group_by("a", records)
    records = [{"a": 1, "b": 1}, {"a": 2, "b": 2}]
    assert {1: {"b": 1}, 2: {"b": 2}} == util.group_by("a", records)
    # we do not keep a list of matching records just the last one
    records = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    assert {1: {"b": 2}} == util.group_by("a", records)


@pytest.mark.parametrize(
    "state, signal, target_delta, result",
    [
        pytest.param(
            {"power": Power.ON, "mode": "cool", "temperature": 12},
            {"power": Power.ON, "mode": "heat", "temperature": None},
            0,
            True,
            id="if the mode is different we should deploy",
        ),
        pytest.param(
            {"power": Power.ON, "mode": "cool", "temperature": "16"},
            {"power": Power.OFF, "mode": "cool", "temperature": "16"},
            0,
            True,
            id="if the power is different we should deploy",
        ),
        pytest.param(
            {"power": Power.ON, "mode": "cool", "temperature": 16},
            {"power": Power.ON, "mode": "cool", "temperature": "16"},
            0,
            False,
            id="if the setting is the same we should not deploy",
        ),
        pytest.param(
            {"power": Power.ON, "mode": "heat", "temperature": 10},
            {"power": Power.ON, "mode": "heat", "temperature": "12"},
            -5,
            False,
            id="if the temperature and delta not aggreing we should not deploy",
        ),
        pytest.param(
            {"power": Power.ON, "mode": "heat", "temperature": 10},
            {"power": Power.ON, "mode": "heat", "temperature": "12"},
            5,
            True,
            id="if the temperature and delta are aggreing we should deploy",
        ),
    ],
)
def test_state_update_requirement(state, signal, target_delta, result):
    assert util.state_update_required(state, signal, target_delta) == result


def test_chunks():
    l_test = [1, 2, 3, 4, 5, 6, 7]
    assert list(util.chunks(l_test, 3)) == [[1, 2, 3], [4, 5, 6], [7]]


@pytest.mark.asyncio
async def test_get_fan_setting(rediscon, appliance_id, ir_feature):
    # no setting in redis, should get lowest from ir_feature
    assert await cache_util.get_fan_redis(rediscon, appliance_id, "cool") is None
    ir_feature["cool"]["fan"]["value"] = ["high", "low"]
    fan_setting = await util.get_fan_setting(rediscon, appliance_id, "cool", ir_feature)
    assert fan_setting == "low"

    # with a setting in redis, we pick that one
    await cache_util.set_fan_redis(rediscon, appliance_id, "auto", "auto")
    fan_setting = await util.get_fan_setting(rediscon, appliance_id, "auto", ir_feature)
    assert fan_setting == "auto"


@pytest.mark.asyncio
async def test_regression_get_fan_settings_missing_fan_ir_features(
    rediscon, appliance_id, ir_feature
):
    # when no fan setting in redis and ir feature, we should return None
    assert await cache_util.get_fan_redis(rediscon, appliance_id, "cool") is None
    del ir_feature["cool"]["fan"]
    assert (
        await util.get_fan_setting(rediscon, appliance_id, "cool", ir_feature) is None
    )


@pytest.mark.asyncio
async def test_regression_adjust_deployment_settings_power_off(
    rediscon, pool, device_id, appliance_id, state, ir_feature
):
    # no mode specified when turning off should not raise a KeyError
    settings = BasicDeployment(power=Power.OFF)
    connections = Connections(redis=rediscon, pool=pool)
    assert await util.adjust_deployment_settings(
        connections, settings, device_id, appliance_id, state, ir_feature
    )


@pytest.fixture
def new_device_id():
    return "new_device_id"


@pytest.mark.asyncio
async def test_regression_adjust_deployment_settings_no_last_on_state(
    rediscon, pool, new_device_id, appliance_id, state, ir_feature
):
    # should not raise even if no last on state is available
    settings = BasicDeployment(power=Power.ON, mode="heat", temperature=24)
    connections = Connections(redis=rediscon, pool=pool)
    assert await util.adjust_deployment_settings(
        connections, settings, new_device_id, appliance_id, state, ir_feature
    )


@pytest.mark.asyncio
async def test_handle_invalid_sensors(rediscon, caplog):
    d = "device_id"
    exc = "exception"
    msg = "msg"

    await util.handle_invalid_sensors(rediscon, d, exc, msg)
    assert "INFO" in caplog.text
    for _ in range(util.N_INVALID_LOG):
        await util.handle_invalid_sensors(rediscon, d, exc, msg)
    assert "ERROR" in caplog.text


@pytest.fixture
def ventilation_ir_feature(ir_feature):
    ir_feature = ir_feature.copy()
    for mode in ir_feature:
        ir_feature[mode]["ventilation"] = dict(
            ftype="select_option", value=["on", "off"]
        )
    return ir_feature


@pytest.mark.parametrize(
    "state, settings, ventilation",
    [
        pytest.param(
            dict(power="off", ventilation=None),
            BasicDeployment(power="off", ventilation="off"),  # type: ignore
            "off",
            id="make sure ventilation is overrriden by current deployment",
        ),
        pytest.param(
            dict(power="on", ventilation="on"),
            BasicDeployment(power="on", temperature=18),  # type: ignore
            "on",
            id="make sure that ventilation setting is propagated from the current state",
        ),
    ],
)
@pytest.mark.asyncio
async def test_adjust_deployment_settings_ventilation(
    device_id,
    appliance_id,
    ventilation_ir_feature,
    state,
    settings,
    ventilation,
    connections,
):
    setting = await util.adjust_deployment_settings(
        connections, settings, device_id, appliance_id, state, ventilation_ir_feature
    )
    assert setting.ventilation == ventilation
