from datetime import datetime, timedelta

import numpy as np
import pytest
from asynctest import mock

from ...control.managed_manual import (
    DEFAULT_SET_TEMPERATURE,
    LOW_AC_USAGE,
    SUPPORTED_MODES,
)
from ...utils.enums import Power


@pytest.mark.asyncio
async def test_fetch_states(managed_manual_):
    states = await managed_manual_._fetch_states()
    for state in states:
        assert all(k in state for k in "mode temperature_set power created_on".split())


now = datetime.utcnow()


@pytest.mark.parametrize(
    "state, result",
    [
        pytest.param(
            dict(temperature_set="25.5", mode="cool", created_on=now, power=Power.ON),
            dict(temperature_set=25.5, mode="cool", created_on=now, power=Power.ON),
            id="make sure temperature is converted to float if mode is supported",
        ),
        pytest.param(
            dict(temperature_set="fan", mode="fan", created_on=now, power=Power.ON),
            dict(temperature_set="fan", mode="fan", created_on=now, power=Power.ON),
            id="make sure state is not modified if mode is not supported",
        ),
        pytest.param(
            dict(
                temperature_set="fan",
                mode="fan",
                created_on=now,
                power=Power.ON.upper(),
            ),
            dict(temperature_set="fan", mode="fan", created_on=now, power=Power.ON),
            id="make sure all strings in state are lower case",
        ),
    ],
)
def test_parse_state(managed_manual_, state, result):
    assert managed_manual_._parse_state(state) == result


@pytest.mark.parametrize(
    "states, mode, target, duration",
    [
        pytest.param(
            [
                dict(power=Power.ON, mode="cool", temperature_set=17, created_on=now),
                dict(
                    power=Power.ON,
                    mode="heat",
                    temperature_set=28,
                    created_on=now + timedelta(minutes=5),
                ),
                dict(
                    power=Power.ON,
                    mode="heat",
                    temperature_set=29,
                    created_on=now + timedelta(minutes=9),
                ),
            ],
            "cool",
            17,
            timedelta(minutes=5),
            id="make sure duration is time to the NEXT state",
        ),
        pytest.param(
            [
                dict(power=Power.ON, mode="cool", temperature_set=20, created_on=now),
                dict(
                    power=Power.ON,
                    mode="cool",
                    temperature_set=30,
                    created_on=now + timedelta(minutes=5),
                ),
                dict(
                    power=Power.ON,
                    mode="heat",
                    temperature_set=29,
                    created_on=now + timedelta(minutes=15),
                ),
            ],
            "cool",
            (20 + 2 * 30) / 3,
            timedelta(minutes=15),
            id="make sure we compute a weighted averaged set point",
        ),
        pytest.param(
            [
                dict(power=Power.ON, mode="cool", temperature_set=17, created_on=now),
                dict(
                    power=Power.ON,
                    mode="fan",
                    temperature_set=18,
                    created_on=now + timedelta(minutes=5),
                ),
                dict(
                    power=Power.ON,
                    mode="auto",
                    temperature_set=19,
                    created_on=now + timedelta(minutes=15),
                ),
            ],
            "cool",
            17,
            timedelta(minutes=5),
            id="make sure we do not use auto or fan mode even if it is the most used",
        ),
    ],
)
def test_precomputation(managed_manual_, states, mode, target, duration):
    c = managed_manual_._precomputation(states)
    assert c["mode"] == mode
    assert np.isclose(c["target"], target)
    assert c["duration"] == duration


@pytest.fixture
def states(state):
    states = []
    settings = [
        dict(power=Power.ON, mode="cool", temperature_set=17),
        dict(power=Power.ON, mode="cool", temperature_set=22),
        dict(power=Power.ON, mode="cool", temperature_set=23),
    ]
    for i, setting in enumerate(settings):
        s = state.copy()
        s.update(setting)
        s["created_on"] += LOW_AC_USAGE * i
        states.append(s)
    return states


@pytest.fixture
def managed_manual_states(managed_manual_, states):
    managed_manual_._fetch_states = mock.CoroutineMock(return_value=states)
    return managed_manual_


@pytest.fixture
def target():
    return 25.3


@pytest.mark.asyncio
async def test_get_settings(managed_manual_states, target, ir_feature):
    setting = await managed_manual_states.get_deployment(target, ir_feature)
    assert setting.power == Power.ON
    assert setting.mode in ir_feature
    assert setting.temperature in ir_feature[setting.mode]["temperature"]["value"]
    assert abs(float(setting.temperature) - target) < 1


@pytest.fixture
def adr_target(target):
    return target + 2


@pytest.mark.asyncio
async def test_get_settings_during_adr(
    managed_manual_states, target, adr_target, ir_feature
):
    setting = await managed_manual_states.get_deployment(target, ir_feature)
    adr_setting = await managed_manual_states.get_deployment(adr_target, ir_feature)
    assert float(setting.temperature) < float(adr_setting.temperature)


@pytest.fixture(params=SUPPORTED_MODES)
def mode(request):
    return request.param


@pytest.fixture
def managed_manual_with_low_ac_usage(managed_manual_, target, mode):
    precomputation = {"mode": mode, "target": target, "duration": timedelta(hours=1)}
    managed_manual_._fetch_precomputation = mock.CoroutineMock(
        return_value=precomputation
    )
    return managed_manual_


@pytest.fixture
def ir_feature_with_heat(ir_feature):
    ir_feature["heat"] = ir_feature["cool"]
    return ir_feature


@pytest.mark.asyncio
async def test_get_deployment_when_low_ac_usage(
    managed_manual_with_low_ac_usage, target, ir_feature_with_heat, mode
):
    deployment = await managed_manual_with_low_ac_usage.get_deployment(
        target, ir_feature_with_heat
    )
    assert deployment.temperature == DEFAULT_SET_TEMPERATURE[mode]
    assert deployment.mode == mode


@pytest.fixture(params=[True, False])
def insufficient_states(request, state):
    return [state] if request.param else []


@pytest.mark.asyncio
async def test_get_deployment_when_insufficient_states(
    managed_manual_, target, ir_feature, insufficient_states
):
    managed_manual_._fetch_states = mock.CoroutineMock(return_value=insufficient_states)
    assert await managed_manual_.get_deployment(target, ir_feature) is None


@pytest.fixture
def off_state(state):
    s = state.copy()
    s["power"] = Power.OFF
    return s


@pytest.mark.asyncio
async def test_no_deployment_ac_off(managed_manual_, target, ir_feature, off_state):
    managed_manual_.state = off_state
    assert await managed_manual_.get_deployment(target, ir_feature) is None
