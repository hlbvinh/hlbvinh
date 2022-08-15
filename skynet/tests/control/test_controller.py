import random
from datetime import datetime, timedelta

import numpy as np
import pytest
from asynctest import mock

from ...control import controller, util
from ...utils import cache_util, event_parsing
from ...utils.config import MAX_COMFORT, MIN_COMFORT
from ...utils.enums import Power
from ...utils.misc import UpdateCounter
from ...utils.types import AutomatedDemandResponse, BasicDeployment


@pytest.fixture
def mode_prefs():
    row = {
        "auto": 0,
        "created_on": datetime.utcnow(),
        "cool": 1,
        "quantity": "climate",
        "heat": 0,
        "dry": 1,
        "row_id": 0,
        "fan": 1,
    }
    return [event_parsing.parse_mode_prefs(row)]


@pytest.fixture
def update_counter():
    return UpdateCounter()


@pytest.fixture
def get_controller(
    connections,
    device_id,
    state,
    target,
    sensors,
    feedback,
    ir_feature,
    mode_prefs,
    prediction_clients,
    timezone,
    update_counter,
):
    async def wrap(
        connections=connections,
        device_id=device_id,
        appliance_id="test",
        ir_feature=ir_feature,
        control_target=target,
        mode_feedback={},
        sensors=sensors,
        state=state,
        mode_preferences=mode_prefs,
        timezone=timezone,
        latest_feedbacks=[feedback],
        nearby_users={feedback["user_id"]},
        automated_demand_response=None,
        prediction_clients=prediction_clients,
        monitor=mock.MagicMock,
        update_counter=update_counter,
    ):
        return await controller.Controller.create(
            connections=connections,
            device_id=device_id,
            appliance_id=appliance_id,
            ir_feature=ir_feature,
            control_target=control_target,
            mode_feedback=mode_feedback,
            sensors=sensors,
            state=state,
            mode_preferences=mode_preferences,
            timezone=timezone,
            latest_feedbacks=latest_feedbacks,
            nearby_users=nearby_users,
            automated_demand_response=automated_demand_response,
            prediction_clients=prediction_clients,
            monitor=monitor,
            update_counter=update_counter,
        )

    return wrap


@pytest.fixture
async def default_controller(get_controller):
    return await get_controller()


@pytest.fixture
async def to_turn_off_controller(get_controller, target):
    state = {
        "power": Power.ON,
        "mode": "heat",
        "temperature": 18,
        "created_on": datetime.utcnow(),
    }
    off_target = target.copy()
    off_target["quantity"] = "off"

    return await get_controller(state=state, control_target=off_target)


@pytest.mark.asyncio
async def test_controller_with_off_mode_but_ac_on_is_turned_off(to_turn_off_controller):

    c = to_turn_off_controller
    assert c.state["power"] == Power.ON
    assert c.control_target["quantity"] == "off"

    c.deploy.deploy_settings = mock.CoroutineMock(return_value=(200, mock.Mock()))
    await c.control_target_update()

    c.deploy.deploy_settings.assert_called_once_with(
        BasicDeployment(
            power=Power.OFF, mode=c.state["mode"], temperature=c.state["temperature"]
        )
    )


@pytest.fixture
def new_feedback(feedback):
    feedback["feedback"] = random.choice(np.arange(MIN_COMFORT, MAX_COMFORT + 1))
    return feedback


@pytest.fixture
async def comfort_controller(get_controller, new_feedback):
    target = {"quantity": "comfort", "created_on": datetime.utcnow(), "value": None}
    return await get_controller(control_target=target, latest_feedbacks=[new_feedback])


@pytest.mark.asyncio
async def test_feedback_state_update(comfort_controller):
    """
    If feedback is given while in comfort mode this should trigger a state
    update
    """

    c = comfort_controller
    c.update_state = mock.CoroutineMock()
    await c.feedback_update()
    assert c.update_state.called


@pytest.mark.asyncio
async def test_target_state_update(default_controller):
    """
    If a new control target is received, this should trigger a state update
    """

    c = default_controller
    c.update_state = mock.CoroutineMock()
    await c.control_target_update()
    assert c.update_state.called


@pytest.fixture
async def need_update_controller(get_controller):
    state = {
        "power": Power.ON,
        "temperature": "28",
        "created_on": datetime.utcnow() - 1.1 * util.SIGNAL_INTERVAL,
    }

    # last_state_update is set to datetime.utcnow() by default, need to
    # override it

    target = {"quantity": "comfort", "created_on": datetime.utcnow(), "value": 35}
    sensors = {
        "created_on": datetime.utcnow(),
        "temperature": 30,
        "humidity": 10,
        "luminosity": 10,
    }
    return await get_controller(state=state, control_target=target, sensors=sensors)


@pytest.mark.asyncio
async def test_sensor_state_update(need_update_controller):
    """
    In any smart mode, new sensor data should trigger an update unless
    datetime.utcnow() - self.last_state_update < UPDATE_INTERVAL
    or
    datetime.utcnow() - self.last_deployment < SIGNAL_INTERVAL
    """

    c = need_update_controller
    c.update_state = mock.CoroutineMock()
    await c.sensors_update()
    assert c.update_state.called


@pytest.mark.asyncio
async def test_predict_comfort_for_feedback(
    default_controller, device_id, user_id, rediscon
):
    c = default_controller
    with mock.patch(
        "skynet.utils.cache_util.fetch_feedback_humidex",
        return_value=20,
        new_callable=mock.CoroutineMock,
    ):
        await c.update_feedback_humidex()
    with mock.patch.object(c.comfort, "predict", return_value=1):
        await c.predict_comfort_for_feedback()
    feedback_prediction = await cache_util.get_comfort_prediction(
        redis=rediscon, key_arg=(device_id, user_id)
    )
    assert feedback_prediction


@pytest.fixture
async def no_ir_feature_controller(get_controller):
    target = {"quantity": "off", "created_on": datetime.utcnow()}
    sensors = {
        "created_on": datetime.utcnow(),
        "temperature": 30,
        "humidity": 10,
        "luminosity": 10,
    }
    return await get_controller(ir_feature=None, control_target=target, sensors=sensors)


@pytest.mark.asyncio
async def test_none_ir_feature(no_ir_feature_controller):
    # when ir_feature is set to none, no state update should be triggered
    c = no_ir_feature_controller
    c.update_state = mock.CoroutineMock()
    await c.control_target_update()
    await c.sensors_update()
    assert not c.update_state.called


@pytest.fixture
async def violating_away_mode_controller(get_controller):
    target = {
        "quantity": "away_temperature_upper",
        "value": 10,
        "created_on": datetime.utcnow(),
    }
    # need to make sure that both the state and sensor result in the update
    state = {
        "power": Power.OFF,
        "created_on": datetime.utcnow()
        - 1.1 * max(controller.AWAY_MODE_SIGNAL_DURATION, util.SIGNAL_INTERVAL),
        "mode": "cool",
        "temperature": 24,
        "appliance_id": "app_a",
    }
    sensors = {
        "created_on": datetime.utcnow(),
        "temperature": 30,
        "humidity": 10,
        "luminosity": 10,
    }
    return await get_controller(control_target=target, state=state, sensors=sensors)


@pytest.mark.asyncio
async def test_away_mode(violating_away_mode_controller):
    """
    when in away mode, violating the threshold should result in setting the
    AC on
    """

    c = violating_away_mode_controller

    # with the current implementation of the controller no other way than to
    # directly deal with the implementation
    c.deploy.deploy_settings = mock.CoroutineMock(return_value=(200, mock.Mock()))
    c.get_away_mode_settings = mock.CoroutineMock(return_value={"power": Power.ON})

    await c.sensors_update()

    c.deploy.deploy_settings.assert_called_once_with({"power": Power.ON})


@pytest.fixture
async def away_mode_controller(get_controller):

    target = {
        "quantity": "away_temperature_upper",
        "value": 10,
        "created_on": datetime.utcnow(),
    }
    return await get_controller(control_target=target)


@pytest.mark.asyncio
async def test_away_mode_mode_pref_change(away_mode_controller):
    c = away_mode_controller

    c.deploy.deploy_settings = mock.CoroutineMock(return_value=(200, mock.Mock()))
    c.get_away_mode_settings = mock.CoroutineMock(return_value={"power": Power.ON})

    # when in away mode, a mode_prefences update should trigger an update
    await c.mode_preferences_update()
    c.deploy.deploy_settings.assert_called_once_with({"power": Power.ON})


@pytest.fixture
def mode_feedback(device_id):
    return dict(device_id=device_id, created_on=datetime.utcnow(), mode_feedback="fan")


@pytest.fixture
async def comfort_cool_controller(get_controller, mode_feedback):
    state = {
        "power": Power.ON,
        "mode": "cool",
        "temperature": "18",
        "created_on": datetime.utcnow(),
    }
    target = {"quantity": "comfort", "created_on": datetime.utcnow(), "value": None}
    return await get_controller(
        state=state, control_target=target, mode_feedback=mode_feedback
    )


@pytest.mark.asyncio
async def test_mode_feedback_triggers_update(comfort_cool_controller, mode_feedback):
    c = comfort_cool_controller
    c.deploy.deploy_settings = mock.CoroutineMock(return_value=(200, mock.Mock()))
    await c.mode_feedback_update()
    c.deploy.deploy_settings.assert_called_once_with(
        BasicDeployment(power=Power.ON, mode=mode_feedback["mode_feedback"])
    )


@pytest.fixture
def expired_mode_feedback(mode_feedback):
    mode_feedback = mode_feedback.copy()
    mode_feedback["created_on"] = (
        datetime.utcnow() - controller.MODE_FEEDBACK_DURATION - timedelta(minutes=1)
    )
    return mode_feedback


def test_set_mode_feedback(default_controller, mode_feedback):
    c = default_controller
    assert c.mode_feedback == {}
    c.set_mode_feedback(mode_feedback)
    assert c.mode_feedback == mode_feedback


def test_expired_mode_feedback_not_set(default_controller, expired_mode_feedback):
    c = default_controller
    c.set_mode_feedback(expired_mode_feedback)
    assert c.mode_feedback == {}


@pytest.fixture
async def adr_off_controller(get_controller):
    target = {"quantity": "comfort", "value": None, "created_on": datetime.utcnow()}
    state = {
        "power": Power.ON,
        "created_on": datetime.utcnow(),
        "mode": "cool",
        "temperature": 24,
        "appliance_id": "app_a",
    }
    automated_demand_response = AutomatedDemandResponse("start", 1)

    return await get_controller(
        control_target=target,
        state=state,
        automated_demand_response=automated_demand_response,
    )


@pytest.mark.asyncio
async def test_adr_off(adr_off_controller):
    """when we receive an adr event of level 1, we should turn off the device"""
    c = adr_off_controller
    c.deploy.deploy_settings = mock.CoroutineMock()

    await c.sensors_update()

    c.deploy.deploy_settings.assert_called_once_with(c.deploy.off_deployment)


@pytest.fixture
async def managed_manual_controller(get_controller):
    target = {
        "quantity": "managed_manual",
        "value": None,
        "created_on": datetime.utcnow(),
    }
    state = {
        "power": Power.ON,
        "created_on": datetime.utcnow(),
        "mode": "cool",
        "temperature": 24,
        "appliance_id": "app_a",
    }
    automated_demand_response = AutomatedDemandResponse("start", 2)
    c = await get_controller(
        control_target=target,
        state=state,
        automated_demand_response=automated_demand_response,
    )
    c.managed_manual._fetch_precomputation = mock.CoroutineMock(
        return_value=dict(mode="cool", target=24.5, duration=timedelta(hours=7))
    )
    return c


@pytest.mark.asyncio
async def test_managed_manual(managed_manual_controller):
    c = managed_manual_controller
    c.deploy.deploy_settings = mock.CoroutineMock()

    # make sure that we have an adr penalty
    assert await c.managed_manual.get_target_value() != c.target.target_value

    await c.sensors_update()
    assert c.deploy.deploy_settings.called


@pytest.fixture(params=["start"])
async def adr_manual_controller(request, get_controller):
    return await get_controller(
        control_target={
            "quantity": "manual",
            "value": None,
            "created_on": datetime.utcnow(),
        },
        automated_demand_response=AutomatedDemandResponse(
            request.param, 2, datetime.utcnow() + timedelta(seconds=1)
        ),
    )


@pytest.mark.asyncio
async def test_manual_to_managed_manual(adr_manual_controller):
    assert adr_manual_controller.target.control_mode == "managed_manual"
