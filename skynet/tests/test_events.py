from datetime import datetime

import pytest
from asynctest import mock

from ..control import controller
from ..utils import cache_util, events
from ..utils.enums import NearbyUser
from ..utils.types import Connections


@pytest.fixture
def sensor():
    return dict(
        created_on=datetime.utcnow(), HM=2.0, TP=1.0, LU=dict(infrared_spectrum=3.0)
    )


@pytest.fixture
def mode_feedback():
    return dict(mode_feedback="cool", created_on=datetime.utcnow())


@pytest.fixture
def nearby_user_action_message(device_id, user_id):
    return dict(
        device_id=device_id,
        user_id=user_id,
        action=NearbyUser.USER_IN.value,
        created_on=datetime.utcnow(),
    )


@pytest.fixture
def automated_demand_response_message(device_id):
    return dict(
        device_id=device_id,
        action="start",
        signal_level=1,
        create_time=datetime.utcnow(),
        group_name="",
    )


@pytest.fixture(params=events.EVENT_REGISTRY.keys())
def listener_message(
    device_id,
    request,
    ir_feature,
    sensor,
    control_target,
    feedback,
    state,
    mode_feedback,
    nearby_user_action_message,
    automated_demand_response_message,
):
    data = dict(device_id=device_id, data=mock.MagicMock())
    if request.param == events.IREvent.topic:
        data = dict(device_id=device_id, irfeature=ir_feature)

    if request.param == events.ModePreferenceEvent.topic:
        data = dict(
            device_id=device_id,
            quantity="climate",
            heat=1,
            cool=1,
            fan=0,
            auto=0,
            dry=0,
        )

    if request.param == events.SensorEvent.topic:
        data = dict(device_id=device_id, **sensor)

    if request.param == events.ControlEvent.topic:
        data = control_target

    if request.param == events.FeedbackEvent.topic:
        data = dict(device_id=device_id, **feedback)

    if request.param == events.StateEvent.topic:
        data = dict(device_id=device_id, **state)

    if request.param == events.ModeFeedbackEvent.topic:
        data = dict(device_id=device_id, **mode_feedback)

    if request.param == events.NearbyUserEvent.topic:
        data = dict(**nearby_user_action_message)

    if request.param == events.AutomatedDemandResponseEvent.topic:
        data = automated_demand_response_message

    return request.param, data


@pytest.fixture
def connections(rediscon, pool, db_service_msger):
    return Connections(redis=rediscon, pool=pool, db_service_msger=db_service_msger)


@pytest.fixture
def controller_mock():
    return mock.Mock(spec=controller.Controller)


@pytest.mark.asyncio
async def test_create_event_from_listener_message(
    listener_message, connections, controller_mock
):
    event = await events.create_event_from_listener_message(
        listener_message, connections
    )
    assert event is not None
    await event.cache()
    if await event.need_processing():
        await event.update_controller(controller_mock)
        assert controller_mock.store_state.called


@pytest.fixture
def feedback_event_message(feedback, device_id):
    return events.FeedbackEvent.topic, dict(device_id=device_id, **feedback)


@pytest.fixture
def non_comfort_control_target(control_target):
    target = control_target.copy()
    target["quantity"] = "temperature"
    return target


@pytest.fixture
async def non_comfort_mode_connections(
    connections, non_comfort_control_target, device_id
):
    await cache_util.set_control_target_all(
        connections, device_id, non_comfort_control_target
    )
    yield connections


@pytest.mark.asyncio
async def test_predict_feedback_comfort_even_when_not_in_comfort_mode(
    non_comfort_mode_connections, controller_mock, feedback_event_message
):
    event = await events.create_event_from_listener_message(
        feedback_event_message, non_comfort_mode_connections
    )
    if await event.need_processing():
        await event.update_controller(controller_mock)

    assert controller_mock.predict_comfort_for_feedback.called
    assert not controller_mock.update_state.called


@pytest.fixture
def off_mode_target(control_target):
    control_target["quantity"] = "off"
    return control_target


@pytest.fixture
async def off_mode_connections(connections, off_mode_target, device_id):
    await cache_util.set_control_target_all(connections, device_id, off_mode_target)
    yield connections


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(events.ControlEvent.topic),
        pytest.param(events.ClimateModelMetricEvent.topic),
    ],
)
@pytest.mark.asyncio
async def test_create_controller_in_off_mode(
    off_mode_connections, event, off_mode_target
):
    off_mode_event_message = (event, off_mode_target)
    event = await events.create_event_from_listener_message(
        off_mode_event_message, off_mode_connections
    )
    assert await event.need_processing()
