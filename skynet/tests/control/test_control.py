import asyncio
from contextlib import ExitStack
from datetime import datetime, timedelta

import pytest
from ambi_utils.zmq_micro_service import msg_util
from asynctest import mock
from voluptuous import Invalid

from ...control import control, controller
from ...utils import cache_util, events, thermo
from ...utils.enums import NearbyUser
from ...utils.types import Connections, NearbyUserAction


@pytest.fixture
def connections(
    device_mode_preference_db_pool, rediscon, cassandra_session, db_service_msger
):
    return Connections(
        pool=device_mode_preference_db_pool,
        redis=rediscon,
        session=cassandra_session,
        db_service_msger=db_service_msger,
    )


@pytest.fixture
async def my_control(rediscon):
    return control.Control(
        redis=rediscon, listener=events.get_event_listener("127.0.0.1", "12346")
    )


@pytest.fixture
async def control_worker(connections, prediction_clients):
    return control.ControlWorker(
        connections=connections, prediction_clients=prediction_clients, testing=True
    )


@pytest.fixture()
def sensors(sensors, device_id):
    sensors["humidex"] = thermo.humidex(sensors["temperature"], sensors["humidity"])
    sensors["device_id"] = device_id
    return sensors


@pytest.fixture()
def listener_events(sensors):
    sensors["created_on"] = sensors["created_on"].replace(microsecond=0)
    return (
        [[events.SensorEvent.topic.encode(), msg_util.encode(sensors)]],
        [(events.SensorEvent.topic, sensors)],
    )


@pytest.fixture
def listener_queue(my_control):
    return my_control.listener_queue


@pytest.mark.flaky(reruns=3, reruns_delay=1)
@mock.patch("skynet.control.control.CONTROL_CONCURRENCY", 1)
@pytest.mark.asyncio
async def test_del_control_data(
    my_control, listener_queue, control_worker, listener_events
):
    events, worker_events = listener_events

    async def new_sensor(events=events):
        try:
            return events.pop()
        except IndexError:
            await asyncio.sleep(100)

    with mock.patch.object(listener_queue.listener, "recv_multipart", new=new_sensor):
        with mock.patch.object(
            control_worker, "handle_listener", new=mock.CoroutineMock()
        ) as handle:
            my_control.start()
            control_worker.start()
            # give some time to process the listener events
            await asyncio.sleep(0.05)
            calls = [mock.call(worker_event) for worker_event in worker_events]
            assert handle.call_args_list == calls


@pytest.fixture
def controller_mock():
    return mock.Mock(spec=controller.Controller)


@pytest.fixture(params=["event_from_event_service", "event_not_from_event_service"])
def topic(request):
    return request.param


class MockEventA(events.Event):
    topic = "event_from_event_service"

    def _parse(self):
        if "wrong" in self.raw_data:
            raise Invalid("got wrong")

        self.data = self.raw_data

    async def _update_controller(self, controller) -> None:
        controller.updated = True


class MockEventB(events.Event):
    topic = "event_not_from_event_service"
    is_from_event_service = False

    def _parse(self):
        if "wrong" in self.raw_data:
            raise Invalid("got wrong")

        self.data = self.raw_data

    async def _update_controller(self, controller) -> None:
        controller.updated = True


@pytest.mark.asyncio
async def test_handle_listener(
    control_worker, controller_mock, device_id, rediscon, topic
):

    # nothing should be updated when it does not pass the schema
    msg = [
        topic,
        {
            "created_on": datetime.utcnow(),
            "device_id": device_id,
            "data": "new_data",
            "wrong": "should not have that",
        },
    ]

    with mock.patch.object(
        control_worker, "get_controller", return_value=controller_mock
    ):
        controller_mock.updated = False
        await control_worker.handle_listener(msg)
        assert not controller_mock.updated

    # make sure that the event leads a controller update, that the controller
    # is set to active and that the controller state is stored at the end of
    # the update
    msg = [
        topic,
        {"created_on": datetime.utcnow(), "device_id": device_id, "data": "new_data"},
    ]

    with mock.patch.object(
        control_worker, "get_controller", return_value=controller_mock
    ):
        await control_worker.handle_listener(msg)
        if topic == "event_from_event_service":
            assert device_id in await cache_util.get_online_devices(rediscon)
        else:
            assert device_id not in await cache_util.get_online_devices(rediscon)
        assert controller_mock.updated
        assert controller_mock.store_state.called_once

    # if the create_on is outdated we should not process the event
    msg = [
        topic,
        {
            "created_on": datetime.utcnow()
            - events.EVENT_EXPIRED
            - timedelta(seconds=1),
            "device_id": device_id,
            "data": "new_data",
        },
    ]

    with mock.patch.object(
        control_worker, "get_controller", return_value=controller_mock
    ):

        controller_mock.updated = False
        await control_worker.handle_listener(msg)
        assert not controller_mock.updated


@pytest.fixture
async def appliance_id_redis(rediscon, device_id, appliance_id):
    await cache_util.set_appliance_state(
        redis=rediscon, key_arg=device_id, value=dict(appliance_id=appliance_id)
    )
    return rediscon


@pytest.fixture
def ir_feature_event(device_id, ir_feature):
    data = dict(device_id=device_id, irfeature=ir_feature)
    return events.IREvent.topic, data


@pytest.mark.asyncio
async def test_cache_listener_ir_feature(
    appliance_id_redis, control_worker, appliance_id, ir_feature_event, ir_feature
):
    # make sure that we have no ir_feature stored in redis yet
    with pytest.raises(LookupError):
        await cache_util.get_ir_feature(redis=appliance_id_redis, key_arg=appliance_id)

    # should be able to get it from the database still
    await control_worker.handle_listener(ir_feature_event)
    assert (
        await cache_util.get_ir_feature(redis=appliance_id_redis, key_arg=appliance_id)
        == ir_feature
    )


@pytest.mark.asyncio
async def test_cache_listener_missing_appliance_id(control_worker, ir_feature_event):
    # in the very unlikely event of missing the appliance_id, better raise
    with mock.patch("skynet.utils.database.queries.get_appliance", return_value=None):
        with pytest.raises(LookupError):
            await control_worker.handle_listener(ir_feature_event)


def test_check_listener(listener_queue):

    listener_queue.setup_listener()
    old_listener = listener_queue.listener

    # if updates, should keep the socket
    with mock.patch.object(listener_queue.monitor, "check", return_value=1):
        listener_queue.check_listener()

        assert old_listener == listener_queue.listener

    # no update, should reconnect
    with mock.patch.object(listener_queue.monitor, "check", return_value=0):
        listener_queue.check_listener()

        assert old_listener != listener_queue.listener


@pytest.mark.asyncio
async def test_get_controller(control_worker, device_id, controller_mock):

    # make sure that we handle exceptions
    with mock.patch(
        "skynet.control.controller.Controller.from_redis", side_effect=LookupError
    ):
        assert await control_worker.get_controller(device_id) is None

    # otherwise we return the controller
    with mock.patch.object(
        control_worker, "get_controller", return_value=controller_mock
    ):
        assert await control_worker.get_controller(device_id) == controller_mock


@pytest.fixture
def periodical_update(my_control):
    return my_control.periodical_update


@pytest.mark.skipif(not control.COMPUTE_METRICS, reason="we disable metrics for now")
@pytest.mark.asyncio
async def test_start_control(periodical_update):
    interval = 0.01
    call_count = 2
    patches = ["COMFORT_PREDICTION_SECOND"]

    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(  # pylint:disable=no-member
                mock.patch("skynet.control.control." + patch, interval)
            )

        patches = ["predict_comfort"]

        with ExitStack() as stack:
            mocks = [
                stack.enter_context(mock.patch.object(periodical_update, patch))
                for patch in patches
            ]

            periodical_update.start()
            await asyncio.sleep(interval * call_count)
            for stack_mock in mocks:
                assert stack_mock.call_count >= 2


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_create_controller_from_redis(
    rediscon, sensors, control_worker, device_id, ir_feature, user_id
):

    # we don't have sensor data in redis, this can't be fetched from anywhere
    # else
    assert await control_worker.get_controller(device_id=device_id) is None

    # insert some sensordata
    await cache_util.set_sensors(rediscon, device_id, sensors)
    await cache_util.set_nearby_users(
        rediscon,
        device_id,
        NearbyUserAction(action=NearbyUser.USER_IN, user_id=user_id),
    )

    # make sure that the controller does not get created when ir features
    # cannot be fetched
    with mock.patch(
        "skynet.utils.cache_util.fetch_ir_feature", side_effect=LookupError
    ):
        assert await control_worker.get_controller(device_id=device_id) is None

    # should now be able to create the controller
    with mock.patch(
        "skynet.utils.cache_util.fetch_ir_feature", return_value=ir_feature
    ):
        my_controller = await control_worker.get_controller(device_id=device_id)
        assert isinstance(my_controller, controller.Controller)
