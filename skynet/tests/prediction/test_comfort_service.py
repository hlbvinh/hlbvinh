import asyncio
from datetime import datetime

import numpy as np
import pytest
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from asynctest import mock

from ...user import comfort_model, comfort_service
from ...utils.types import Connections

np.random.seed(0)

IP = "127.0.0.1"


async def async_single_reply(
    connections, device_id, user_id, timestamp  # pylint: disable=unused-argument
):
    return {"created_on": timestamp, "comfort": 0.0}


@pytest.fixture
def connections(pool, cassandra_session, rediscon):
    return Connections(pool=pool, session=cassandra_session, redis=rediscon)


@pytest.fixture
async def client_service(model_store, connections, port, trained_comfort_model):
    model_store.save(
        comfort_model.ComfortModel.get_storage_key(), trained_comfort_model
    )
    service = comfort_service.ComfortService(
        ip=IP, port=port, connections=connections, storage=model_store
    )
    client = DealerActor(ip=IP, port=port, log=mock.MagicMock())

    return client, service


@pytest.fixture
def client(client_service):
    return client_service[0]


@pytest.mark.asyncio
async def test_response(get_response, client, device_intervals, device_id, user_id):
    """Test comfort service over ZMQ."""
    msg = {
        "method": "ComfortModelActor",
        "params": {"device_id": device_id},
        "session_id": None,
    }

    # test bad input params
    for params in [
        {},
        {"device_id": 42},
        {"unknown_key": "unknown_value"},
        {"device_id": device_id, "user_id": None},
    ]:
        msg["params"] = params
        resp = await get_response(client, msg)
        assert resp == []

    # the current comfort query should return a best guess even if there
    # is no data
    msg["params"] = {"device_id": device_id, "user_id": user_id}
    resp = await get_response(client, msg)
    assert len(resp) == 1
    assert isinstance(resp[0]["comfort"], float)
    assert isinstance(resp[0]["created_on"], datetime)

    # for a device having actual data, queries should return data
    for device in device_intervals:
        msg["params"] = {"device_id": device["device_id"], "user_id": user_id}
        resp = await get_response(client, msg)
        assert len(resp) == 1
        assert isinstance(resp[0]["comfort"], float)
        assert isinstance(resp[0]["created_on"], datetime)


@pytest.fixture
async def actor():
    storage = mock.MagicMock()
    connections = mock.MagicMock()
    actor = comfort_service.ComfortModelActor(connections, storage)
    # sleep to make sure the model is loaded
    await asyncio.sleep(0)
    return actor


@pytest.mark.asyncio
async def test_comfort_service(actor, user_id):
    params = [
        [
            {"user_id": user_id, "device_id": "a"},
            [lambda x: len(x["data"]) == 1, lambda x: x["status"] == 200],
        ],
        [
            {
                # range query not supported anymore, should not raise any
                # error, start and end params are simply discarded
                "start": datetime(2015, 1, 1),
                "end": datetime(2015, 1, 2),
                "device_id": "a",
                "user_id": user_id,
            },
            [lambda x: len(x["data"]) > 0, lambda x: x["status"] == 200],
        ],
    ]

    msgs = []
    for p, _ in params:
        msg = {}
        msg["params"] = p.copy()
        msg.update({"method": "ComfortModelActor", "session_id": "id"})
        msgs.append(msg)

    actor.models["comfort_model"].get_adjusted_comfort_prediction = async_single_reply

    for in_msg, (_, checks) in zip(msgs, params):
        resp = await actor.process(AircomRequest.from_dict(in_msg))
        for fun in checks:
            assert fun(resp), in_msg

    actor.models["comfort_model"].get_adjusted_comfort_prediction = mock.CoroutineMock(
        side_effect=Exception
    )
    for in_msg in msgs:
        resp = await actor.process(AircomRequest.from_dict(in_msg))
        assert not resp["data"]
        assert resp["status"] == 400
