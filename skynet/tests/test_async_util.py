import asyncio
from asyncio import sleep
from datetime import timedelta

import pytest
from ambi_utils.zmq_micro_service.actor import Actor
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from asynctest import mock

from skynet.utils import async_util


def test_run_sync():
    async def mow(*args, **kwargs):
        await sleep(0)
        return args, kwargs

    args, kwargs = async_util.run_sync(mow, "bob", name="uncle")
    assert args == ("bob",)
    assert kwargs == {"name": "uncle"}


@pytest.mark.asyncio
async def test_add_callback():
    async def mow(mutable):
        mutable.append(1)

    mutable = []
    async_util.add_callback(mow, mutable)
    await sleep(0)
    assert mutable


@pytest.fixture
def actor():
    async def timeout(_request):
        await asyncio.sleep(10)

    a = mock.Mock(spec=Actor)
    a.ask = timeout
    return a


@pytest.fixture
def req():
    r = mock.Mock(spec=AircomRequest)
    r.method = "test"
    return r


@pytest.mark.asyncio
async def test_request_with_timeout(actor, req):
    with pytest.raises(asyncio.TimeoutError):
        await async_util.request_with_timeout(timedelta(milliseconds=1), actor, req)


@pytest.mark.asyncio
async def test_multi():
    async def identity(x):
        return x

    assert await async_util.multi([identity(1), identity(2)]) == [1, 2]
    assert await async_util.multi({"a": identity(1), "b": identity(2)}) == dict(
        a=1, b=2
    )
    assert await async_util.multi(identity(x) for x in [1, 2]) == [1, 2]
    with pytest.raises(NotImplementedError):
        await async_util.multi(identity(0))
