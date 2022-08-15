from datetime import timedelta

import pytest

from skynet.utils.status import DummyService, status_request

from ..utils.log_util import get_logger

log = get_logger(__name__)


@pytest.mark.asyncio
async def test_status_actor(port):
    ip = "127.0.0.1"
    service = DummyService(ip, port)  # noqa, pylint: disable=unused-variable

    # test with timeout
    resp = await status_request(ip, port, "test", timedelta(seconds=1e-5))
    assert resp["status"] == 500
    assert resp["service"] == "test"

    # test success
    resp = await status_request(ip, port, "test", timedelta(seconds=1))
    assert resp["status"] == 200
    assert resp["service"] == "test"
