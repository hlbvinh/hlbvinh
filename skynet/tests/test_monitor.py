import time
from asyncio import sleep
from unittest import mock

import pytest

from ..utils import monitor

MONITOR_NAME = "my monitored function"


class Monitored:
    def __init__(self, monitor):
        self.monitor = monitor


@pytest.fixture
def decorated(my_monitor):
    class A(Monitored):
        @monitor.monitored(MONITOR_NAME)
        def fun(self):
            pass

    return A(monitor=my_monitor)


@pytest.fixture
def monitoring(my_monitor):
    class A(Monitored):
        def fun(self):
            self.monitor.tick(MONITOR_NAME)

    return A(monitor=my_monitor)


@pytest.fixture(params=["decorated", "monitoring"])
def monitored_obj(request, decorated, monitoring):
    if request.param == "decorated":
        return decorated

    if request.param == "monitoring":
        return monitoring

    raise ValueError


def test_monitor(my_monitor, monitored_obj):
    assert "DOWN" in my_monitor.maybe_message(MONITOR_NAME)
    monitored_obj.fun()
    assert "recovered" in my_monitor.maybe_message(MONITOR_NAME)
    monitored_obj.fun()
    assert my_monitor.maybe_message(MONITOR_NAME) is None
    for _ in range(10):
        monitored_obj.fun()
    assert my_monitor.maybe_message(MONITOR_NAME) is None
    time.sleep(0.1)
    assert "DOWN" in my_monitor.maybe_message(MONITOR_NAME)
    assert my_monitor.maybe_message(MONITOR_NAME) is None
    monitored_obj.fun()
    assert "recovered" in my_monitor.maybe_message(MONITOR_NAME)


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_monitor_log_error(my_monitor, monitored_obj, monitor_interval):
    with mock.patch.object(monitor, "log") as m:
        my_monitor.start()
        monitored_obj.fun()
        await sleep(monitor_interval * 2)
        monitored_obj.fun()
        await sleep(monitor_interval)
        monitored_obj.fun()
        await sleep(monitor_interval)
        assert len(m.mock_calls) == 2
        level, (msg,), _ = m.mock_calls[0]
        assert (level == "error") and (MONITOR_NAME in msg) and ("DOWN" in msg)
        level, (msg,), _ = m.mock_calls[1]
        assert (level == "error") and (MONITOR_NAME in msg) and ("recovered" in msg)
