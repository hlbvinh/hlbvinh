import time
from datetime import datetime, timedelta

import numpy as np

from ..utils import misc


def test_update_counter():
    u = misc.UpdateCounter()
    u.start()
    time.sleep(0.1)
    u.update()
    assert np.allclose(u.read(), 600, rtol=0.3)
    u = misc.UpdateCounter()
    u.start()
    time.sleep(0.1)
    u.update()
    assert np.allclose(u.read(reset=True), 600, rtol=0.5)
    assert u.read() == 0


def test_elapsed():
    timestamp, period = None, timedelta(milliseconds=5)
    assert misc.elapsed(timestamp, period) is True
    timestamp, period = datetime.utcnow(), timedelta(milliseconds=1)
    assert misc.elapsed(timestamp, period) is False
    time.sleep(0.01)
    assert misc.elapsed(timestamp, period) is True
