from copy import deepcopy
from unittest import mock
from datetime import datetime, timedelta

import numpy as np
import pytest

from ...control import adjust, util
from ...control.controller import Controller
from ...utils.log_util import get_logger

log = get_logger(__name__)


class TimestampSource:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        self.t += timedelta(seconds=util.AMBI_SENSOR_INTERVAL_SECONDS)
        return self.t


@pytest.fixture()
def tracker():
    c = mock.Mock(spec=Controller, logger=log, device_id="device_id")

    # mock Controller log method

    def log_fun(msg, **kwargs):
        Controller.log(c, msg, **kwargs)

    return adjust.DeviationTracker(log_fun=log_fun)


def get_redis_tracker(redis, device_id, quantity):
    return adjust.RedisDeviationTracker(redis, device_id, quantity, log_fun=mock.Mock())


@pytest.fixture(params=util.TRACKING_QUANTITIES)
def redis_trackers(request, rediscon, device_id):
    quantity = request.param

    tracker = get_redis_tracker(rediscon, device_id, quantity)
    tracker_bis = get_redis_tracker(rediscon, device_id, quantity)

    return tracker, tracker_bis


@pytest.mark.asyncio
async def test_redis_tracker_state(redis_trackers, t):
    redis_tracker, redis_tracker_bis = redis_trackers
    redis_tracker._add(1.5, t(), "cool")
    tracker_repr = repr(redis_tracker)
    await redis_tracker.store_state()
    assert repr(redis_tracker_bis) != tracker_repr

    await redis_tracker_bis.load_state()
    assert repr(redis_tracker_bis) == tracker_repr


def test_create_redis_trackers(rediscon, device_id):
    trackers = adjust.create_redis_trackers(rediscon, device_id, log_fun=print)
    for quantity in util.TRACKING_QUANTITIES:
        assert isinstance(trackers[quantity], adjust.DeviationTracker)


@pytest.fixture
def t():
    return TimestampSource(datetime.utcnow())


def test_tracker_add(tracker):
    increasing_errors = [1.5, 3]
    timestamps = [datetime(2015, 1, 1, 1, 1), datetime(2015, 1, 1, 1, 1, 30)]

    # test add
    tracker._add(increasing_errors[0], timestamps[0])
    assert len(tracker.error_stream) == 1
    assert tracker.error_stream[0] == (increasing_errors[0], 0.0)
    assert tracker.error_stream.start == timestamps[0]


def test_tracker_add_no_timestamp(tracker):
    tracker._add(1.5)
    assert tracker.error_stream.last_error == 1.5
    tracker._add(3.0)
    assert tracker.error_stream.last_error == 3.0


def test_penalty_remove(tracker, t):
    n_min = int(adjust.MIN_SECONDS / util.AMBI_SENSOR_INTERVAL_SECONDS) + 1
    for i in range(n_min):
        tracker._add(2.0, t(), "cool")
    assert tracker._offset == 1
    for i in range(n_min):
        tracker._add(2.0, t(), "cool")
    assert tracker._offset == 2
    for i in range(n_min):
        tracker._add(2.0 - 0.2 * i, t(), "cool")
    for i in range(n_min):
        tracker._add(-2.0, t(), "cool")
    assert tracker._offset == 1


def test_remove_offset(tracker):
    tracker._offset = -1
    assert tracker.remove_offset(None, [None]) is None
    assert tracker.remove_offset(16, ["16"]) == 16
    assert tracker.remove_offset(16, ["16", "17"]) == "17"
    assert tracker.remove_offset(16.5, ["16", "17", "17.5"]) == "17.5"
    assert tracker.remove_offset(10, ["10", "9", "11"]) == "11"
    assert tracker.remove_offset("blah", ["blah"]) == "blah"


def test_sine_waves(tracker, t):

    # oscillate around 0, offset should stay 0
    for e in np.sin(np.linspace(0, 20, 50)):
        tracker._add(e, t(), "cool")
        assert tracker._offset == 0

    # oscillate around 2 * threshold for a bit, offset should be larger than 0
    for e in 2 * adjust.MEAN_THRESHOLD + np.sin(np.linspace(0, 20, 50)):
        tracker._add(e, t(), "cool")
    assert tracker._offset > 0

    # oscillate around -2 * threshold for a bit, offset should be smaller than 0
    for e in -2 * adjust.MEAN_THRESHOLD + np.sin(np.linspace(0, 20, 200)):
        tracker._add(e, t(), "cool")
    assert tracker._offset < 0

    # oscillate around 2 * threshold for a shorter period
    # offset should go back to 0
    offsets = []
    for e in 2 * adjust.MEAN_THRESHOLD + np.sin(np.linspace(0, 20, 200)):
        tracker._add(e, t(), "cool")
        offsets.append(tracker._offset)
    assert any(offset == 0 for offset in offsets)


def test_tracker_reset(tracker, t):
    # test need reset
    tracker._current_mode = "cool"
    for _ in range(40):
        tracker._add(-2.0, t(), "cool")
    assert tracker._offset < 0
    current_offset = tracker._offset

    tracker._add(1.0, t(), "cool")
    # we crossed zero, errors should be reset
    assert len(tracker.error_stream) == 1
    # offset not reset unless mode changes
    assert tracker._offset == current_offset

    tracker._add(1.0, t(), "heat")
    assert len(tracker.error_stream) == 1
    assert tracker._offset == 0


@pytest.fixture
def allowed_tempsets():
    return [str(i) for i in range(18, 31)]


@pytest.fixture
def best_tempset():
    return "30"


def test_tracker_offset(tracker, best_tempset, allowed_tempsets):
    # test offset is not applied if tracker is not updated for
    # UNUSED_RESET_INTERVAL
    start_time = datetime.utcnow() - (
        adjust.UNUSED_RESET_INTERVAL + timedelta(minutes=1)
    )

    # adding two points, because we only compute seconds_to_target if
    # we have at least 3 points available
    tracker._add(3, start_time, "cool")
    tracker._add(3, start_time + timedelta(seconds=1), "cool")

    # here we jump directly from the last timestamp to "now"
    assert (
        tracker.get_set_temperature_with_offset(
            3,
            best_tempset=best_tempset,
            best_mode="cool",
            allowed_tempsets=allowed_tempsets,
        )
        == "30"
    )


def test_tracker_offset_float(tracker):
    assert (
        tracker.get_set_temperature_with_offset(3, "15.5", "cool", ["15", "16"])
        == "15.5"
    )
    assert (
        tracker.get_set_temperature_with_offset(3, "+0.5", "cool", ["+0", "+1"])
        == "+0.5"
    )


def test_tracker_offset_string(tracker):
    assert tracker.get_set_temperature_with_offset(3, "24", "fan", ["24"]) == "24"
    assert (
        tracker.get_set_temperature_with_offset(3, "+-0", "auto", ["-1", "+-0", "+1"])
        == "+-0"
    )
    assert tracker.get_set_temperature_with_offset(3, None, "fan", [None]) is None


def test_offset_is_applied_if_tracker_is_updated_within_unused_reset_interval(
    tracker, t, best_tempset, allowed_tempsets
):
    flat_error = np.ones(30) * -3
    start_time = datetime.utcnow() - (
        adjust.UNUSED_RESET_INTERVAL - timedelta(minutes=1)
    )
    tracker._add(-3, start_time, "cool")
    adjusted_temperatures = []
    for err in flat_error:
        adjusted_temperatures.append(
            tracker.get_set_temperature_with_offset(
                err,
                best_tempset=best_tempset,
                best_mode="cool",
                allowed_tempsets=allowed_tempsets,
                timestamp=t(),
            )
        )
    assert any(float(temperature) < 30 for temperature in adjusted_temperatures)


@pytest.fixture
def current_timestamp():
    return datetime.utcnow()


def test_offset_comfort_slowly_getting_worse(
    tracker, allowed_tempsets, current_timestamp
):
    best_tempset = 25
    staling_errors = [0.0, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56, 0.57, 0.58, 0.6]
    timestamps = [
        current_timestamp - timedelta(minutes=i)
        for i in reversed(range(len(staling_errors)))
    ]
    offsets = [
        int(
            tracker.get_set_temperature_with_offset(
                error, best_tempset, "cool", allowed_tempsets, timestamp
            )
        )
        - int(best_tempset)
        for error, timestamp in zip(staling_errors, timestamps)
    ]
    initial_offset = offsets[0]
    assert any(offset > initial_offset for offset in offsets)


def test_offset_work_with_humidex_that_is_constantly_close_to_target(
    tracker, best_tempset, allowed_tempsets
):
    flat_but_small_error = np.ones(10) * 0.5
    for err in flat_but_small_error:
        new_t = tracker.get_set_temperature_with_offset(
            err,
            best_tempset=best_tempset,
            best_mode="cool",
            allowed_tempsets=allowed_tempsets,
        )
        assert isinstance(new_t, str)
        assert new_t == "30"


def test_offset_work_with_humidex_that_is_reaching_target_too_slowly(
    tracker, best_tempset, allowed_tempsets
):
    work_change_too_slow = np.linspace(-5, -4, 10)
    slow_timestamps = [
        (datetime(2015, 1, 1, 1, 1) + i * timedelta(minutes=3)) for i in range(10)
    ]
    for err, ts, i in zip(work_change_too_slow, slow_timestamps, range(10)):
        tracker.get_set_temperature_with_offset(
            err,
            best_tempset=best_tempset,
            best_mode="cool",
            allowed_tempsets=allowed_tempsets,
            timestamp=ts,
        )
        # need to have three points to track if change is too slow
        if i >= 2:
            assert (
                float(
                    tracker.get_set_temperature_with_offset(
                        err,
                        best_tempset=best_tempset,
                        best_mode="cool",
                        allowed_tempsets=allowed_tempsets,
                        timestamp=ts,
                    )
                )
                < 30.0
            )


def test_offset_should_not_change_when_the_error_is_decreasing(
    tracker, allowed_tempsets
):
    error_shows_appliance_is_working = np.linspace(5, 1, 5)
    for err in error_shows_appliance_is_working:
        new_t = tracker.get_set_temperature_with_offset(
            err, best_tempset=30, best_mode="cool", allowed_tempsets=allowed_tempsets
        )
        assert isinstance(new_t, str)
        assert new_t == "30"


def test_seconds_to_target():
    # this case will cause poly fit to return a = 0 and b = 1
    assert abs(adjust.seconds_to_target([1, 2, 3], [1, 1, 1])) == adjust.CLIP_SECONDS
    assert np.allclose(adjust.seconds_to_target([1, 2, 3], [3, 2, 1]), 1)
    assert np.allclose(adjust.seconds_to_target([0, 1, 2], [0, 0, 0]), 0)
    assert adjust._to_seconds(0, 1) == adjust.CLIP_SECONDS


def test_deviation_tracker_with_boundary_set_temperatures(tracker, t, allowed_tempsets):

    runs = [
        # Case 1: predicted temperature_set
        # is the lowest available temperature_set
        # user is too hot, offset stay at 0
        (
            0,
            {
                "error": -1.3,
                "best_tempset": 18,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            0,
        ),
        # Case 2: predicted temperature_set
        # is the highest available temperature_set
        # user is too cold, offset stay at 0
        (
            0,
            {
                "error": 1.3,
                "best_tempset": 30,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            0,
        ),
        # Case 3: predicted temperature_set
        # is the not highest or lowest available temperature_set
        # user is too cold, offset = 2
        (
            0,
            {
                "error": 1.3,
                "best_tempset": 25,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            3,
        ),
        # Case 4: predicted temperature_set
        # is the not highest or lowest available temperature_set
        # user is too hot, offset = -2
        (
            0,
            {
                "error": -1.3,
                "best_tempset": 25,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            -3,
        ),
        # Case 5
        # offset is -1 and predictor predicted 18 degree be best temp_set
        # user feeling too cold, the minimum temp set in ir features is 18
        # offset will change to 2
        (
            -1,
            {
                "error": 1.3,
                "best_tempset": 18,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            3,
        ),
        # Case 6
        # offset is 1 and predictor predicted 30 degree be best temp_set
        # user feeling too hot, the maximum temp set in ir features is 30
        # offset will change to -2
        (
            1,
            {
                "error": -1.3,
                "best_tempset": 30,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            -3,
        ),
        # Case 7
        # offset is -1 and predictor predicted 19 degree be best temp_set
        # user feeling too hot, the minimum temp set in ir features is 18
        # offset remain at -1
        (
            -1,
            {
                "error": -1.3,
                "best_tempset": 19,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            -1,
        ),
        # Case 7
        # offset is 1 and predictor predicted 30 degree be best temp_set
        # user feeling too cold, the maximum temp set in ir features is 30
        # offset remain at 1
        (
            1,
            {
                "error": 1.3,
                "best_tempset": 29,
                "best_mode": "cool",
                "allowed_tempsets": allowed_tempsets,
            },
            1,
        ),
    ]

    for run in runs:
        tracker_copy = deepcopy(tracker)
        assert tracker_copy._offset == 0
        for _ in range(40):
            run[1]["timestamp"] = t()
            tracker_copy.get_set_temperature_with_offset(**run[1])
        assert tracker_copy._offset == run[2], run


@pytest.mark.asyncio
async def test_clip_offset_when_max_offset_changed(redis_trackers):
    r1, r2 = redis_trackers
    with mock.patch("skynet.control.adjust.MAX_OFFSET", 4):
        r1._offset = adjust.MAX_OFFSET
        await r1.store_state()
    with mock.patch("skynet.control.adjust.MAX_OFFSET", 2):
        await r2.load_state()
        assert r2._offset == 2


@pytest.mark.asyncio
async def test_regression_serialisable_offset_type(redis_trackers):
    r1, r2 = redis_trackers
    r1._offset = 0
    await r1.store_state()
    await r2.load_state()
    await r2.store_state()


def test_offset_kept_throughout_mode_switch(tracker, t):
    for _ in range(40):
        tracker._add(2.0, t(), "cool")
    cool_offset = tracker._offset
    for _ in range(40):
        tracker._add(-2.0, t(), "heat")
    heat_offset = tracker._offset

    tracker._add(2.0, t(), "cool")
    assert tracker._offset == cool_offset
    tracker._add(-2.0, t(), "heat")
    assert tracker._offset == heat_offset


@pytest.mark.asyncio
async def test_regression_serialisation_of_default_dict(redis_trackers):
    r1, r2 = redis_trackers
    await r1.store_state()
    await r2.load_state()
    r2._current_mode = "heat"
    assert r1._current_mode != r2._current_mode
    r2._offset  # pylint:disable=pointless-statement


def test_is_not_numeric():
    assert not adjust.is_numeric([None])
    assert not adjust.is_numeric([None, 14])
    assert not adjust.is_numeric(["+-0"])
    assert adjust.is_numeric(["14", "15.5", 15, 15.5, "-1", "-1.5", -1])
