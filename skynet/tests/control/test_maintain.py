from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest

from skynet.control import maintain


@pytest.fixture()
def maintain_(state):
    return maintain.Maintain(state["temperature"], log_fun=mock.Mock())


@pytest.fixture
def redis_maintain(rediscon, device_id, state):
    return maintain.RedisMaintain(
        rediscon, device_id, state["temperature"], log_fun=mock.Mock()
    )


@pytest.fixture
def small_error():
    return 0.05


@pytest.fixture
def best_mode(state):
    return state["mode"]


@pytest.fixture
def best_tempset():
    return "30"


@pytest.fixture
def allowed_tempsets():
    return [str(i) for i in range(18, 31)]


@pytest.fixture
def current_timestamp():
    return datetime.utcnow()


@pytest.fixture
def reverse_timestamps(current_timestamp):
    return [
        current_timestamp - timedelta(minutes=minute) for minute in reversed(range(25))
    ]


@pytest.fixture
def timestamps(current_timestamp):
    return [current_timestamp + timedelta(minutes=minute) for minute in range(25)]


def test_no_tempset_change(
    maintain_,
    reverse_timestamps,
    small_error,
    state,
    best_mode,
    best_tempset,
    allowed_tempsets,
):
    for timestamp in reverse_timestamps:
        maintain_.add(small_error, state["mode"], best_mode, timestamp)

    assert (
        maintain_.get_temperature_for_maintenance(allowed_tempsets, best_tempset)
        == state["temperature"]
    )


def test_exit_maintain(
    maintain_,
    current_timestamp,
    reverse_timestamps,
    small_error,
    state,
    best_mode,
    best_tempset,
    allowed_tempsets,
):
    for timestamp in reverse_timestamps:
        maintain_.add(small_error, state["mode"], best_mode, timestamp)

    maintain_.add(
        maintain.ERROR_THRESHOLD + small_error,
        state["mode"],
        best_mode,
        current_timestamp,
    )

    assert (
        maintain_.get_temperature_for_maintenance(allowed_tempsets, best_tempset)
        == best_tempset
    )


def test_back_to_starting_tempset(
    maintain_,
    reverse_timestamps,
    timestamps,
    state,
    best_mode,
    best_tempset,
    allowed_tempsets,
):
    # Increasing errors
    for error, timestamp in zip(
        np.linspace(0, 0.74, len(reverse_timestamps)), reverse_timestamps
    ):
        maintain_.add(error, state["mode"], best_mode, timestamp)

    assert (
        maintain_.get_temperature_for_maintenance(allowed_tempsets, best_tempset)
        != state["temperature"]
    )

    # Decreasing errors
    for error, timestamp in zip(np.linspace(0.74, 0, len(timestamps)), timestamps):
        maintain_.add(error, state["mode"], best_mode, timestamp)

    assert (
        maintain_.get_temperature_for_maintenance(allowed_tempsets, best_tempset)
        == state["temperature"]
    )


@pytest.mark.asyncio
async def test_load_store_state(redis_maintain, small_error, state, best_mode):
    m = redis_maintain
    m.add(small_error, state["mode"], best_mode)
    await m.store_state()
    del m

    m = redis_maintain
    await m.load_state()
    assert len(m.maintain_tracker.items) == 1
