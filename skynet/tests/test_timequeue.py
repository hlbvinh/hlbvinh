from datetime import datetime, timedelta

import pytest

from ..utils.timequeue import TimeQueue


@pytest.fixture
def max_size():
    return timedelta(minutes=5)


@pytest.fixture
def timequeue(max_size):
    return TimeQueue(max_size)


@pytest.fixture
def item():
    return 0.5


@pytest.fixture
def timestamp():
    return datetime.utcnow()


def test_append(timequeue, item, timestamp):
    timequeue.append(item, timestamp)
    assert timequeue.items[-1] == item
    assert timequeue.timestamps[-1] == timestamp


def test_isfull(timequeue, item, timestamp, max_size):
    timequeue.append(item, timestamp)
    assert not timequeue.has_enough_data
    timequeue.append(item, timestamp + max_size)
    assert timequeue.has_enough_data


def test_outside_limit(timequeue, item, timestamp, max_size):
    timequeue.append(item, timestamp)
    next_item, next_timestamp = item + 10, timestamp + max_size
    timequeue.append(next_item, next_timestamp)
    assert timequeue.items == [next_item]
    assert timequeue.timestamps == [next_timestamp]


def test_filter_queue(timequeue, item, timestamp, max_size):
    timequeue.append(item, timestamp)
    timequeue.append(item, timestamp + max_size)
    assert timequeue.has_enough_data
