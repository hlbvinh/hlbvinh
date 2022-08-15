from datetime import datetime, timedelta

import numpy as np
import pytest

from ..utils import data


def test_is_valid_feature_value():
    d = [
        (None, False),
        (np.nan, False),
        ("a", True),
        ("", True),
        (0, True),
        (1.0, True),
        ([], True),
        ([1.0], True),
    ]
    for value, result in d:
        assert data.is_valid_feature_value(value) is result


def test_group_by():

    # test single string
    records = [{"a": 1, "b": 1}]
    assert {1: [{"b": 1}]} == data.group_by("a", records)
    records = [{"a": 1, "b": 1}, {"a": 2, "b": 2}]
    assert {1: [{"b": 1}], 2: [{"b": 2}]} == data.group_by("a", records)
    records = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    assert {1: [{"b": 1}, {"b": 2}]} == data.group_by("a", records)

    # test multikey
    assert {(1, 1): [{}], (1, 2): [{}]} == data.group_by(("a", "b"), records)
    records = [{"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 2}]
    assert {(1, 1): [{"c": 1}], (1, 2): [{"c": 2}]} == data.group_by(
        ("a", "b"), records
    )

    # test keeping keys
    assert {
        (1, 1): [{"a": 1, "b": 1, "c": 1}],
        (1, 2): [{"a": 1, "b": 2, "c": 2}],
    } == data.group_by(("a", "b"), records, keep_key=True)


def test_intervals():
    td = timedelta
    dt = datetime
    iv = data.intervals
    start = dt(2015, 1, 1)
    end = dt(2015, 1, 2)

    for a, b in [(0, 1), (start, end)]:
        with pytest.raises(ValueError):
            data.intervals([], start=b, end=a)

    assert iv([0], 0, 5) == [5]
    assert iv([2, 2], 0, 5) == [0, 3]
    assert iv([-5], 0, 5) == [5]
    assert iv([0], 0, 1) == [1]
    assert iv([2], 0, 1) == [0]
    assert iv([-1], 0, 2) == [2]

    ts = [datetime(2015, 1, 1)]
    assert data.intervals(ts, start, end) == [td(days=1)]
    assert data.intervals([dt(2015, 1, 1, 22)], start, end) == [td(hours=2)]


def transform_function(listlike):
    return np.abs([e - 70 for e in listlike])


find_nearest_runs = [
    (range(10), 5, None, 5),
    (range(10), 5.5, None, 5),
    (range(10), 13, None, 9),
    (range(10), -12.3, None, 0),
    (np.arange(10), 5.5, None, 5),
    (range(70, 80), 5, transform_function, 75),
]


@pytest.mark.parametrize("listlike, val, transform_function, result", find_nearest_runs)
def test_find_nearest(listlike, val, transform_function, result):
    nearest = data.find_nearest(listlike, val, transform_function)
    assert isinstance(nearest, (int, float))
    assert nearest == result


def test_is_int():
    test_data = [(True, "42"), (False, "a_string"), (False, "32.34"), (True, "-2")]

    for td in test_data:
        assert data.is_int(td[1]) == td[0]


def test_is_float():
    test_data = [
        (False, "42"),
        (False, "a_string"),
        (True, "32.34"),
        (False, "-2"),
        (True, "-2.2"),
    ]

    for td in test_data:
        assert data.is_float(td[1]) == td[0]


def test_max_dict():
    mode_probas = {
        "cool": 0.6686337721142254,
        "dry": 0.318371878092076,
        "fan": 0.012994349793698689,
        "first_layer_cool": 0.9777530866660457,
        "first_layer_heat": 0.022246913333954375,
        "heat": 0.0,
    }
    MULTIMODES = sorted(["cool", "heat", "auto", "fan", "dry"])
    assert data.argmax_dict(mode_probas, MULTIMODES) == "cool"
