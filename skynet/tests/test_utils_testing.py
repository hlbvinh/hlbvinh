from copy import deepcopy
from datetime import datetime
from typing import Callable, List

import numpy as np

from ..utils import testing

# functions that can take same arguments
FUNCS: List[Callable] = [
    testing.gen_random_feature_normal,
    testing.gen_feature_uniform,
    testing.gen_feature_int,
]
TYPES = [float, float, int]
TEST_DATA = [
    {"name": "temp", "type": "uniform", "min_value": 2, "max_value": 5},
    {"name": "height", "type": "int"},
    {"name": "handsome", "type": "string", "n_values": 4, "string_length": 3},
    {"name": "weight", "type": "uniform", "min_value": 6.6, "max_value": 10},
    {"name": "feature_1", "type": "normal", "mean": 10.0},
    {"name": "feature_2", "type": "string", "n_values": 7, "string_length": 5},
    {"name": "datetime", "type": "timestamp", "start": datetime(2015, 7, 1)},
]
TYPES_TEST_DATA = [
    "uniform",
    "int",
    "string",
    "uniform",
    "normal",
    "string",
    "timestamp",
]


def test_size():
    for fun in FUNCS:
        x = fun(name="test", n_samples=10)
        assert x.size == 10


def test_column_name():
    for fun in FUNCS:
        x = fun(name="test", n_samples=10)
        assert list(x.columns) == ["test"]


def test_type():
    for fun, type_ in zip(FUNCS, TYPES):
        x = fun(name="test", n_samples=10)
        assert x.dtypes[0] == type_


def test_gen_feature_string():
    x = testing.gen_feature_string(
        name="test", n_samples=100, n_values=5, string_length=8
    )
    assert x.size == 100
    assert list(x.columns) == ["test"]
    assert len(x.drop_duplicates().values.tolist()) == 5
    for string in x.values:
        assert len("".join(string)) == 8


def test_gen_matrix():
    x = testing.gen_matrix(TEST_DATA, 10)
    assert x.size == 70
    assert list(x.columns) == [
        "temp",
        "height",
        "handsome",
        "weight",
        "feature_1",
        "feature_2",
        "datetime",
    ]
    for i, type_ in zip(TEST_DATA, TYPES_TEST_DATA):
        temp = i.copy()
        test_type = temp.pop("type")
        assert test_type == type_
        assert x.shape == (10, 7)


def test_gen_feature_datetime():
    x = testing.gen_feature_datetime(name="test", n_samples=10)
    assert x.size == 10
    assert list(x.columns) == ["test"]
    assert np.dtype("datetime64[ns]") == x.dtypes[0]


def test_gen_feature_matrix():
    original = deepcopy(testing.DEFAULT_FEATURE_CONFIG)
    all_keys = list(testing.DEFAULT_FEATURE_CONFIG)
    x = testing.gen_feature_matrix(all_keys, 10)
    assert x.size == len(all_keys) * 10
    assert list(x.columns) == all_keys
    assert original == testing.DEFAULT_FEATURE_CONFIG
