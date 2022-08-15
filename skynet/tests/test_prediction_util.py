import pytest

from ..control import prediction_util
from ..utils.enums import Power


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "mode": "fan",
            "ir_feature": {"fan": {"temperature": {"value": None}}},
            "current_temperature": 22.1,
            "current_tempset": None,
        },
        {
            "mode": "auto",
            "ir_feature": {"auto": {}},
            "current_temperature": 32.3,
            "current_tempset": "26",
        },
    ],
)
def test_get_default_state_no_temperature_setting(kwargs):
    default_setting = prediction_util.get_default_state(**kwargs)
    assert default_setting == {
        "mode": kwargs["mode"],
        "power": Power.ON,
        "temperature": None,
    }


def test_get_default_state_one_temperature_setting():

    default_setting = prediction_util.get_default_state(
        "auto", {"auto": {"temperature": {"value": ["auto"]}}}, 22.1, "auto"
    )
    assert default_setting == {"mode": "auto", "power": Power.ON, "temperature": "auto"}


def test_get_default_state_all_strings():

    default_setting = prediction_util.get_default_state(
        "auto", {"auto": {"temperature": {"value": ["low", "mid", "high"]}}}, 25.5, "25"
    )
    assert default_setting == {"mode": "auto", "power": Power.ON, "temperature": "mid"}


def test_is_string_temperature():
    for temperature in ["1", "+1", "16", "77", "77.5"]:
        assert not prediction_util.is_string_temperature(temperature)
    for temperature in ["+-0", "mid"]:
        assert prediction_util.is_string_temperature(temperature)


@pytest.mark.parametrize(
    "kwargs, default_setting",
    [
        (
            {
                "current_temperature": 23.5,
                "current_tempset": "auto",
                "ir_feature": {
                    "fan": {"temperature": {"value": [str(i) for i in range(16, 31)]}}
                },
                "mode": "fan",
            },
            {"mode": "fan", "power": Power.ON, "temperature": "21"},
        ),
        (
            {
                "current_temperature": 23.5,
                "current_tempset": "22",
                "ir_feature": {
                    "fan": {"temperature": {"value": [str(i) for i in range(16, 31)]}}
                },
                "mode": "fan",
            },
            {"mode": "fan", "power": Power.ON, "temperature": "21"},
        ),
        (
            {
                "current_temperature": 22,
                "current_tempset": 16,
                "ir_feature": {
                    "fan": {"temperature": {"value": [str(i) for i in range(16, 31)]}}
                },
                "mode": "fan",
            },
            {"mode": "fan", "power": Power.ON, "temperature": "16"},
        ),
        (
            {
                "current_temperature": 22,
                "current_tempset": "16",
                "ir_feature": {
                    "fan": {"temperature": {"value": [str(i) for i in range(20, 22)]}}
                },
                "mode": "fan",
            },
            {"mode": "fan", "power": Power.ON, "temperature": "20"},
        ),
        (
            {
                "current_temperature": 16,
                "current_tempset": "22",
                "ir_feature": {
                    "fan": {"temperature": {"value": [str(i) for i in range(16, 31)]}}
                },
                "mode": "fan",
            },
            {"mode": "fan", "power": Power.ON, "temperature": "16"},
        ),
    ],
)
def test_get_default_state_fan_mode(kwargs, default_setting):
    assert prediction_util.get_default_state(**kwargs) == default_setting


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "current_temperature": 23.5,
            "current_tempset": 16,
            "ir_feature": {
                "cool": {"temperature": {"value": [str(i) for i in range(16, 31)]}}
            },
            "mode": "cool",
        },
        {
            "current_temperature": 23.5,
            "current_tempset": 16,
            "ir_feature": {
                "cool": {
                    "temperature": {"value": [str(i + 0.5) for i in range(16, 31)]}
                }
            },
            "mode": "cool",
        },
        {
            "current_temperature": 23.5,
            "current_tempset": 16,
            "ir_feature": {"auto": {"temperature": {"value": ["-1", "0", "+1"]}}},
            "mode": "auto",
        },
    ],
)
def test_no_default_state(kwargs):
    assert prediction_util.get_default_state(**kwargs) is None


def test_find_nearest():
    assert prediction_util.find_nearest(23.3, ["23", "23.5", "24"]) == "23.5"
    assert prediction_util.find_nearest(14, ["16", "17"]) == "16"


def test_is_room_temperature():
    for temperature in ["23", "23.5", "78", "78.5"]:
        assert prediction_util.is_room_temperature(temperature)
    for temperature in ["+1", "-1.5", "auto"]:
        assert not prediction_util.is_room_temperature(temperature)


@pytest.mark.parametrize(
    "values, default",
    [
        (["auto", "blank", "high", "low"], "auto"),
        (["auto", "Blank", "high", "low"], "Blank"),
        (["high", "med-high", "med", "low"], "med"),
        (["high", "low"], "low"),
    ],
)
def test_get_default_string_values(values, default):
    assert prediction_util.get_default_string_values(values) == default
