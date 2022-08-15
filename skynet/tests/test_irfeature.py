import json
from itertools import permutations

import pytest

from ..utils import ir_feature
from ..utils.enums import Power

#
# TODO make work without temperature settings for modes without temperature
#


def get_state(power, mode, fan, temperature, louver, swing):
    return {
        "power": power,
        "mode": mode,
        "fan": fan,
        "temperature": temperature,
        "louver": louver,
        "swing": swing,
        "ventilation": None,
    }


STATES = [
    get_state(*properties)
    for properties in [
        (Power.ON, "cool", "high", 22, "up", "mid"),
        (Power.OFF, "heat", "high", 18, "up", "mid"),
        (Power.ON, "cool", "high", "22", "up", "right"),
    ]
]


def test_selective_preserve_fan(device_id):
    ir_feat = {
        "dry": {
            "fan": {"ftype": "select_option", "value": ["high", "med", "low", "auto"]}
        }
    }
    ir_feat["cool"] = ir_feat["dry"]

    # if setting fan, should contain the set fan setting
    deployment = ir_feature.best_possible_deployment(
        {"fan": "auto"}, {"fan": "high"}, {"fan": "low"}, ir_feat, device_id
    )
    assert deployment.fan == "low"

    # if setting fan that is not in ir_feature, should contain
    # last_on_state setting
    deployment = ir_feature.best_possible_deployment(
        {"fan": "auto"}, {"fan": "high"}, {"fan": "blah"}, ir_feat, device_id
    )
    assert deployment.fan == "high"

    # if not setting fan, should contain fan setting of last_on_state
    deployment = ir_feature.best_possible_deployment(
        {"fan": "auto"}, {"fan": "high"}, {"mode": "cool"}, ir_feat, device_id
    )
    assert deployment.fan == "high"


def test_keep_fan(device_id):

    irfeature = get_irfeature()["data"]
    for state, on_state, best_setting in permutations(STATES, 3):
        # TODO better test, this one might just raise every time
        #      in the meantime, check coverage
        try:
            deployment = ir_feature.best_possible_deployment(
                state, on_state, best_setting, irfeature, device_id
            )
        except ValueError:
            continue

        for prop in ["fan", "louver", "swing"]:
            assert getattr(deployment, prop) == on_state[prop]

        for prop in ["temperature", "mode", "power"]:
            assert getattr(deployment, prop) == best_setting[prop]


def test_button_press():

    # check for error if states are equal
    for state_a, state_b in zip(*[STATES] * 2):
        with pytest.raises(ValueError):
            ir_feature.button_press(state_a, state_b)

    # check right button for all properties
    for k in ir_feature.APPLIANCE_PROPERTIES:

        # temperature also has temp_up button
        if k == "temperature":
            state_a = {k: 2}
            state_b = {k: 1}
            b = ir_feature.button_press(state_a, state_b)
            assert b == "temp_down"
            state_a = {k: 1}
            state_b = {k: 2}
            b = ir_feature.button_press(state_a, state_b)
            assert b == "temp_up"
        # power off -> off raises error as no button required
        elif k == "power":
            state_a = {k: Power.ON}
            state_b = {k: Power.OFF}
            assert ir_feature.button_press(state_a, state_b) == "power"
            assert ir_feature.button_press(state_b, state_a) == "power"
            # raise if power stays the same and is the only property
            for s in [state_a, state_b]:
                with pytest.raises(ValueError):
                    ir_feature.button_press(s, s)
        else:
            state_a = {k: "a"}
            state_b = {k: "b"}
            b = ir_feature.button_press(state_a, state_b)
            assert b == k


def get_irfeature(default=False):
    if not default:
        return json.loads(
            """
        {
            "appliance_id": "f8abbbaf-8226-4977-8605-2ab603597dfc",
            "data": {
                "dry": {
                    "fan": {
                        "ftype": "select_option",
                        "value": [
                            "high",
                            "med",
                            "low",
                            "auto"
                        ]
                    },
                    "louver": {
                        "ftype": "select_option",
                        "value": [
                            "off",
                            "up",
                            "mid-up",
                            "mid",
                            "mid-down",
                            "auto",
                            "down"
                        ]
                    },
                    "temperature": {
                        "ftype": "select_option",
                        "value": [
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30"
                        ]
                    },
                    "swing": {
                        "ftype": "select_option",
                        "value": [
                            "off",
                            "mid-left",
                            "mid",
                            "mid-right",
                            "right",
                            "left-right",
                            "right-left",
                            "auto",
                            "left"
                        ]
                    }
                },
                "auto": {
                    "swing": {
                        "ftype": "select_option",
                        "value": [
                            "auto",
                            "right-left",
                            "left-right",
                            "right",
                            "mid-right",
                            "mid",
                            "mid-left",
                            "left",
                            "off"
                        ]
                    },
                    "fan": {
                        "ftype": "select_option",
                        "value": [
                            "low",
                            "auto",
                            "high",
                            "med"
                        ]
                    },
                    "louver": {
                        "ftype": "select_option",
                        "value": [
                            "up",
                            "down",
                            "mid-down",
                            "mid",
                            "mid-up",
                            "off",
                            "auto"
                        ]
                    },
                    "temperature": {
                        "ftype": "select_option",
                        "value": [
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30"
                        ]
                    }
                },
                "brand": "Mitsubishi Heavy Industries Ltd",
                "heat": {
                    "fan": {
                        "ftype": "select_option",
                        "value": [
                            "med",
                            "high",
                            "auto",
                            "low"
                        ]
                    },
                    "louver": {
                        "ftype": "select_option",
                        "value": [
                            "off",
                            "up",
                            "mid-up",
                            "mid-down",
                            "down",
                            "mid",
                            "auto"
                        ]
                    },
                    "temperature": {
                        "ftype": "select_option",
                        "value": [
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30"
                        ]
                    },
                    "swing": {
                        "ftype": "select_option",
                        "value": [
                            "right",
                            "left-right",
                            "right-left",
                            "auto",
                            "mid-right",
                            "mid",
                            "mid-left",
                            "left",
                            "off"
                        ]
                    }
                },
                "fan": {
                    "fan": {
                        "ftype": "select_option",
                        "value": [
                            "low",
                            "med",
                            "high",
                            "auto"
                        ]
                    },
                    "louver": {
                        "ftype": "select_option",
                        "value": [
                            "mid-down",
                            "mid",
                            "mid-up",
                            "up",
                            "off",
                            "down",
                            "auto"
                        ]
                    },
                    "swing": {
                        "ftype": "select_option",
                        "value": [
                            "off",
                            "auto",
                            "right-left",
                            "left-right",
                            "right",
                            "mid-right",
                            "mid",
                            "mid-left",
                            "left"
                        ]
                    }
                },
                "model": "RKX502A001C",
                "cool": {
                    "fan": {
                        "ftype": "select_option",
                        "value": [
                            "auto",
                            "high",
                            "med",
                            "low"
                        ]
                    },
                    "louver": {
                        "ftype": "select_option",
                        "value": [
                            "off",
                            "up",
                            "mid-up",
                            "mid",
                            "mid-down",
                            "auto",
                            "down"
                        ]
                    },
                    "temperature": {
                        "ftype": "select_option",
                        "value": [
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30"
                        ]
                    },
                    "swing": {
                        "ftype": "select_option",
                        "value": [
                            "auto",
                            "right-left",
                            "off",
                            "left",
                            "mid-left",
                            "mid",
                            "mid-right",
                            "right",
                            "left-right"
                        ]
                    }
                }
            },
            "result_code": 200
        }"""
        )
    return json.loads(
        """
        {
            "appliance_id": "f8abbbaf-8226-4977-8605-2ab603597dfc",
            "data":{
            "cool": {
                "fan": {"ftype": "select_option", "value": ["auto"]},
                "temperature": {"ftype": "select_option", "value": ["25"]}
            }
            },
            "result_code":200
        }"""
    )
