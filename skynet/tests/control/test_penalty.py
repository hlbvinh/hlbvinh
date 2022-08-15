from datetime import datetime

import numpy as np
import pytest

from ...control import penalty
from ...utils import thermo
from ...utils.enums import Power


def test_control_target_change_penalty_factor():
    fun = penalty.control_target_change_penalty_factor
    default = penalty.STATE_UPDATE_FACTOR
    trials = [
        (None, "away", 0),
        (None, "comfort", 0),
        ("off", "comfort", 0),
        ("comfort", "off", 0),
        ("manual", "comfort", 0),
        ("comfort", "manual", 0),
        ("away", "comfort", 0),
        ("away", "temperature", 0),
        ("temperature", "temperature", default),
        ("comfort", "comfort", default),
    ]
    for old_mode, new_mode, result in trials:
        assert fun(old_mode, new_mode) == result


@pytest.mark.parametrize(
    "deviations, states, current_state, predicted_deviation",
    [
        pytest.param(
            np.arange(12),
            [
                {"mode": "cool", "temperature": str(i), "power": Power.ON}
                for i in range(20, 32)
            ],
            {"mode": "cool", "temperature": 0, "power": Power.ON},
            penalty.DEFAULT_PREDICTED_DEVIATION,
            id="testing not found",
        ),
        pytest.param(
            np.arange(12),
            [
                {"mode": "cool", "temperature": str(i), "power": Power.ON}
                for i in range(20, 32)
            ],
            {"mode": "cool", "temperature": "na", "power": Power.ON},
            penalty.DEFAULT_PREDICTED_DEVIATION,
            id="testing non numeric set temperature",
        ),
        pytest.param(
            np.arange(12),
            [
                {"mode": "cool", "temperature": str(i), "power": Power.ON}
                for i in range(20, 32)
            ],
            {"mode": "cool", "temperature": 22, "power": Power.ON},
            2,
            id="testing lookup",
        ),
        pytest.param(
            np.arange(12),
            [
                {"mode": "cool", "temperature": str(i), "power": Power.OFF}
                for i in range(20, 32)
            ],
            {"mode": "cool", "temperature": 22, "power": Power.ON},
            2,
            id="Going from OFF to ON still considers previous state",
        ),
    ],
)
def test_current_state_deviation(
    deviations, states, current_state, predicted_deviation
):
    assert (
        penalty.current_state_deviation(deviations, states, current_state)
        == predicted_deviation
    )


def test_normal_l1():
    x = penalty.l1(20.0, 21.0, penalty.L1_FACTOR)
    assert x == 1.0 * penalty.L1_FACTOR


def test_equal():
    x = penalty.l1("blah", "blah", None)
    assert x == 0.0


def test_non_numeric():
    string_temperature = thermo.fix_temperature("blah")
    x = penalty.l1(10.0, np.nan, None)
    assert x == penalty.DEFAULT_PENALTY
    x = penalty.l1(10.0, string_temperature, None)
    assert x == penalty.DEFAULT_PENALTY
    x = penalty.l1(string_temperature, 10.0, None)
    assert x == penalty.DEFAULT_PENALTY


def test_range():
    ct = 25.0
    states = [{"temperature": str(i), "mode": "cool"} for i in range(20, 30)]
    predictions = [1] * 10
    ps = penalty.penalize_deviations(predictions, states, ct)
    assert len(ps) == 10
    l1_factor = np.std(predictions) * penalty.L1_FACTOR
    l2_factor = np.std(predictions) * penalty.L2_FACTOR
    for pen, temperature in zip(ps, range(20, 30)):
        assert (
            pen
            == abs((ct - temperature)) * l1_factor
            + (ct - temperature) ** 2.0 * l2_factor
        )


def test_non_numeric_current_temperature_range():
    ct = "blah"
    temps = np.arange(20, 30)
    states = [{"temperature": str(i)} for i in temps]
    mean_temp = np.mean(temps)
    predictions = [1] * 10
    ps = penalty.penalize_deviations(predictions, states, ct)
    assert len(temps) == len(ps)
    l1_factor = np.std(predictions) * penalty.L1_FACTOR
    l2_factor = np.std(predictions) * penalty.L2_FACTOR
    for pen, temperature in zip(ps, temps):
        assert (
            pen
            == abs((mean_temp - temperature)) * l1_factor
            + (mean_temp - temperature) ** 2.0 * l2_factor
        )


def test_decay():
    t_0 = datetime(2015, 1, 1)
    t = datetime(2015, 1, 10)

    assert penalty.time_factor(None, None) == 1.0
    assert penalty.time_factor(None, None, 10.0) == 1.0
    assert penalty.time_factor(None, t_0) == 1.0
    assert penalty.time_factor(t, None) == 1.0

    assert pytest.approx(penalty.time_factor(t, t_0)) == 0.0

    t_1 = t_0 + penalty.TIME_TAU
    assert np.exp(-1) == penalty.time_factor(t_1, t_0)


def test_error_decay():
    assert penalty.error_factor(1.0) == np.exp(-1.0)
