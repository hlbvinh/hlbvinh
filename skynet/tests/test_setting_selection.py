import numpy as np
import pytest

from ..control import setting_selection
from ..utils.enums import Power


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {
                "states": [{"power": Power.ON, "mode": "dry", "temperature": None}],
                "threshold_type": "upper",
                "predictions": np.array([np.nan]),
                "quantity": "humidity",
            },
            {"power": Power.ON, "mode": "dry", "temperature": None},
            id="Case 1: prediction is 0 since we use default temperature set "
            "for dry mode, then adjustment will be bypassed",
        ),
        pytest.param(
            {
                "states": [
                    {"power": Power.ON, "mode": "cool", "temperature": "16"},
                    {"power": Power.ON, "mode": "cool", "temperature": "17"},
                    {"power": Power.ON, "mode": "cool", "temperature": "18"},
                ],
                "threshold_type": "upper",
                "predictions": np.array([2, -1, 1]),
                "quantity": "temperature",
            },
            {"power": Power.ON, "mode": "cool", "temperature": "16"},
            id="Case 2: threshold is upper, and we can use the hardcoded lowest",
        ),
        pytest.param(
            {
                "states": [
                    {"power": Power.ON, "mode": "dry", "temperature": "16"},
                    {"power": Power.ON, "mode": "dry", "temperature": "17"},
                    {"power": Power.ON, "mode": "dry", "temperature": "18"},
                ],
                "threshold_type": "lower",
                "predictions": np.array([0, 1, -1]),
                "quantity": "humidity",
            },
            {"power": Power.ON, "mode": "dry", "temperature": "17"},
            id="Case 3: threshold is lower, choose the tempset "
            "that generate the highest target delta",
        ),
    ],
)
def test_select_away_mode_setting(kwargs, result):
    assert setting_selection.select_away_mode_setting(**kwargs) == result
