import pytest

from ...prediction import mode_filtering


def test_prevent_using_auto_mode():
    assert mode_filtering.prevent_using_auto_mode(["auto"]) == ["auto"]
    assert mode_filtering.prevent_using_auto_mode(["auto", "dry"]) == ["dry"]


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {"mode_selection": ["cool", "fan"], "scaled_target_delta": -5},
            ["cool"],
            id="discard fan when too far from target and only another mode",
        ),
        pytest.param(
            {"mode_selection": ["heat", "cool", "fan"], "scaled_target_delta": -5},
            ["heat", "cool", "fan"],
            id="keep fan when too far from target and more than two modes",
        ),
        pytest.param(
            {"mode_selection": ["heat", "fan"], "scaled_target_delta": 0.3},
            ["heat", "fan"],
            id="keep fan when close to target",
        ),
    ],
)
def test_prevent_fan_mode_when_too_far_from_target(kwargs, result):
    assert mode_filtering.prevent_fan_mode_when_too_far_from_target(**kwargs) == result


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {
                "mode_selection": ["cool", "heat"],
                "temperature": 22,
                "temperature_out": 20,
            },
            ["cool", "heat"],
            id="should not change anything",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "heat"],
                "temperature": 26,
                "temperature_out": 4,
            },
            ["heat"],
            id="should not try to cool",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "heat", "dry"],
                "temperature": 22,
                "temperature_out": 30,
            },
            ["cool", "dry"],
            id="should not try to heat",
        ),
        pytest.param(
            {"mode_selection": ["heat"], "temperature": 22, "temperature_out": 30},
            ["heat"],
            id="should not have an empty selection",
        ),
    ],
)
def test_prevent_bad_heat_cool_selection(kwargs, result):
    assert mode_filtering.prevent_bad_heat_cool_selection(**kwargs) == result


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {
                "current_mode": "cool",
                "mode_selection": ["dry"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 2.0,
            },
            ["dry"],
            id="If dry mode is the only mode selected then keep it.",
        ),
        pytest.param(
            {
                "current_mode": "dry",
                "mode_selection": ["cool", "dry"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 2.0,
            },
            ["cool"],
            id="If using dry mode and its too cold for dry mode",
        ),
        pytest.param(
            {
                "current_mode": "dry",
                "mode_selection": ["cool", "dry"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 1.4,
            },
            ["cool", "dry"],
            id="If using dry mode and not too cold for dry mode",
        ),
        pytest.param(
            {
                "current_mode": "cool",
                "mode_selection": ["cool", "dry"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 1.4,
            },
            ["cool"],
        ),
        pytest.param(
            {
                "current_mode": "cool",
                "mode_selection": ["cool", "dry"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": -0.1,
            },
            ["cool", "dry"],
        ),
        pytest.param(
            {
                "current_mode": "dry",
                "mode_selection": ["auto", "cool", "dry", "heat"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 2.0,
            },
            ["auto", "cool", "heat"],
        ),
        pytest.param(
            {
                "current_mode": "dry",
                "mode_selection": ["auto", "cool", "dry", "heat"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 1.4,
            },
            ["auto", "cool", "dry", "heat"],
        ),
        pytest.param(
            {
                "current_mode": "auto",
                "mode_selection": ["auto", "cool", "dry", "heat"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": 1.4,
            },
            ["auto", "cool", "heat"],
        ),
        pytest.param(
            {
                "current_mode": "auto",
                "mode_selection": ["auto", "cool", "dry", "heat"],
                "ir_feature": {"dry": {"temperature": {"value": ["1", "0"]}}},
                "scaled_target_delta": -0.1,
            },
            ["auto", "cool", "dry", "heat"],
        ),
        pytest.param(
            {
                "current_mode": "dry",
                "mode_selection": ["auto", "cool", "dry", "heat"],
                "ir_feature": {"dry": {"temperature": {"value": ["0"]}}},
                "scaled_target_delta": 1.4,
            },
            ["auto", "cool", "heat"],
            id="If dry mode has a single temperature value then don't select dry mode",
        ),
    ],
)
def test_prevent_dry_mode_from_cooling_too_much(kwargs, result):
    assert mode_filtering.prevent_dry_mode_from_cooling_too_much(**kwargs) == result


@pytest.fixture(params=["one temperature", "no temperature", "no dry"])
def single_temperature_dry_mode_ir_feature(request):
    if request.param == "one temperature":
        return {"dry": {"temperature": {"ftype": "select_option", "value": ["24"]}}}
    if request.param == "no temperature":
        return {"dry": {}}
    return {}


@pytest.fixture
def multi_temperature_dry_mode_ir_feature():
    return {
        "dry": {"temperature": {"ftype": "select_option", "value": ["23", "24", "25"]}}
    }


def test_dry_mode_has_a_single_setting(
    single_temperature_dry_mode_ir_feature, multi_temperature_dry_mode_ir_feature
):
    assert mode_filtering.dry_mode_has_a_single_setting(
        single_temperature_dry_mode_ir_feature
    )
    assert not mode_filtering.dry_mode_has_a_single_setting(
        multi_temperature_dry_mode_ir_feature
    )
