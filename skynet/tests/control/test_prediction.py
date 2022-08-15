import pytest

from ...control import prediction


@pytest.fixture
def climate_states(state):
    states = []
    for temperature in ["17", "24", "16"]:
        temp = state.copy()
        temp["temperature"] = temperature
        states.append(temp)
    return states


@pytest.fixture
def climate_predictions():
    return [15, 16, 17]


def test_sort_temperatur_predictions(climate_states, climate_predictions):
    states, predictions = prediction.sort_predictions(
        climate_states, climate_predictions, "temperature"
    )
    assert float(states[0]["temperature"]) < float(states[-1]["temperature"])
    assert predictions[0] < predictions[1]


def test_dont_sort_humidity_predictions(climate_states, climate_predictions):
    states, predictions = prediction.sort_predictions(
        climate_states, climate_predictions, "humidity"
    )
    assert climate_states == states
    assert climate_predictions == predictions


@pytest.fixture
def string_states(state):
    states = []
    for temperature in ["27", "blank", "25"]:
        temp = state.copy()
        temp["temperature"] = temperature
        states.append(temp)
    return states


def test_dont_sort_string_temperature_predictions(string_states, climate_predictions):
    states, predictions = prediction.sort_predictions(
        string_states, climate_predictions, "humidity"
    )
    assert string_states == states
    assert climate_predictions == predictions


@pytest.fixture
def multi_mode_states(state):

    states = []
    for mode in ["fan", "cool", "heat"]:
        temp = state.copy()
        temp["mode"] = mode
        states.append(temp)
    return states


def test_fail_sorting_predictions_from_different_modes(
    multi_mode_states, climate_predictions
):
    with pytest.raises(AssertionError):
        prediction.sort_predictions(multi_mode_states, climate_predictions, "comfort")
