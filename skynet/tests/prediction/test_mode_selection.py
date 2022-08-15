from datetime import datetime, timedelta
from unittest import mock

import pytest

from ...control import prediction, target
from ...utils.types import ModePrefKey


@pytest.fixture
def mode_preferences(control_mode, quantity, threshold_type):
    return {ModePrefKey(control_mode, quantity, threshold_type): ["cool", "dry", "fan"]}


@pytest.fixture
def control_mode():
    return "comfort"


@pytest.fixture
def quantity():
    return "humidex"


@pytest.fixture
def threshold_type():
    return None


@pytest.fixture
def default_target(control_mode, quantity, threshold_type):
    m = mock.Mock(spec=target.Target)
    m.mode_pref_key = ModePrefKey(control_mode, quantity, threshold_type)
    m.control_mode = control_mode
    return m


@pytest.fixture
def mode_selection(
    mode_preferences, ir_feature, mode_feedback, feedback, default_target
):
    mode_selection = prediction.ModeSelection(
        mode_preferences, ir_feature, mode_feedback, [feedback], default_target
    )
    assert mode_selection.overriding_mode is mode_feedback["mode_feedback"]
    return mode_selection


@pytest.fixture
def mode_feedback(device_id):
    return dict(device_id=device_id, created_on=datetime.utcnow(), mode_feedback="fan")


@pytest.fixture
def recent_uncomfortable_feedback(feedback, mode_feedback):
    f = feedback.copy()
    f["feedback"] = 1
    f["created_on"] = mode_feedback["created_on"] + timedelta(minutes=5)
    return f


def test_recent_uncomfortable_feedback_should_override_mode_feedback(
    mode_selection, recent_uncomfortable_feedback
):
    mode_selection.latest_feedbacks = [recent_uncomfortable_feedback]
    assert mode_selection.overriding_mode is None


@pytest.fixture
def recent_comfortable_feedback(feedback, mode_feedback):
    f = feedback.copy()
    f["feedback"] = 0
    f["created_on"] = mode_feedback["created_on"] + timedelta(minutes=5)
    return f


def test_recent_comfortable_feedback_should_not_affect_mode_feedback(
    mode_selection, recent_comfortable_feedback
):
    overriding_mode = mode_selection.overriding_mode
    mode_selection.latest_feedbacks = [recent_comfortable_feedback]
    assert mode_selection.overriding_mode == overriding_mode


@pytest.fixture(params=[0, 1, None])
def old_feedback(request, feedback, mode_feedback):
    if request.param is None:
        return {}
    f = feedback.copy()
    f["feedback"] = request.param
    f["created_on"] = mode_feedback["created_on"] - timedelta(minutes=5)
    return f


def test_old_feedback_should_not_affect_mode_feedback(mode_selection, old_feedback):
    overriding_mode = mode_selection.overriding_mode
    mode_selection.latest_feedbacks = [old_feedback] if old_feedback else []
    assert mode_selection.overriding_mode == overriding_mode


@pytest.fixture
def mode_preferences_without_fan(control_mode, quantity, threshold_type):
    return {ModePrefKey(control_mode, quantity, threshold_type): ["cool", "dry"]}


@pytest.fixture
def mode_selection_without_fan_preference(mode_selection, mode_preferences_without_fan):
    mode_selection.mode_preferences = mode_preferences_without_fan
    return mode_selection


@pytest.fixture
def fan_mode_feedback(mode_feedback):
    m = mode_feedback.copy()
    m["mode_feedback"] = "fan"
    return m


def test_mode_feedback_not_used_when_mode_not_in_mode_preference(
    mode_selection_without_fan_preference, fan_mode_feedback
):
    mode_selection_without_fan_preference.mode_feedback = fan_mode_feedback
    assert mode_selection_without_fan_preference.overriding_mode is None


@pytest.fixture
def mode_selection_without_mode_preference(mode_selection):
    mode_selection.mode_preferences = {}
    return mode_selection


def test_regression_mode_feedback_works_with_empty_mode_preference(
    mode_selection_without_mode_preference, mode_feedback
):
    mode_selection_without_mode_preference.mode_feedback = mode_feedback
    mode_selection_without_mode_preference.overriding_mode  # pylint:disable=pointless-statement


@pytest.fixture
def cool_mode_feedback(mode_feedback):
    f = mode_feedback.copy()
    f["mode_feedback"] = "cool"
    return f


@pytest.fixture(params=["off", "away", "temperature"])
def non_comfort_mode_selection(request, mode_selection, cool_mode_feedback):
    mode_selection.target.control_mode = request.param
    assert mode_selection.target.control_mode != "comfort"
    assert cool_mode_feedback["mode_feedback"] in mode_selection.mode_selection
    return mode_selection


def test_mode_feedback_should_not_affect_non_comfort_modes(
    non_comfort_mode_selection, cool_mode_feedback
):
    overriding_mode = non_comfort_mode_selection.overriding_mode
    non_comfort_mode_selection.mode_feedback = cool_mode_feedback
    assert non_comfort_mode_selection.overriding_mode == overriding_mode
