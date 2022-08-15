import random

import pytest

from ...control import feedback_adjustment


@pytest.fixture
def feedback_adjustment_(feedback):
    feedback["feedback"] = 3
    fa = feedback_adjustment.FeedbackAdjustment("28", feedback)
    fa.is_feedback_update = True
    return fa


@pytest.fixture
def allowed_tempsets(ir_feature):
    allowed_tempsets = ir_feature["cool"]["temperature"]["value"]
    random.shuffle(allowed_tempsets)
    return allowed_tempsets


@pytest.fixture
def best_set_temp():
    return "22"


@pytest.fixture
def best_set_temp_high():
    return "26"


@pytest.fixture
def mid_set_temp(allowed_tempsets):
    return sorted(allowed_tempsets, key=float)[len(allowed_tempsets) // 2]


def test_feedback_adjustment_best_setpoint_higher(
    feedback_adjustment_, allowed_tempsets, best_set_temp_high, mid_set_temp
):
    assert (
        feedback_adjustment_.override_temperature(best_set_temp_high, allowed_tempsets)
        == mid_set_temp
    )


def test_no_adjustment_best_setpoint_lower(
    feedback_adjustment_, allowed_tempsets, best_set_temp
):
    assert (
        feedback_adjustment_.override_temperature(best_set_temp, allowed_tempsets)
        == best_set_temp
    )


def test_no_adjustment_single_setpoint(feedback_adjustment_, best_set_temp):
    assert (
        feedback_adjustment_.override_temperature(best_set_temp, [best_set_temp])
        == best_set_temp
    )


@pytest.mark.parametrize(
    "current_tempset",
    [
        pytest.param("20", id="no adjustment when the current set point is low"),
        pytest.param("cool", id="no adjustment for string set point"),
    ],
)
def test_no_adjustment(
    feedback_adjustment_, current_tempset, allowed_tempsets, best_set_temp
):
    feedback_adjustment_.current_tempset = current_tempset
    assert (
        feedback_adjustment_.override_temperature(best_set_temp, allowed_tempsets)
        == best_set_temp
    )


@pytest.fixture
def feedback_adjustment_with_no_feedback():
    feedback = {}
    fa = feedback_adjustment.FeedbackAdjustment("28", feedback)
    fa.is_feedback_update = True
    return fa


def test_regression_empty_feedback(
    feedback_adjustment_with_no_feedback, best_set_temp, allowed_tempsets
):
    feedback_adjustment_with_no_feedback.override_temperature(
        best_set_temp, allowed_tempsets
    )
