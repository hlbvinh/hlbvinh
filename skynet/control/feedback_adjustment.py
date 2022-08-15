from typing import List

from ..utils.types import Feedback, StateTemperature
from .prediction_util import is_string_temperature

HIGH_SET_TEMPERATURE_LOWER_BOUND = 27


class FeedbackAdjustment:
    def __init__(self, current_tempset: StateTemperature, feedback: Feedback):
        self.current_tempset = current_tempset
        self.feedback = feedback
        self.is_feedback_update = False

    def override_temperature(self, best_set_temp: str, allowed_tempsets: List) -> str:
        if (
            self._is_adjustment_possible(allowed_tempsets)
            and self._is_adjustment_needed
        ):
            mid_set_temp = sorted(allowed_tempsets, key=float)[
                len(allowed_tempsets) // 2
            ]
            return min(best_set_temp, mid_set_temp, key=float)
        return best_set_temp

    def _is_adjustment_possible(self, allowed_tempsets: List) -> bool:
        return len(allowed_tempsets) > 1 and not is_string_temperature(
            self.current_tempset
        )

    @property
    def _is_adjustment_needed(self) -> bool:
        return (
            self.feedback.get("feedback") == 3
            and self.is_feedback_update
            and self._is_current_set_temp_high
        )

    @property
    def _is_current_set_temp_high(self) -> bool:
        return (
            float(self.current_tempset)  # type: ignore
            > HIGH_SET_TEMPERATURE_LOWER_BOUND
        )
