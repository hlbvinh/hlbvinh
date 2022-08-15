from operator import itemgetter
from typing import List, Optional

from ..control.target import Target
from ..utils import types
from ..utils.types import Feedback, IRFeature, ModeFeedback, ModePreferences
from . import mode_filtering, mode_model_util


class ModeSelection:
    def __init__(
        self,
        mode_preferences: ModePreferences,
        ir_feature: IRFeature,
        mode_feedback: ModeFeedback,
        latest_feedbacks: List[Feedback],
        target: Target,
    ):
        self.mode_preferences = mode_preferences
        self.ir_feature = ir_feature
        self.mode_feedback = mode_feedback
        self.latest_feedbacks = latest_feedbacks
        self.target = target

    def get_mode_selection(
        self, current_mode: str, temperature: float, temperature_out: float
    ) -> types.ModeSelection:
        if self.overriding_mode:
            return [self.overriding_mode]
        return mode_filtering.filter_mode_selection(
            self.mode_selection,
            temperature,
            temperature_out,
            current_mode,
            self.ir_feature,
            self.target.scaled_mode_model_target_delta,
        )

    @property
    def mode_selection(self) -> types.ModeSelection:
        selection = mode_model_util.select_modes(
            self.mode_preferences, self.target.mode_pref_key
        )
        if self.ir_feature:
            return list(set(selection).intersection(self.ir_feature))
        return []

    @property
    def overriding_mode(self) -> Optional[str]:
        has_no_mode_feedback = not self.mode_feedback
        is_not_in_comfort_mode = self.target.control_mode != "comfort"
        has_more_recent_uncomfortable_feedback = (
            self.has_feedback
            and self.mode_feedback
            and self.latest_feedback["feedback"] != 0
            and self.latest_feedback["created_on"] > self.mode_feedback["created_on"]
        )
        mode_feedback_not_in_mode_selection = (
            self.mode_feedback
            and self.mode_feedback["mode_feedback"] not in self.mode_selection
        )

        if any(
            [
                has_no_mode_feedback,
                is_not_in_comfort_mode,
                has_more_recent_uncomfortable_feedback,
                mode_feedback_not_in_mode_selection,
            ]
        ):
            return None
        return self.mode_feedback["mode_feedback"]

    @property
    def has_feedback(self) -> bool:
        return bool(self.latest_feedback)

    @property
    def latest_feedback(self) -> Feedback:
        latest_feedbacks_sorted_on_created_on = sorted(
            self.latest_feedbacks, key=itemgetter("created_on")
        )
        if latest_feedbacks_sorted_on_created_on:
            return latest_feedbacks_sorted_on_created_on[-1]
        return {}
