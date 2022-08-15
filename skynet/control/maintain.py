from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
from aredis import StrictRedis as Redis

from ..utils import cache_util
from ..utils.progressive_rollout import is_in
from ..utils.timequeue import TimeQueue
from .adjust import ErrorStream, is_numeric
from .target import Target

TRACK_DURATION = timedelta(minutes=20)
MAINTAIN_THRESHOLD = 0.45
ERROR_THRESHOLD = 0.8


class Maintain:
    """
    The Maintain class keeps the current close to the target in tracking modes without
    the use of machine learning. When the current is close to the target, +/-1 degree
    changes should be able to maintain the control. When the current is far from the
    target, the maintain logic will be inactive and the control is managed by machine
    learning predictions from the climate model.
    """

    def __init__(
        self, current_tempset: Optional[str], log_fun: Callable, experiments=[]
    ):
        # Tracks errors within the track duration to determine if current has been
        # close to the target
        self.maintain_tracker = TimeQueue(TRACK_DURATION)
        # Tracks errors to determine if small adjustment is needed to keep current
        # close to the target
        self.error_stream = ErrorStream(log_fun)
        self.current_tempset = current_tempset
        self.log_fun = log_fun
        self.is_in_trend = is_in("maintain_trend", experiments)
        self.is_in_threshold = is_in("maintain_threshold", experiments)

    def maintain_temperature(
        self,
        target: Target,
        current_mode: str,
        best_mode: str,
        best_tempset: Optional[str],
        allowed_tempsets: List[Optional[str]],
    ) -> Optional[str]:
        if not is_numeric(allowed_tempsets + [best_tempset, self.current_tempset]):
            return best_tempset

        self.add(target.scaled_target_delta, current_mode, best_mode)

        new_tempset = self.get_temperature_for_maintenance(
            allowed_tempsets, best_tempset  # type: ignore
        )

        self.log_fun(
            "maintaining",
            maintaining={
                "current_tempset": self.current_tempset,
                "best_tempset": best_tempset,
                "new_tempset": new_tempset,
                "error": target.scaled_target_delta,
                "average_long_term_error": self.average_long_term_error(),
                "average_error": self.error_stream.average_error(),
                "is_maintaining": int(self.is_maintaining()),
                "control_target": target.quantity,
                "minutes_since_last_control_target": target.minute_since_last_control_target,
                "trend": self.trend(is_in=True),
                "is_in_trend": self.is_in_trend,
                "is_in_threshold": self.is_in_threshold,
            },
        )

        if self.tempset_changed(new_tempset):
            self.error_stream = ErrorStream(self.log_fun)

        return new_tempset

    def add(
        self,
        error: float,
        current_mode: str,
        best_mode: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if self.should_reset_error_stream(current_mode, best_mode, error):
            self.error_stream = ErrorStream(self.log_fun)
        self.error_stream.add_to_stream(error, timestamp)
        self.maintain_tracker.append(error, timestamp)

    def should_reset_error_stream(
        self, current_mode: str, best_mode: str, error: float
    ) -> bool:
        return current_mode != best_mode or self.error_stream.crossed_target(error)

    def get_temperature_for_maintenance(
        self, allowed_tempsets: List[str], best_tempset: str
    ) -> str:
        if not self.is_maintaining():
            return best_tempset

        new_tempset = float(self.current_tempset)  # type: ignore

        # FIXME in practice the +1 -1 adjustment never happens because the
        # threshold for errors_not_getting_better is too high: 0.5.
        if (
            self.error_stream.has_enough_points
            and self.error_stream.errors_not_getting_better()
        ):
            if self.error_stream.last_error <= 0:
                new_tempset -= 1
            elif self.error_stream.last_error > 0:
                new_tempset += 1

        for t in allowed_tempsets:
            if new_tempset == float(t):
                return t

        return best_tempset

    def is_maintaining(self) -> bool:
        maintain_threshold = MAINTAIN_THRESHOLD
        error_threshold = ERROR_THRESHOLD
        if self.is_in_threshold:
            maintain_threshold *= 2 / 3
            error_threshold *= 2 / 3
        return (
            self.maintain_tracker.has_enough_data
            and self.average_long_term_error() < maintain_threshold
            # we also want to make sure that the overall trend is not to
            # quickly cross the line (aka derivative is not too big)
            and abs(self.trend()) < maintain_threshold
            and abs(self.error_stream.last_error) < error_threshold
        )

    def average_long_term_error(self) -> float:
        return np.average([abs(e) for e in self.maintain_tracker.items])

    def trend(self, is_in=False):
        items = self.maintain_tracker.items
        if len(items) < 2 or not (self.is_in_trend or is_in):
            return 0
        items = self.maintain_tracker.items
        first, second = items[: len(items) // 2], items[len(items) // 2 :]
        return np.average(second) - np.average(first)

    def tempset_changed(self, new_tempset: str) -> bool:
        return float(new_tempset) != float(self.current_tempset)  # type: ignore


class RedisMaintain(Maintain):
    def __init__(
        self,
        redis: Redis,
        device_id: str,
        current_tempset: Optional[str],
        log_fun: Callable,
        experiments=[],
    ) -> None:
        self.redis = redis
        self.device_id = device_id
        super().__init__(current_tempset, log_fun, experiments)

    async def load_state(self) -> None:
        data = await cache_util.get_maintain_tracker(
            redis=self.redis, key_arg=self.device_id
        )
        self.maintain_tracker = TimeQueue(
            TRACK_DURATION, data["long_term_errors"], data["timestamps"]
        )
        self.error_stream.restart_start(data["start"])
        self.error_stream.extend(zip(data["errors"], data["seconds"]))

    async def store_state(self) -> None:
        data = {
            "long_term_errors": self.maintain_tracker._items,
            "timestamps": self.maintain_tracker._timestamps,
            "start": self.error_stream.start,
            "errors": list(self.error_stream.errors),
            "seconds": list(self.error_stream.seconds),
        }
        await cache_util.set_maintain_tracker(
            redis=self.redis, key_arg=self.device_id, value=data
        )
