from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Callable, DefaultDict, Deque, Dict, List, Optional, Sequence

import numpy as np
from aredis import StrictRedis as Redis

from ..utils import cache_util
from ..utils.types import StateTemperature
from .util import TRACKING_QUANTITIES

N_ERRORS_KEEP = 60
MAX_SECONDS = 30 * 60
CLIP_SECONDS = 69120  # 1 day
MIN_SECONDS = 300.0
UNUSED_RESET_INTERVAL = timedelta(minutes=5)
MEAN_THRESHOLD = 0.5

MAX_OFFSET = 4
ERROR_DECAY_TAU = 10


class ErrorStream:
    def __init__(self, log_fun: Callable):
        self.errors: Deque[float] = deque(maxlen=N_ERRORS_KEEP)
        self.seconds: Deque[float] = deque(maxlen=N_ERRORS_KEEP)
        self.start = datetime.utcnow()
        self.log_fun = log_fun

    def __len__(self):
        return len(self.errors)

    def __getitem__(self, index):
        return self.errors[index], self.seconds[index]

    def _add(self, error, second):
        self.errors.append(error)
        self.seconds.append(second)
        assert len(self.errors) == len(self.seconds)

    def extend(self, errors):
        for error, second in errors:
            self._add(error, second)

    def add_to_stream(self, error, timestamp):
        self._add(error, self.elapsed_time(timestamp))

    def elapsed_time(self, timestamp):
        if not self.errors:
            self.restart_start(timestamp)
            return 0.0

        t = timestamp or datetime.utcnow()
        elapsed = (t - self.start).total_seconds()
        seconds_since_last = elapsed - self.last_second
        if seconds_since_last > UNUSED_RESET_INTERVAL.total_seconds():
            self.log_fun(
                "reset tracker: unused for " f"{seconds_since_last / 60:.1f} min"
            )
            self.reset()
        return elapsed

    def restart_start(self, timestamp):
        self.start = timestamp or datetime.utcnow()

    def reset(self):
        self.errors = deque(maxlen=N_ERRORS_KEEP)
        self.seconds = deque(maxlen=N_ERRORS_KEEP)

    def crossed_target(self, error: float) -> bool:
        if self.errors:
            return np.sign(error) != np.sign(self.last_error)
        return False

    def errors_not_getting_better(self):
        return (
            self.long_enough()
            and self.is_diverging()
            and self.error_not_reaching_zero()
        )

    def long_enough(self):
        return self.last_second >= MIN_SECONDS

    def is_diverging(self):
        return self.average_error() > MEAN_THRESHOLD

    def average_error(self):
        return abs(
            np.average(
                self.errors,
                weights=np.flip(np.exp(-np.arange(len(self.errors)) / ERROR_DECAY_TAU)),
            )
        )

    def error_not_reaching_zero(self):
        return (
            seconds_to_target(self.seconds, self.errors) < 0
            or seconds_to_target(self.seconds, self.errors)
            > self.adjusted_max_seconds()
        )

    def adjusted_max_seconds(self):
        return MAX_SECONDS * max(1, np.log(abs(self.last_error) / MEAN_THRESHOLD))

    @property
    def has_enough_points(self) -> bool:
        return len(self) > 2

    @property
    def last_error(self):
        return self.errors[-1]

    @property
    def last_second(self):
        return self.seconds[-1]


def create_redis_trackers(
    redis: Redis, device_id: str, log_fun: Callable
) -> Dict[str, "RedisDeviationTracker"]:
    return {
        quantity: RedisDeviationTracker(redis, device_id, quantity, log_fun)
        for quantity in TRACKING_QUANTITIES
    }


class DeviationTracker:
    def __init__(self, log_fun: Callable) -> None:
        self.log_fun = log_fun
        self._mode_offset: DefaultDict[str, int] = defaultdict(int)
        self._current_mode = ""
        self.error_stream: ErrorStream = ErrorStream(log_fun)
        self._lower_lim: float = -2.0
        self._upper_lim: float = 2.0

    @property
    def _offset(self):
        return self._mode_offset[self._current_mode]

    @_offset.setter
    def _offset(self, value):
        self._mode_offset[self._current_mode] = value

    def get_set_temperature_with_offset(
        self,
        error: float,
        best_tempset: Optional[str],
        best_mode: str,
        allowed_tempsets: List[Optional[str]],
        timestamp=None,
    ) -> Optional[str]:

        if not is_numeric(allowed_tempsets + [best_tempset]):
            return best_tempset

        self._set_offset_limits(best_tempset, allowed_tempsets)  # type: ignore

        self._add(error, timestamp, best_mode)

        set_temp_with_offset = self._adjusted_set_temperature(
            best_tempset, allowed_tempsets  # type: ignore
        )

        log_data = {
            "best_tempset": float(best_tempset),  # type: ignore
            "adjusted_temperature_set": float(set_temp_with_offset),
            "offset": self._offset,
        }
        self.log_fun("deviation_tracker", deviation_tracker=log_data)

        return set_temp_with_offset

    def _set_offset_limits(self, best_tempset: str, allowed_tempsets: List[str]):
        self._lower_lim = min([float(t) for t in allowed_tempsets]) - float(
            best_tempset
        )
        self._upper_lim = max([float(t) for t in allowed_tempsets]) - float(
            best_tempset
        )

    def _add(self, error: float, timestamp=None, best_mode=None) -> None:
        # Adds error to error stream when the mode has not changed
        if best_mode != self._current_mode or self.error_stream.crossed_target(error):
            self.error_stream = ErrorStream(log_fun=self.log_fun)
        self._current_mode = best_mode
        self.error_stream.add_to_stream(error, timestamp)
        self._adjust_offset()

    def _adjust_offset(self) -> None:
        if self.error_stream.has_enough_points:
            offset = self._offset
            if self.error_stream.errors_not_getting_better():
                if self.error_stream.last_error <= 0:
                    limit = max(self._lower_lim, -MAX_OFFSET)
                    offset = max(offset - 1, limit)
                elif self.error_stream.last_error > 0:
                    limit = min(self._upper_lim, MAX_OFFSET)
                    offset = min(offset + 1, limit)

            log_data = {
                "error": self.error_stream.last_error,
                "mins": self.error_stream.last_second / 60,
                "abs_mean": self.error_stream.average_error(),
                "t0_mins": seconds_to_target(
                    self.error_stream.seconds, self.error_stream.errors
                )
                / 60,
                "adjusted_max_mins": self.error_stream.adjusted_max_seconds() / 60,
                "offset": offset,
            }
            self.log_fun("deviation_tracker", deviation_tracker=log_data)

            if offset != self._offset:
                self.error_stream = ErrorStream(log_fun=self.log_fun)
                self._offset = offset

    def _adjusted_set_temperature(
        self, best_tempset: str, allowed_tempsets: List[str]
    ) -> str:
        new_temp = float(best_tempset) + self._offset

        for temp in allowed_tempsets:
            if new_temp == float(temp):
                return temp
        return best_tempset

    def remove_offset(
        self, current_tempset: StateTemperature, allowed_tempsets: List[Optional[str]]
    ) -> StateTemperature:

        if not is_numeric(allowed_tempsets + [current_tempset]):  # type: ignore
            return current_tempset

        for t in allowed_tempsets:
            if float(current_tempset) - self._offset == float(t):  # type: ignore
                return t
        return current_tempset

    def __repr__(self) -> str:
        return "DeviationTracker(errors={}, offset={})".format(
            self.error_stream.errors, self._offset
        )


class RedisDeviationTracker(DeviationTracker):
    def __init__(self, redis: Redis, device_id: str, quantity: str, log_fun) -> None:
        self.redis = redis
        self.key_arg = (device_id, quantity)
        super().__init__(log_fun=log_fun)

    async def load_state(self):
        try:
            data = await cache_util.get_deviation_tracker(
                redis=self.redis, key_arg=self.key_arg
            )
        except LookupError:
            self.log_fun("no deviation tracker state stored in redis")
        else:
            mode_offset = {
                mode: np.clip(offset, -MAX_OFFSET, MAX_OFFSET).item()
                for mode, offset in data["mode_offset"].items()
            }
            self._mode_offset = defaultdict(int, mode_offset)
            self._current_mode = data["current_mode"]
            self.error_stream.restart_start(data["start"])
            self.error_stream.extend(zip(data["errors"], data["seconds"]))

    async def store_state(self):
        data = {
            "mode_offset": self._mode_offset,
            "current_mode": self._current_mode,
            "start": self.error_stream.start,
            "errors": list(self.error_stream.errors),
            "seconds": list(self.error_stream.seconds),
        }
        await cache_util.set_deviation_tracker(
            redis=self.redis, key_arg=self.key_arg, value=data
        )


def seconds_to_target(seconds: Sequence[float], errors: Sequence[float]) -> float:
    a, b = np.polyfit(np.array(seconds) - seconds[-1], errors, 1)
    return _to_seconds(a, b)


def _to_seconds(a: float, b: float) -> float:
    if np.isclose(abs(b), 0.0):
        return 0.0
    if np.isclose(abs(a), 0.0):
        return CLIP_SECONDS
    return np.clip(-b / a, -CLIP_SECONDS, CLIP_SECONDS)


def is_numeric(temperatures: List) -> bool:
    try:
        for t in temperatures:
            float(t)  # type: ignore
    except (ValueError, TypeError):
        return False
    return True
