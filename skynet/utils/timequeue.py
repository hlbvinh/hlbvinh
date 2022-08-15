from datetime import datetime, timedelta
from typing import List, Optional


class TimeQueue:
    """Time based queue only keeping track of items (with associated timestamp)
    that are within the max_size"""

    def __init__(
        self,
        max_size: timedelta,
        items: Optional[List[float]] = None,
        timestamps: Optional[List[datetime]] = None,
    ):
        self.max_size = max_size
        self._items = items or []
        self._timestamps = timestamps or []

    def append(self, item: float, timestamp: Optional[datetime] = None):
        self._items.append(item)
        self._timestamps.append(timestamp or datetime.utcnow())
        self._filter()

    @property
    def has_enough_data(self) -> bool:
        return not self._is_within_max_size(self._timestamps[0])

    @property
    def items(self):
        return [
            e
            for e, t in zip(self._items, self._timestamps)
            if self._is_within_max_size(t)
        ]

    @property
    def timestamps(self):
        return [t for t in self._timestamps if self._is_within_max_size(t)]

    def _filter(self):
        # in order to know whether the queue is full we need to know if there
        # exists an element before the max_size

        # index of the element just before the max_size.
        index = max(len(self._timestamps) - len(self.timestamps) - 1, 0)
        self._items = self._items[index:]
        self._timestamps = self._timestamps[index:]

    def _is_within_max_size(self, timestamp):
        return (self._timestamps[-1] - timestamp) < self.max_size
