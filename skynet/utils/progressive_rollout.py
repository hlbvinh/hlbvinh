from random import random
from typing import List, Optional

from aredis import StrictRedis as Redis

from .cache_util import _key

PROGRESSIVE_PROPORTIONS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
WHITELIST_COIN_FLIP = -1
BLACKLIST_COIN_FLIP = 2


class Experiment:
    def __init__(self, redis: Redis, device_id: str, experiment: str):
        self.redis = redis
        self.device_id = device_id
        self.experiment = experiment
        self.coin_flip = random()
        self.level: int
        self.is_in: bool

    async def update(self) -> None:
        await self.update_level()
        self.is_in = await self.get_membership()  # type: ignore
        if self.is_in is None:
            await self.set_flip()  # type: ignore
            self.is_in = self.is_in_from_flip()

    async def update_level(self) -> None:
        level = await self.redis.hget(self.key, "level")
        if level is None:
            self.level = 0
        else:
            self.level = int(level)

    async def get_membership(self) -> Optional[bool]:
        flip = await self.redis.hget(self.key, self.device_id)
        if flip is not None:
            self.coin_flip = float(flip)
            return self.is_in_from_flip()
        return None

    async def set_flip(self) -> None:
        await self.redis.hset(self.key, self.device_id, self.coin_flip)

    def is_in_from_flip(self) -> bool:
        return self.coin_flip < PROGRESSIVE_PROPORTIONS[self.level]

    @property
    def key(self):
        return _key(self.experiment)


def is_in(experiment: str, experiments: List[Experiment]) -> bool:
    for e in experiments:
        if e.experiment == experiment:
            return e.is_in
    return False
