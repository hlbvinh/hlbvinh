from datetime import timedelta
from typing import Dict

import aredis
from aredis import StrictRedis as Redis
from aredis.exceptions import LockError, WatchError
from aredis.lock import Lock
from aredis.sentinel import MasterNotFoundError, Sentinel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .log_util import get_logger
from .misc import get_config_parameter

log = get_logger(__name__)

REDIS_TRIES = 5
REDIS_RETRY_DELAY_SECOND = 5
REDIS_CONNECTION_TIMEOUT = timedelta(seconds=5)


# during the failover we also got AttributeError and MasterNotFoundError due to
# race conditions in the implementation of the connection pool in aredis and
# redispy
# FIXME: redis_retry breaks the type hinting
redis_retry = retry(
    retry=retry_if_exception_type(
        (
            aredis.ConnectionError,
            MasterNotFoundError,
            AttributeError,
            aredis.TimeoutError,
        )
    ),
    stop=stop_after_attempt(REDIS_TRIES),
    wait=wait_fixed(REDIS_RETRY_DELAY_SECOND),
)


def get_redis(config=None, use_sentinel=True, **kwargs):
    if config is None:
        config = {}
    kw = {
        "decode_responses": True,
        "connect_timeout": REDIS_CONNECTION_TIMEOUT.total_seconds(),
        "stream_timeout": REDIS_CONNECTION_TIMEOUT.total_seconds(),
    }
    kw.update(kwargs)

    # we use simple redis (most likely elastic cache) over sentinels if both
    # are present
    if config.get("redis"):
        cnf = config["redis"]
        redis_master = Redis(**cnf, **kw)

    elif config.get("sentinels") and use_sentinel:
        cnf = config["sentinels"]
        sentinel = get_redis_sentinel(hosts=cnf["hosts"], port=cnf["port"])
        redis_master = sentinel.master_for(
            service_name=cnf["service_name"],
            redis_class=Redis,
            password=cnf["password"],
            **kw
        )

    elif config.get("forwarded_redis"):
        cnf = config["forwarded_redis"]
        redis_master = Redis(**cnf, **kw)
    else:
        redis_master = Redis(**kw)
    return redis_master


def get_redis_sentinel(hosts, port):
    sentinel = Sentinel([(h, port) for h in hosts])
    return sentinel


def get_config(**kwargs) -> Dict[str, str]:
    return {
        "password": get_config_parameter("password", kwargs, "REDIS_PASSWORD", None),
        "host": get_config_parameter("host", kwargs, "REDIS_HOST", "localhost"),
    }


class RenewLock(Lock):
    """we need to override the extend method of aredis version of the redis-py lock
    as it does not work well when the client is using `decode_responses`.
    In that case the `token` is a bytestring but fetching it returns a string.
    Comparisons between the two then fail"""

    async def renew(self):
        if self.token is None:
            raise LockError("Cannot renew an unlocked lock")
        if self.timeout is None:
            raise LockError("Cannot renew a lock with no timeout")
        return await self.do_renew()

    async def do_renew(self):
        pipe = await self.redis.pipeline()
        await pipe.watch(self.name)
        lock_value = await pipe.get(self.name)
        token = self.token.decode() if self.is_decode_responses else self.token
        if lock_value != token:
            raise LockError("Cannot renew a lock that's no longer owned")
        pipe.multi()
        await pipe.pexpire(self.name, int(self.timeout * 1000))
        try:
            response = await pipe.execute()
        except WatchError:
            # someone else acquired the lock
            raise LockError("Cannot renew a lock that's no longer owned")
        if not response[0]:
            # pexpire returns False if the key doesn't exist
            raise LockError("Cannot renew a lock that's no longer owned")
        return True

    @property
    def is_decode_responses(self):
        return self.redis.connection_pool.connection_kwargs["decode_responses"]

    @property
    def token(self):
        # should be `self.local.get()` for aredis 1.1.7
        return self.local.token
