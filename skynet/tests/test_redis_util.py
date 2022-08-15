import pytest

from ..utils import redis_util


def test_get_redis():
    # test getting master from sentinels
    config = {
        "sentinels": {
            "hosts": [
                "pred1.ai.sg.tryambi.com",
                "pred2.ai.sg.tryambi.com",
                "train1.ai.sg.tryambi.com",
            ],
            "port": 26379,
            "service_name": "redis_ai_master",
            "password": "password",
        }
    }
    redis = redis_util.get_redis(config=config, use_sentinel=True)
    assert redis.connection_pool
    assert redis.connection_pool.service_name == "redis_ai_master"

    # test getting local redis
    redis = redis_util.get_redis()
    assert redis.connection_pool
    params = redis_util.get_config()
    assert redis.connection_pool.connection_kwargs["host"] == params["host"]


@pytest.mark.asyncio
async def test_get_lock_and_renew(rediscon):
    lock = redis_util.RenewLock(rediscon, "test", timeout=0.1)
    await lock.acquire(blocking_timeout=0)
    assert await lock.renew()
