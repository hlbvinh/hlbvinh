import pytest

from ..utils import progressive_rollout


@pytest.fixture
def experiment(rediscon, device_id):
    return progressive_rollout.Experiment(rediscon, device_id, "test")


@pytest.mark.asyncio
async def test_default_experiment_is_level_0(experiment):
    await experiment.update()
    assert experiment.level == 0


@pytest.mark.asyncio
async def test_set_level(rediscon, experiment):
    await rediscon.hset(experiment.key, "level", 3)
    await experiment.update()
    assert experiment.level == 3


@pytest.mark.asyncio
async def test_set_coin_flip(rediscon, experiment):
    await rediscon.hset(experiment.key, experiment.device_id, 0.5)
    assert experiment.coin_flip != 0.5
    await experiment.update()
    assert experiment.coin_flip == 0.5


@pytest.mark.asyncio
async def test_whitelist_device_is_in(rediscon, experiment):
    await rediscon.hset(
        experiment.key, experiment.device_id, progressive_rollout.WHITELIST_COIN_FLIP
    )
    await rediscon.hset(
        experiment.key, "level", progressive_rollout.PROGRESSIVE_PROPORTIONS.index(0)
    )
    await experiment.update()
    assert experiment.is_in is True


@pytest.mark.asyncio
async def test_blacklist_device_is_not_in(rediscon, experiment):
    await rediscon.hset(
        experiment.key, experiment.device_id, progressive_rollout.BLACKLIST_COIN_FLIP
    )
    await rediscon.hset(
        experiment.key, "level", progressive_rollout.PROGRESSIVE_PROPORTIONS.index(1.0)
    )
    await experiment.update()
    assert experiment.is_in is False


@pytest.mark.asyncio
async def test_coin_flip_stored(rediscon, experiment):
    assert await rediscon.hget(experiment.key, experiment.device_id) is None
    await experiment.update()
    assert (
        float(await rediscon.hget(experiment.key, experiment.device_id))
        == experiment.coin_flip
    )


@pytest.mark.asyncio
async def test_coin_flip_is_not_changed(rediscon, experiment):
    await rediscon.hset(experiment.key, experiment.device_id, 0.5)
    await experiment.update()
    assert float(await rediscon.hget(experiment.key, experiment.device_id)) == 0.5


@pytest.mark.asyncio
async def test_progressive_rollout(rediscon, experiment):
    for level, propertion in enumerate(progressive_rollout.PROGRESSIVE_PROPORTIONS):
        await rediscon.hset(experiment.key, "level", level)
        await experiment.update()
        assert experiment.is_in == (experiment.coin_flip < propertion)
