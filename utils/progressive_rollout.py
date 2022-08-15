from datetime import date
from typing import Dict, List

import click
import pandas as pd
import yaml

from skynet.utils.async_util import run_sync
from skynet.utils.cache_util import _key
from skynet.utils.progressive_rollout import (
    BLACKLIST_COIN_FLIP,
    PROGRESSIVE_PROPORTIONS,
    WHITELIST_COIN_FLIP,
)
from skynet.utils.script_utils import get_connections

HARDCODED_DEVICES: Dict[str, List[Dict[str, List[str]]]] = {
    "maintain_trend": {
        "whitelist": """
    6eca0a82-a53e-40ac-a74f-7c02a330f7ed
    a71a24ff-50c2-4ba4-b577-42148323517f
    e47d0615-7b01-4cf8-86bf-386284ab396b
    """.split(),
        "blacklist": """""".split(),
    },
    "maintain_threshold": {
        "whitelist": """
    7593beaf-be9f-47fa-aa52-fe56565e5cc4
    """.split(),
        "blacklist": """""".split(),
    },
    "average_humidity": {
        "whitelist": """
    45cfb0f1-cc5d-4fc3-9621-02dcef35c404
    72dfee16-5bad-4577-90f8-6994e9b5a770
    abf9d1f6-f539-4364-9136-1128fed6b25c
    36da7efc-e83d-4c30-a2fc-f67a3aa4747d
    962415b7-8861-4462-a4ad-4ad339dfb5fa
    7ef68de3-0f62-4805-81d2-6f033b942721
    f38ebb14-f9a7-4074-8c6d-f6e36e6eb047
    6feedcf5-c584-4549-9a93-0dcc456a5d0d
    """.split(),
        "blacklist": """""".split(),
    },
}


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="viewer")
@click.option("--experiment", required=True)
@click.option(
    "--task",
    required=True,
    default="run",
    type=click.Choice(
        [
            "get_level",
            "set_level",
            "set_hardcoded",
            "get_devices",
            "get_users",
            "delete",
        ]
    ),
)
@click.option("--level", type=int)
def main(config, mysql, experiment, task, level):

    with open(config) as filehandle:
        cnf = yaml.safe_load(filehandle)

    connections = get_connections(cnf, redis=True, mysql=mysql)

    run_sync(is_master, connections.redis)
    if task == "get_level":
        run_sync(get_level, connections.redis, experiment)
    elif task == "set_level":
        run_sync(set_level, connections.redis, experiment, level)
    elif task == "set_hardcoded":
        run_sync(set_hardcoded, connections.redis, experiment)
    elif task == "get_devices":
        run_sync(get_devices, connections.redis, experiment)
    elif task == "get_users":
        run_sync(get_users, connections, experiment)
    elif task == "delete":
        run_sync(delete, connections.redis, experiment)


async def is_master(redis):
    assert (await redis.info())["role"] == "master"


async def set_level(redis, experiment, level):
    await redis.hset(_key(experiment), "level", level)


async def get_level(redis, experiment):
    print(await redis.hget(_key(experiment), "level"))


async def set_hardcoded(redis, experiment):
    hardcoded = {
        **{
            device_id: WHITELIST_COIN_FLIP
            for device_id in HARDCODED_DEVICES[experiment]["whitelist"]
        },
        **{
            device_id: BLACKLIST_COIN_FLIP
            for device_id in HARDCODED_DEVICES[experiment]["blacklist"]
        },
    }
    await redis.hmset(_key(experiment), hardcoded)


async def get_devices(redis, experiment):
    device_ids, _ = await _get_devices(redis, experiment)
    for device_id in device_ids:
        print(device_id)


async def get_users(connections, experiment):
    device_ids, level = await _get_devices(connections.redis, experiment)
    formatted_device_ids = "('" + "', '".join(device_ids) + "')"

    sql = (
        """
    SELECT user_id FROM UserDeviceList WHERE device_id IN {device_ids}
    """.format(
            device_ids=formatted_device_ids
        ),
    )
    pd.DataFrame(await connections.pool.execute(*sql)).to_csv(
        experiment + "_" + str(level) + "_" + str(date.today()) + ".csv"
    )


async def delete(redis, experiment):
    await redis.delete(_key(experiment))


async def _get_devices(redis, experiment):
    result = await redis.hgetall(_key(experiment))
    level = result.pop("level", 0)
    return (
        {
            device_id
            for device_id, coin_flip in result.items()
            if float(coin_flip) < PROGRESSIVE_PROPORTIONS[int(level)]
        },
        level,
    )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
