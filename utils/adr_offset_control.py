from datetime import datetime

import click
import pandas as pd
import sobol_seq
import yaml

from skynet.utils.async_util import run_sync
from skynet.utils.cache_util import _key
from skynet.utils.redis_util import get_redis
from skynet.utils.script_utils import get_connections

# for this year adr trial we want to be able to better learn the impact of
# offsets on the overall energy saving, in order to do so every user will be
# assigned a different offset during an adr event. That offset will be within
# the range for that adr event, we'll try to spread the offsets evenly during
# the event and to be optimal rather than using uniformly random sampling we
# are using quasi random sequence (sobol sequence )to make sure that even if we
# have few users joining we are still able to most of the range due to low
# discrepancy.


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--task", required=True, type=click.Choice(["get_range", "set_range", "get_offset"])
)
@click.option("--range", type=float, nargs=2)
@click.option("--local", is_flag=True, default=False)
def main(config, task, range, local):

    with open(config) as filehandle:
        cnf = yaml.safe_load(filehandle)

    redis = get_redis()
    if not local:
        connections = get_connections(cnf, redis=True)
        redis = connections.redis

    run_sync(is_master, redis)
    if task == "get_range":
        run_sync(get_range, redis)

    elif task == "set_range":
        run_sync(set_range, redis, *range)

    elif task == "get_offset":
        run_sync(get_offset, redis)


async def is_master(redis):
    assert (await redis.info())["role"] == "master"


async def set_range(redis, a, b):
    key = _key("adr:range")
    key2 = _key("adr:sobol")
    await redis.delete(key, key2)
    await redis.rpush(key, a, b)
    await redis.rpush(key2, *get_sobol(a, b))


async def get_range(redis):
    print("range ", await redis.lrange(_key("adr:range"), 0, -1))
    print("sobol ", await redis.lrange(_key("adr:sobol"), 0, -1))


async def get_offset(redis):
    # from skynet.control.target import get_adr_offset
    # await get_adr_offset(redis, "a")
    # await get_adr_offset(redis, "a")
    # await get_adr_offset(redis, "a")
    # await get_adr_offset(redis, "b")
    # await get_adr_offset(redis, "c")
    # await get_adr_offset(redis, "d")
    # await get_adr_offset(redis, "e")
    # await get_adr_offset(redis, "f")

    offsets = await redis.hgetall(_key("adr:offset"))
    df = pd.DataFrame.from_dict(data=offsets, orient="index")
    print(df)
    csv_file = "offset_" + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + ".csv"
    df.to_csv(csv_file, header=False)


def get_sobol(a, b, n=500):
    seq = sobol_seq.i4_sobol_generate(1, n) * (b - a) + a
    return list(seq.flatten())


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
