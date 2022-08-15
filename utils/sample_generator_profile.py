from time import sleep

import click

# not needed here, but to avoid strange relative import error with pandas
# import pandas as pd
import yaml
from fabric.contrib.console import confirm

import yappi
from skynet.sample import sample
from skynet.sample.climate_sample_store import ClimateSampleStore
from skynet.utils.async_util import run_sync
from skynet.utils.database.cassandra import CassandraSession
from skynet.utils.database.dbconnection import get_pool
from skynet.utils.log_util import get_logger, init_logging
from skynet.utils.mongo import Client

log = get_logger("skynet")


@click.command()
@click.option(
    "--task", required=True, default="run", type=click.Choice(["run", "clear"])
)
@click.option("--config", default="config.yml")
@click.option("--mysql_config", default="prediction", help="mysql api config section")
@click.option("--cassandra_config", default="production")
@click.option("--mongo_config", default="production")
@click.option("--device_id", default=None, help="only process specified device")
def main(task, config, mysql_config, cassandra_config, mongo_config, device_id):

    cnf = yaml.safe_load(open(config))
    pool = get_pool(**cnf[mysql_config])
    mongo_client = Client(**cnf["mongo"][mongo_config])
    sample_store = ClimateSampleStore(mongo_client)
    session = CassandraSession(**cnf["cassandra"][cassandra_config])
    init_logging("sample_generator")

    yappi.start(builtins=True)
    if task == "run":
        while True:
            try:
                run_sync(
                    sample.generate_samples,
                    pool,
                    session,
                    sample_store,
                    device_id=device_id,
                )
            except KeyboardInterrupt:
                log.debug("KeyboardInterrupt, exiting")
                yappi.stop()
                stats = yappi.get_func_stats()
                stats.save("callgrind.out", type="callgrind")
                exit(0)
            except Exception as exc:
                log.exception(exc)

            log.info("all done, waiting...")
            sleep(3600)

    elif task == "clear":
        if confirm("clear {}?".format(device_id or "all"), default=False):
            key = {"device_id": device_id} if device_id is not None else {}
            sample_store.clear(key=key)
            sample_store.reset_watermarks(key=key)


if __name__ == "__main__":
    main()
