import asyncio
import logging
import sys

import click
import yaml

import uvloop
from skynet.user.comfort_service import ComfortService
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections
from skynet.utils.storage import get_storage

log = get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="production")
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
def main(config, mysql, cassandra, storage):

    try:
        start_comfort_service(config, mysql, cassandra, storage)
    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        log.info("Shutting down")
        sys.exit()


def start_comfort_service(config, mysql, cassandra, storage):
    uvloop.install()
    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config(
        "comfort_service", cnf=cnf, loglevel=logging.INFO, log_json=True
    )

    model_store = get_storage(storage, **cnf["model_store"], directory="data/models")
    service = ComfortService(
        ip=cnf["comfort_service"]["ip"],
        port=cnf["comfort_service"]["port"],
        connections=get_connections(cnf, mysql=mysql, cassandra=cassandra, redis=True),
        storage=model_store,
    )
    log.info(f"created {service}")
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
