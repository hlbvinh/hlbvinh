import sys

import click
import uvloop
import yaml
from dateutil.parser import parse
from fabric.contrib.console import confirm

from skynet.sample import sample
from skynet.sample.climate_sample_store import (
    SAMPLE_COLLECTION,
    WATERMARK_COLLECTION,
    ClimateSampleStore,
)
from skynet.utils.async_util import run_sync
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")


@click.command()
@click.option(
    "--task", default="run", type=click.Choice(["run", "clear", "set_watermarks"])
)
@click.option("--config", default="config.yml")
@click.option("--mysql", default="prediction", help="mysql api config section")
@click.option("--mongo", default="production")
@click.option("--cassandra", default="production")
@click.option("--device_id", default=None, help="only process specified device")
@click.option("--watermark", default=None, help="Watermark to set.")
@click.option("--log_directory", default="log")
@click.option("--sample_collection", default=SAMPLE_COLLECTION)
@click.option("--watermark_collection", default=WATERMARK_COLLECTION)
def main(
    task,
    config,
    mysql,
    mongo,
    cassandra,
    device_id,
    watermark,
    log_directory,
    sample_collection,
    watermark_collection,
):
    uvloop.install()

    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config(
        "sample_generator", cnf=cnf, log_json=True, log_directory=log_directory
    )

    connections = get_connections(cnf, mysql=mysql, cassandra=cassandra, mongo=mongo)
    sample_store = ClimateSampleStore(
        connections.mongo,
        sample_collection=sample_collection,
        watermark_collection=watermark_collection,
    )

    if task == "run":
        try:
            run_sync(
                sample.generate_samples,
                connections.pool,
                connections.session,
                sample_store,
                device_id=device_id,
            )
        except KeyboardInterrupt:
            log.debug("KeyboardInterrupt, exiting")
            sys.exit(0)
        except Exception as exc:
            log.exception(exc)
            raise

    elif task == "clear":
        if confirm(f"clear {device_id or 'all'}?", default=False):
            key = {"device_id": device_id} if device_id is not None else {}
            sample_store.clear(key=key)
            sample_store.reset_watermarks(key=key)

    elif task == "set_watermarks":

        if watermark is None:
            raise ValueError("pass --watermark YYYY-MM-DD")

        watermark = parse(watermark)
        if confirm(f"set watermarks to {watermark}?", default=False):
            for wm in sample_store.get_watermarks():
                sample_store.set_watermark(wm["device_id"], watermark)
        print(f"watermarks set to {watermark}")


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
