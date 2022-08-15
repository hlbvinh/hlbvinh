import click
import yaml
from fabric.contrib.console import confirm

import uvloop
from skynet.user import sample
from skynet.user.store import SAMPLE_COLLECTION, WATERMARK_COLLECTION, UserSampleStore
from skynet.utils.async_util import run_sync
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")


def clear(sample_store, device_id):
    key = {}
    if device_id is not None:
        key["device_id"] = device_id
    if not key:
        text = "all"
    else:
        text = key
    if confirm(f"clear {text}", default=False):
        sample_store.clear(key)
        sample_store.reset_watermarks(key)
    else:
        if confirm(f"reset watermarks for {text}", default=False):
            sample_store.reset_watermarks(key)


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="prediction")
@click.option("--mongo", default="production")
@click.option("--cassandra", default="production")
@click.option("--task", default="fetch", type=click.Choice(["fetch", "clear"]))
@click.option("--device_id", default=None)
@click.option("--ignore_watermark", default=False, is_flag=True)
@click.option("--log_directory", default="log")
@click.option("--sample_collection", default=SAMPLE_COLLECTION)
@click.option("--watermark_collection", default=WATERMARK_COLLECTION)
def main(
    config,
    mysql,
    mongo,
    cassandra,
    task,
    device_id,
    ignore_watermark,
    log_directory,
    sample_collection,
    watermark_collection,
):
    uvloop.install()

    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config(
        "user_sample_generator", cnf=cnf, log_directory=log_directory
    )

    connections = get_connections(cnf, mysql=mysql, mongo=mongo, cassandra=cassandra)

    sample_store = UserSampleStore(
        connections.mongo,
        sample_collection=sample_collection,
        watermark_collection=watermark_collection,
    )

    if task == "fetch":
        log.debug("fetching")
        try:
            run_sync(
                sample.make_all_feedback_samples,
                sample_store,
                connections.pool,
                connections.session,
                ignore_watermark=ignore_watermark,
                device_id=device_id,
            )

        except Exception as e:
            log.exception(e)
            raise

    elif task == "clear":
        clear(sample_store, device_id)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
