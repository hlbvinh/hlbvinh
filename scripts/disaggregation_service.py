import asyncio
import logging
import sys

import click

import uvloop
import yaml
from skynet.disaggregation.service import DisaggregationService
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="production")
def main(config, mysql, cassandra):
    uvloop.install()
    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config(
        "disaggregation_service", cnf=cnf, loglevel=logging.INFO, log_json=True
    )

    DisaggregationService(
        ip=cnf["disaggregation_service"]["ip"],
        port=cnf["disaggregation_service"]["port"],
        connections=get_connections(cnf, mysql=mysql, cassandra=cassandra, redis=True),
    )
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        sys.exit()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
