import asyncio
import logging
import sys
from time import sleep

import click
import uvloop
import yaml

from skynet.control.control import Control
from skynet.utils import events, log_util
from skynet.utils.script_utils import get_connections
from skynet.utils.status import DummyService

log = log_util.get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--loglevel", default="INFO")
def main(config, loglevel):
    uvloop.install()

    with open(config) as f:
        cnf = yaml.safe_load(f)

    if "sentry" not in cnf:
        log.error("Sentry Handler not configured")
        sentry_dsn = None
    else:
        sentry_dsn = cnf["sentry"]["dsn"]

    log_util.init_logging_from_config(
        "control",
        cnf=cnf,
        backup_count=15,
        loglevel=getattr(logging, loglevel.upper()),
        log_json=True,
        sentry_dsn=sentry_dsn,
    )
    while True:
        try:
            connections = get_connections(cnf, redis=True)

            listener = events.get_event_listener(**cnf["event_service"])

            control_cnf = cnf["control_service"]
            service = DummyService(**control_cnf)
            log.debug(f"STARTING {service}")

            Control(redis=connections.redis, listener=listener).start()

            asyncio.get_event_loop().run_forever()

        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            log.error(e)
            log.exception(e)
            sleep(5)
        finally:
            try:
                del control  # FIXME: no such variable
            except NameError:
                pass


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
