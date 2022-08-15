import asyncio
import logging
import sys

import click
import yaml

import uvloop
from skynet.user.analytics_service import AnalyticsService
from skynet.user.store import UserSampleStore
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="production")
@click.option("--mongo", default="production")
def main(config, mysql, cassandra, mongo):
    service = None
    uvloop.install()
    try:
        with open(config) as f:
            cnf = yaml.safe_load(f)

        init_logging_from_config(
            "analytics_service", cnf=cnf, loglevel=logging.INFO, log_json=True
        )

        connections = get_connections(
            cnf, mysql=mysql, mongo=mongo, cassandra=cassandra, redis=True
        )
        user_sample_store = UserSampleStore(connections.mongo)
        service = AnalyticsService(
            ip=cnf["analytics_service"]["ip"],
            port=cnf["analytics_service"]["port"],
            redis=connections.redis,
            pool=connections.pool,
            session=connections.session,
            user_sample_store=user_sample_store,
        )
        print(service)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        print("Shutting down")
        sys.exit()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
