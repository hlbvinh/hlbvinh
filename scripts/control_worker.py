import asyncio
import logging
import sys
from time import sleep

import click
import uvloop
import yaml
import zmq
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from skynet.control.control import ControlWorker
from skynet.utils import log_util
from skynet.utils.script_utils import get_connections
from skynet.utils.status import DummyService

log = log_util.get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mysql", default="prediction")
@click.option("--cassandra", default="production")
@click.option("--db_service", default="db_service")
@click.option("--testing", is_flag=True, default=False)
@click.option("--loglevel", default="INFO")
def main(config, mysql, cassandra, db_service, testing, loglevel):
    uvloop.install()

    with open(config) as f:
        cnf = yaml.safe_load(f)

    if "sentry" not in cnf:
        log.error("Sentry Handler not configured")
        sentry_dsn = None
    else:
        sentry_dsn = cnf["sentry"]["dsn"]

    log_util.init_logging_from_config(
        "control_worker",
        cnf=cnf,
        backup_count=15,
        loglevel=getattr(logging, loglevel.upper()),
        log_json=True,
        sentry_dsn=sentry_dsn,
    )
    while True:
        try:
            connections = get_connections(
                cnf,
                mysql=mysql,
                cassandra=cassandra,
                redis=True,
                db_service_msger=db_service,
            )

            prediction_clients = {}
            for model in ["climate_model", "mode_model", "comfort_model"]:
                service_cnf = cnf["prediction_services"][model]
                dealer_actor = DealerActor(log=log, **service_cnf)

                # Set the high water mark to 100 messages
                # such as to not queue up too many requests when
                # a prediction service goes offline for a while.

                # pylint:disable=no-member
                dealer_actor.sock.setsockopt(zmq.SNDHWM, 100)
                # pylint:enable=no-member

                prediction_clients[model] = dealer_actor

            control_cnf = cnf["control_worker_service"]
            service = DummyService(**control_cnf)
            log.debug(f"STARTING {service}")

            ControlWorker(
                connections=connections,
                prediction_clients=prediction_clients,
                testing=testing,
            ).start()

            asyncio.get_event_loop().run_forever()

        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(e)
            log.error(e)
            log.exception(e)
            sleep(5)
        finally:
            try:
                del worker
            except NameError:
                pass


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
