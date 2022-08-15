import asyncio

import click
import tornado.web
import yaml
from tornado import gen

import uvloop
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.status import status_request

log = get_logger("skynet")

SUPERVISOR_SERVICE_NAME_MAP = {
    "analytics_service": "analytics_service",
    "comfort_service": "comfort_service",
    "prediction_service_climate": "climate_model",
    "prediction_service_comfort": "comfort_model",
    "prediction_service_mode": "mode_model",
    "prediction_service_user": "user_model",
    "control": "control_service",
    "control_worker": "control_worker_service",
}


class ServiceHandler(tornado.web.RequestHandler):
    def initialize(self, services):
        self.services = services

    async def get(self):
        requested = set(self.get_arguments("service"))
        available = set(self.services)

        not_monitored = requested - available
        if not_monitored:
            log.info(f"{not_monitored} not monitored")
        to_monitor = list((requested & available) or available)
        log.info(f"monitor status of {to_monitor}")

        futs = [
            gen.convert_yielded(
                status_request(
                    ip=self.services[service]["ip"],
                    port=self.services[service]["port"],
                    service=self.services[service]["service"],
                )
            )
            for service in to_monitor
        ]
        response = {}

        wait_iterator = gen.WaitIterator(*futs)
        async for result in wait_iterator:
            response[to_monitor[wait_iterator.current_index]] = result

        log.info(response)
        self.write(response)


def get_services_from_cnf(cnf):
    services = {}

    for service in cnf["prediction_services"]:
        services[service] = cnf["prediction_services"][service]
    for service in [
        "analytics_service",
        "comfort_service",
        "control_service",
        "control_worker_service",
    ]:
        services[service] = cnf[service]
    for service_name, service in services.items():
        service["service"] = service_name

    return {
        supervisor: services[service]
        for supervisor, service in SUPERVISOR_SERVICE_NAME_MAP.items()
    }


@click.command()
@click.option("--config", default="config.yml")
def main(config):
    uvloop.install()

    cnf = yaml.safe_load(open(config))

    init_logging_from_config("status_monitor", cnf=cnf)

    services = get_services_from_cnf(cnf)

    app = tornado.web.Application([(r"/?", ServiceHandler, dict(services=services))])
    app.listen(cnf["status_monitor"]["port"])
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
