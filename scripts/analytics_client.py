from pprint import pprint
from time import time

import click
import yaml
from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from skynet.utils.async_util import run_sync
from skynet.utils.log_util import get_logger, init_logging_from_config

log = get_logger("skynet")


class ComfortModelClientActor(Actor):
    def __init__(self):
        super().__init__(log)

    async def get(self, msg):
        log.debug(f"sending {msg}")
        req = AircomRequest.from_dict(msg)
        msger = self.context.find_actor("msger")
        tic = time()
        response = await msger.ask(req)
        log.debug("took {:.5f} s".format(time() - tic))

        pprint(response)
        return response


class ComfortClientService(MicroService):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        super().__init__(None, log)

    def setup_resources(self):
        log.debug("Setting up resources")
        self.actor_ctx = ActorContext(log)

        self.actor_ctx.add_actor(
            "msger", DealerActor(ip=self.ip, port=self.port, log=log)
        )
        self.actor_ctx.add_actor("ComfortModelClientActor", ComfortModelClientActor())


@click.command()
@click.option("--config", default="config.yml")
@click.option("--user_id", default="1e5aa52a-9302-11e3-a724-0683ac059bd8")
@click.option("--device_id", default="05DBFF303830594143206932")
def main(config, user_id, device_id):

    msg = {
        "session_id": None,
        "params": {"device_id": device_id, "user_id": user_id},
        "method": "UserAnalyticsActor",
    }

    service = None
    try:
        with open(config) as f:
            cnf = yaml.safe_load(f)

        init_logging_from_config("analytics_client", cnf=cnf)

        service = ComfortClientService(
            ip=cnf["analytics_service"]["ip"], port=cnf["analytics_service"]["port"]
        )

        async def request():
            await service.actor_ctx.find_actor("ComfortModelClientActor").get(msg)

        run_sync(request)

    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        print("Shutting down")


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
