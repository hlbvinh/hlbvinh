import asyncio
from datetime import datetime, timedelta

import click
import yaml
from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from skynet.utils.async_util import add_callback
from skynet.utils.log_util import get_logger, init_logging_from_config

log = get_logger("skynet")
USER = "1e5aa52a-9302-11e3-a724-0683ac059bd8"


class ComfortModelClientActor(Actor):

    def __init__(self):
        super().__init__(log)
        add_callback(self.get_range)
        add_callback(self.get_range, "with_user")
        add_callback(self.get_single)
        add_callback(self.get_single, USER)

    async def get_range(self, user_id=None):
        data = {
            "start": datetime.utcnow() - timedelta(hours=2),
            "end": datetime.utcnow(),
            "device_id": "05DBFF303830594143206932",
        }

        if user_id is not None:
            data["user_id"] = user_id

        msg = {"method": "ComfortModelActor", "params": data, "session_id": None}
        log.debug(f"sending {msg}")
        req = AircomRequest.from_dict(msg)
        msger = self.context.find_actor("msger")
        response = await msger.ask(req)
        log.debug(f"received {response}")
        return response

    async def get_single(self, user_id=None):
        data = {"device_id": "05DBFF303830594143206932"}
        msg = {"method": "ComfortModelActor", "params": data, "session_id": None}
        if user_id is not None:
            data["user_id"] = user_id
        log.debug(f"sending {msg}")
        req = AircomRequest.from_dict(msg)
        msger = self.context.find_actor("msger")
        response = await msger.ask(req)
        log.debug(f"received {response}")
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
def main(config):
    service = None
    try:
        with open(config) as f:
            cnf = yaml.safe_load(f)

        init_logging_from_config("comfort_client", cnf=cnf)

        ip = cnf["comfort_service"]["ip"]
        port = cnf["comfort_service"]["port"]
        service = ComfortClientService(ip=ip, port=port)
        log.info(f"created {service}")
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        print("Shutting down")


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
