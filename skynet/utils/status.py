import asyncio
import os
import sys
from datetime import timedelta
from typing import Any, Dict

from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor, RouterActor

from skynet.utils.log_util import get_logger

log = get_logger("__name__")

STATUS_TIMEOUT = 5


class DummyService(MicroService):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        super().__init__(None, log)

    def setup_resources(self):
        log.debug("Setting up resources")
        self.actor_ctx = ActorContext(log)
        self.actor_ctx.add_actor(
            "msger", RouterActor(ip=self.ip, port=self.port, log=log)
        )
        self.actor_ctx.add_actor("StatusActor", StatusActor())


class StatusActor(Actor):
    def __init__(self):
        super().__init__(log)

    @staticmethod
    def process(req):
        # params = req.params

        reply = {
            "method": "StatusActor",
            "data": {
                "status": 200,
                "executable": os.path.realpath(sys.argv[0]).split("/")[-1],
            },
            "status": 200,
            "context_id": req.context_id,
            "message_id": req.message_id,
        }

        return reply

    async def do_tell(self, msg):
        reply = self.process(msg)
        msger = self.context.find_actor("msger")
        msger.tell(reply)


async def status_request(
    ip: str,
    port: str,
    service: str,
    timeout: timedelta = timedelta(seconds=STATUS_TIMEOUT),
) -> Dict[str, Any]:
    actor = DealerActor(ip, port, log)
    req = AircomRequest.new("StatusActor", params={})

    try:
        status = await asyncio.wait_for(actor.ask(req), timeout.total_seconds())
    except asyncio.TimeoutError:
        status = {"status": 500}
    status.update({"service": service})

    return status
