import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.zmq_actor import RouterActor

from ..utils.log_util import get_logger
from ..utils.status import StatusActor
from ..utils.storage import Loader, ModelReloadActor, Storage
from ..utils.types import Connections
from .comfort_model import RELOAD_INTERVAL_SECONDS, ComfortModel

log = get_logger(__name__)


class ComfortService(MicroService):
    def __init__(
        self, ip: str, port: int, connections: Connections, storage: Storage
    ) -> None:
        self.ip = ip
        self.port = port
        self.connections = connections
        self.storage = storage
        super().__init__(None, log)

    def setup_resources(self):
        log.info("Setting up resources")
        self.actor_ctx = ActorContext(log)
        self.actor_ctx.add_actor(
            "msger", RouterActor(ip=self.ip, port=self.port, log=log)
        )
        self.actor_ctx.add_actor(
            "ComfortModelActor", ComfortModelActor(self.connections, self.storage)
        )
        self.actor_ctx.add_actor("StatusActor", StatusActor())


class ComfortModelActor(Actor, ModelReloadActor):
    def __init__(self, connections: Connections, storage: Storage) -> None:

        ModelReloadActor.__init__(
            self,
            Loader(
                storage,
                {"comfort_model": ComfortModel.get_storage_key()},
                RELOAD_INTERVAL_SECONDS,
            ),
        )
        self.connections = connections
        super().__init__(log)

    async def do_tell(self, req):

        log.info(
            "received request",
            extra={
                "data": {
                    "params": req.params,
                    "context_id": req.context_id,
                    "message_id": req.message_id,
                }
            },
        )
        tic = time.perf_counter()
        reply = await self.process(req)

        msger = self.context.find_actor("msger")

        log_msg = {"reply": reply, "response_time": 1000 * (time.perf_counter() - tic)}
        log.info("responding", extra={"data": log_msg})
        msger.tell(reply)

    async def process(self, req):
        params = req.params

        try:
            data = await process_single(
                self.model, self.connections, params["device_id"], params["user_id"]
            )
            if data is None:
                data = []
                status = 400
            else:
                status = 200
        except Exception as exc:
            log.exception(exc)
            data = []
            status = 400

        reply = {
            "method": "ComfortModelActor",
            "data": data,
            "status": status,
            "context_id": req.context_id,
            "message_id": req.message_id,
        }
        return reply

    @property
    def model(self):
        return self.models["comfort_model"]


async def process_single(
    model: ComfortModel, connections: Connections, device_id: str, user_id: str
) -> Optional[List[Dict[str, Any]]]:
    if is_input_valid(device_id, user_id):
        return [
            await model.get_adjusted_comfort_prediction(
                connections, device_id, user_id, datetime.utcnow()
            )
        ]
    return None


def is_input_valid(device_id, user_id):
    return all([isinstance(device_id, str), isinstance(user_id, str)])
