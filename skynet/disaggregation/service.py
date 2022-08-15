from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.zmq_actor import RouterActor
from voluptuous import Invalid, Required, Schema

from ..utils import misc
from ..utils.log_util import get_logger
from ..utils.status import StatusActor
from .dummy import dummy_disaggregation

log = get_logger(__name__)


class DisaggregationService(MicroService):
    def __init__(self, ip, port, connections):
        self.ip = ip
        self.port = port
        self.connections = connections
        super().__init__(None, log)

    def setup_resources(self):
        self.actor_ctx = ActorContext(log)
        self.actor_ctx.add_actor(
            "msger", RouterActor(ip=self.ip, port=self.port, log=log)
        )
        self.actor_ctx.add_actor(
            "DisaggregationActor", DisaggregationActor(connections=self.connections)
        )
        self.actor_ctx.add_actor("StatusActor", StatusActor())


class DisaggregationActor(Actor):

    schema = Schema(
        {Required("location_id"): str, Required("start"): str, Required("end"): str}
    )

    def __init__(self, connections):
        self.connections = connections
        super().__init__(log)

    @misc.json_timeit(service="disaggregation_service", event="disaggregation")
    async def do_tell(self, req):
        params = req.params
        log.info(f"processing {params}")
        try:
            data, status = await self._process(**self.schema(params))
        except Invalid as exc:
            log.error(f"invalid params {params}")
            log.error(exc)
            data, status = {}, 400
        except Exception as exc:
            log.exception(exc)
            data, status = {}, 400
        self.context.find_actor("msger").tell(
            {
                "method": "DisaggregationActor",
                "data": data,
                "status": status,
                "context_id": req.context_id,
                "message_id": req.message_id,
            }
        )

    async def _process(self, location_id: str, start: str, end: str):
        return await dummy_disaggregation(self.connections, location_id, start, end)
