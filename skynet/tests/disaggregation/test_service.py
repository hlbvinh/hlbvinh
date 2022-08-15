import pytest
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from ...disaggregation.service import DisaggregationService
from ...utils.log_util import get_logger
from ...utils.types import Connections

log = get_logger(__name__)


@pytest.fixture
def connections(rediscon, pool, cassandra_session):
    return Connections(redis=rediscon, pool=pool, session=cassandra_session)


@pytest.fixture
def micro_service(port, connections):
    service = DisaggregationService("127.0.0.1", port, connections)
    dealer = DealerActor(ip="127.0.0.1", port=port, log=log)

    return service, dealer


def get_request(
    location_id="3bd757cd-fc9f-407a-b0fd-f1de3f5a3dea",
    start="2019-09-23",
    end="2019-09-24",
):
    return AircomRequest.new(
        "DisaggregationActor", {"location_id": location_id, "start": start, "end": end}
    )


@pytest.fixture
def valid_request():
    return get_request()


@pytest.fixture
def invalid_request():
    return get_request(start=-1.5)


@pytest.mark.asyncio
async def test_service(micro_service, valid_request, invalid_request):
    _, dealer = micro_service
    for request in (valid_request, invalid_request):
        assert isinstance(await dealer.ask(request), dict)
