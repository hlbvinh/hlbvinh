from datetime import datetime

import pytest
import pytz
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from asynctest import mock

from ...control import comfort, managed_manual
from ...prediction import climate_model
from ...sample import feature_cache
from ...utils import cache_util
from ...utils.types import Connections


@pytest.fixture
def target():
    return {
        "quantity": "climate",
        "created_on": datetime.utcnow(),
        "value": None,
        "origin": "user",
    }


@pytest.fixture
def prediction_clients():
    prediction_clients = {}
    models = ["mode_model", "climate_model", "comfort_model"]

    async def climate_comfort_ask(request):
        if request.params["quantity"] is None:
            return {"prediction": [[1] * len(climate_model.QUANTITIES)]}
        return {"prediction": 1}

    async def comfort_ask(request):
        return {"prediction": [1] * len(request.params["features"])}

    for model in models:
        m = mock.Mock(spec=DealerActor)
        if model == "mode_model":
            m.ask = mock.CoroutineMock(return_value={"prediction": mock.Mock()})
        elif model == "climate_model":
            m.ask = climate_comfort_ask
        elif model == "comfort_model":
            m.ask = comfort_ask
        prediction_clients[model] = m
    return prediction_clients


@pytest.fixture
def timezone():
    return pytz.timezone("Asia/Hong_Kong")


@pytest.fixture
async def controller_redis(rediscon, device_id, sensors, weather, state):
    # insert data for feature_data
    await cache_util.set_sensors(rediscon, device_id, sensors)
    await cache_util.set_weather_redis(rediscon, device_id, [weather])
    await cache_util.set_appliance_state(redis=rediscon, key_arg=device_id, value=state)
    yield rediscon


@pytest.fixture
def default_db_service_msger(get_db_service_msger):
    return get_db_service_msger(code=200, response={})


@pytest.fixture
def connections(pool, controller_redis, cassandra_session, default_db_service_msger):
    return Connections(
        pool=pool,
        redis=controller_redis,
        session=cassandra_session,
        db_service_msger=default_db_service_msger,
    )


@pytest.fixture
async def feature_data(connections, device_id):
    feature_data = feature_cache.RedisFeatureData(
        connections=connections, device_id=device_id
    )
    await feature_data.load_state()
    return feature_data


@pytest.fixture
def default_comfort(
    device_id, timezone, prediction_clients, feedback, feature_data, connections
):
    return comfort.Comfort(
        device_id,
        timezone,
        prediction_clients,
        feature_data,
        nearby_users={feedback["user_id"]},
        latest_feedbacks=[feedback],
        connections=connections,
    )


@pytest.fixture
def managed_manual_(device_id, connections, state):
    return managed_manual.ManagedManual(device_id, state, connections)
