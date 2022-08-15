from datetime import datetime, timedelta

import numpy as np
import pytest
from asynctest import mock

from ...user import comfort_model
from ...utils import cache_util
from ...utils.types import ComfortPrediction, Connections


@pytest.fixture
def timestamp():
    return datetime(2015, 1, 1)


@pytest.fixture
def feedback(device_id, user_id, timestamp):
    return dict(
        device_id=device_id, user_id=user_id, created_on=timestamp, feedback=2.0
    )


def test_comfort_model_store_and_load(
    comfort_features, trained_comfort_model, model_store
):
    m, X = trained_comfort_model, comfort_features
    yp = m.predict(X)
    model_store.save(comfort_model.ComfortModel.get_storage_key(), m)
    m2 = model_store.load(comfort_model.ComfortModel.get_storage_key())
    yp2 = m2.predict(X)
    assert np.allclose(yp, yp2)


@pytest.fixture
def sensors_connections(cassandra_session, pool, sensors_redis):
    return Connections(pool=pool, session=cassandra_session, redis=sensors_redis)


@pytest.mark.asyncio
async def test_predicted_feedback(
    trained_comfort_model, sensors_connections, device_id, user_id, timestamp
):
    m = trained_comfort_model

    fb = await m.get_adjusted_comfort_prediction(
        sensors_connections, device_id, user_id, timestamp
    )
    assert isinstance(fb["comfort"], float)
    assert fb["created_on"] == timestamp


@pytest.fixture
def adjustment_connections(cassandra_session, pool, adjustment_data_redis):
    return Connections(
        pool=pool, session=cassandra_session, redis=adjustment_data_redis
    )


@pytest.mark.asyncio
async def test_predicted_feedback_adjustment(
    trained_comfort_model,
    adjustment_connections,
    device_id,
    user_id,
    feedback,
    comfort_prediction,
):
    m = trained_comfort_model
    # test that within a short term interval of the feedback, the adjusted
    # prediction is still close to the feedback
    with mock.patch.object(m, "get_predicted_comfort", return_value=comfort_prediction):
        fb = await m.get_adjusted_comfort_prediction(
            adjustment_connections,
            device_id,
            user_id,
            feedback["created_on"] + timedelta(minutes=1),
        )
        assert fb["comfort"] == pytest.approx(feedback["feedback"], 0.01)


def test_comfort_model_train(raw_comfort_dataset):
    model = comfort_model.train(raw_comfort_dataset, bypass=True)
    assert isinstance(model, comfort_model.ComfortModel)


@pytest.mark.asyncio
async def test_get_predicted_comfort(
    trained_comfort_model, sensors_connections, device_id, user_id, timestamp
):
    m = trained_comfort_model

    prediction = await m.get_predicted_comfort(
        sensors_connections, device_id, user_id, timestamp
    )
    assert isinstance(prediction, float)


@pytest.fixture
async def sensors_redis(rediscon, device_id, sensors):
    await cache_util.set_sensors(redis=rediscon, key_arg=device_id, value=sensors)
    yield rediscon


@pytest.fixture
def comfort_prediction():
    return 2.5


@pytest.fixture
async def feedback_prediction_redis(
    rediscon, device_id, user_id, feedback, comfort_prediction
):
    feedback_prediction = ComfortPrediction(
        feedback["feedback"], comfort_prediction, 40, feedback["created_on"]
    )
    await cache_util.set_comfort_prediction(
        redis=rediscon, key_arg=(device_id, user_id), value=feedback_prediction
    )
    yield rediscon


@pytest.fixture
def adjustment_data_redis(
    sensors_redis, feedback_prediction_redis  # pylint:disable=unused-argument
):
    yield feedback_prediction_redis
