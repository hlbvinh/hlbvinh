from collections import defaultdict
from datetime import datetime, timedelta

import pytest

from ..utils import metrics
from ..utils.enums import Power
from ..utils.types import Prediction


class MockClimatePredictionStore(metrics.ClimatePredictionStore):
    def __init__(self):
        self.predictions = defaultdict(list)
        self.last_state = defaultdict(lambda: None)

    async def set_last_state(self, device_id, state):
        self.last_state[device_id] = state

    async def get_last_state(self, device_id):
        return self.last_state[device_id]

    async def set(self, key, predictions):
        self.predictions[key] = predictions

    async def get(self, key):
        return self.predictions[key]

    async def reset(self, key):
        self.predictions[key] = []


@pytest.fixture
def basic_state(state):
    return metrics.get_basic_state_from_appliance_state(state)


@pytest.fixture(params=["mock", "redis"])
def store(request, rediscon):
    if request.param == "mock":
        return MockClimatePredictionStore()
    return metrics.RedisClimatePredictionStore(rediscon)


@pytest.mark.asyncio
async def test_get_set_last_state(store, device_id, basic_state):
    assert await store.get_last_state(device_id) is None
    await store.set_last_state(device_id, basic_state)
    assert await store.get_last_state(device_id) == basic_state


@pytest.fixture
def prediction():
    return Prediction(
        temperature=16,
        humidity=60,
        humidex=18,
        created_on=datetime.utcnow(),
        horizon=timedelta(hours=3),
    )


@pytest.mark.asyncio
async def test_get_set_predictions(store, device_id, prediction):
    assert await store.get(device_id) == []
    await store.set(device_id, [prediction])
    assert await store.get(device_id) == [prediction]
    await store.reset(device_id)
    assert await store.get(device_id) == []


@pytest.fixture
def climate_prediction_store():
    return MockClimatePredictionStore()


@pytest.fixture
def climate_metric(climate_prediction_store, device_id, state):
    return metrics.ClimateMetric(device_id, state, climate_prediction_store)


@pytest.fixture
def current_condition(sensors):
    s = sensors.copy()
    s["humidex"] = 22
    return s


@pytest.mark.asyncio
async def test_with_no_predictions_no_error_computed(climate_metric, current_condition):
    assert await climate_metric.get_errors(current_condition) == []


@pytest.fixture
def old_enough_prediction(prediction):
    return prediction._replace(created_on=datetime.utcnow() - prediction.horizon)


@pytest.mark.asyncio
async def test_should_not_compute_error_when_horizon_is_not_reached(
    climate_metric, prediction, current_condition
):
    await climate_metric.add_prediction(prediction)
    assert await climate_metric.get_errors(current_condition) == []


@pytest.mark.asyncio
async def test_should_compute_error_for_old_enough_prediction(
    climate_metric, old_enough_prediction, current_condition
):
    await climate_metric.add_prediction(old_enough_prediction)
    assert await climate_metric.get_errors(current_condition)


@pytest.fixture
def too_old_prediction(prediction):
    return prediction._replace(
        created_on=datetime.utcnow() - prediction.horizon - timedelta(minutes=5)
    )


@pytest.mark.asyncio
async def test_should_not_compute_error_for_too_old_prediction(
    climate_metric, too_old_prediction, current_condition
):
    await climate_metric.add_prediction(too_old_prediction)
    assert await climate_metric.get_errors(current_condition) == []


@pytest.mark.asyncio
async def test_should_clean_too_old_prediction(
    climate_metric, too_old_prediction, current_condition
):
    await climate_metric.add_prediction(too_old_prediction)
    assert await climate_metric.store.get(climate_metric.key) != []
    await climate_metric.get_errors(current_condition)
    assert await climate_metric.store.get(climate_metric.key) == []


@pytest.fixture
async def climate_metric_with_old_enough_prediction(
    climate_metric, old_enough_prediction
):
    await climate_metric.add_prediction(old_enough_prediction)
    return climate_metric


@pytest.mark.asyncio
async def test_error_can_only_be_computed_once(
    climate_metric_with_old_enough_prediction, current_condition
):

    m = climate_metric_with_old_enough_prediction
    assert await m.get_errors(current_condition)
    assert not await m.get_errors(current_condition)


@pytest.fixture
async def climate_metric_with_predictions(
    climate_metric, old_enough_prediction, prediction
):
    await climate_metric.add_prediction(old_enough_prediction)
    await climate_metric.add_prediction(prediction)
    return climate_metric


@pytest.fixture
def future_condition(current_condition, prediction):
    c = current_condition.copy()
    c["created_on"] += prediction.horizon
    return c


@pytest.mark.asyncio
async def test_given_several_predictions_its_possible_to_succesively_get_errors(
    climate_metric_with_predictions, current_condition, future_condition
):

    assert await climate_metric_with_predictions.get_errors(current_condition)
    assert await climate_metric_with_predictions.get_errors(future_condition)


@pytest.fixture
def climate_metrics_with_different_device_ids(climate_prediction_store, state):
    return (
        metrics.ClimateMetric("device_a", state, climate_prediction_store),
        metrics.ClimateMetric("device_b", state, climate_prediction_store),
    )


@pytest.mark.asyncio
async def test_should_get_errors_only_if_the_device_is_the_same(
    climate_metrics_with_different_device_ids, old_enough_prediction, current_condition
):
    m1, m2 = climate_metrics_with_different_device_ids
    await m1.add_prediction(old_enough_prediction)
    assert not await m2.get_errors(current_condition)


@pytest.fixture
def states(state):
    s1, s2 = state, state.copy()
    s1["power"] = Power.OFF
    return s1, s2


@pytest.fixture
def climate_metrics_with_different_ac_settings(
    climate_prediction_store, device_id, states
):
    s1, s2 = states
    return (
        metrics.ClimateMetric(device_id, s1, climate_prediction_store),
        metrics.ClimateMetric(device_id, s2, climate_prediction_store),
    )


@pytest.mark.asyncio
async def test_should_get_errors_only_if_its_still_the_same_ac_setting(
    climate_metrics_with_different_ac_settings, old_enough_prediction, current_condition
):

    m1, m2 = climate_metrics_with_different_ac_settings
    await m1.add_prediction(old_enough_prediction)
    assert not await m2.get_errors(current_condition)


@pytest.mark.asyncio
async def test_should_not_get_remaining_errors_after_switching_back_to_previous_ac_setting(
    climate_metrics_with_different_ac_settings, old_enough_prediction, current_condition
):
    m1, m2 = climate_metrics_with_different_ac_settings
    await m1.add_prediction(old_enough_prediction)
    await m2.add_prediction(old_enough_prediction)
    assert await m2.get_errors(current_condition)
    assert not await m1.get_errors(current_condition)
