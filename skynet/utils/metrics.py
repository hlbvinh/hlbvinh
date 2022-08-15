import abc
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from aredis import StrictRedis as Redis

from . import cache_util, json
from ..prediction import climate_model, prediction_service
from ..sample import selection
from .enums import Power
from .thermo import humidex
from .types import ApplianceState, BasicState, Prediction, Sensors

CLIMATE_MODEL_METRIC_INTERVAL = timedelta(minutes=5)


async def log_climate_model_metric(
    prediction_client: DealerActor,
    history_features,
    state: ApplianceState,
    sensors: Sensors,
    device_id: str,
    redis: Redis,
    log,
) -> None:
    prediction = await get_climate_model_prediction(
        prediction_client, history_features, state
    )
    climate_metric = ClimateMetric(device_id, state, RedisClimatePredictionStore(redis))
    await climate_metric.add_prediction(prediction)
    errors = await climate_metric.get_errors(sensors)
    for error in errors:
        log("climate_metric", metric=error)


async def get_climate_model_prediction(
    prediction_client: DealerActor, history_features, state: ApplianceState
) -> Prediction:
    state = get_prediction_state_from_appliance_state(state)
    predictions = await prediction_service.get_climate_prediction(
        prediction_client, history_features, [state], quantity=None
    )
    return Prediction(
        created_on=datetime.utcnow(),
        horizon=selection.STATIC_INTERPOLATION,
        **{
            quantity: predictions[0][climate_model.QUANTITY_MAP[quantity]]
            for quantity in climate_model.QUANTITIES
        },
    )


def get_prediction_state_from_appliance_state(state: ApplianceState):
    return {key: state[key] for key in state if key in ["power", "mode", "temperature"]}


class ClimatePredictionStore(abc.ABC):
    @abc.abstractmethod
    async def set_last_state(self, device_id: str, state: BasicState) -> None:
        pass

    @abc.abstractmethod
    async def get_last_state(self, device_id: str) -> Optional[BasicState]:
        pass

    @abc.abstractmethod
    async def set(self, key: str, predictions: List[Prediction]) -> None:
        pass

    @abc.abstractmethod
    async def get(self, key: str) -> List[Prediction]:
        pass

    @abc.abstractmethod
    async def reset(self, key: str) -> None:
        pass

    async def add(self, key: str, prediction: Prediction) -> None:
        predictions = await self.get(key)
        predictions.append(prediction)
        await self.set(key, predictions)


class RedisClimatePredictionStore(ClimatePredictionStore):
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    @staticmethod
    def _last_state_key(device_id: str) -> str:
        return cache_util._key(f"last_basic_state:v2:{device_id}")

    async def set_last_state(self, device_id, state):
        await cache_util.set_redis(
            redis=self.redis,
            key_arg=device_id,
            value=state,
            key_fun=self._last_state_key,
            redis_encode=cache_util.namedtuple_encode,
        )

    async def get_last_state(self, device_id):
        return await cache_util.get_redis(
            redis=self.redis,
            key_arg=device_id,
            key_fun=self._last_state_key,
            redis_decode=cache_util.namedtuple_decode(BasicState),
            default_value=None,
        )

    @staticmethod
    def _predictions_key(key: str) -> str:
        return cache_util._key(f"predictions:{key}")

    @staticmethod
    def _predictions_encode(predictions: List[Prediction]) -> str:
        predictions_ = [prediction._asdict() for prediction in predictions]
        for prediction in predictions_:
            prediction["horizon"] = prediction["horizon"].seconds
        return json.dumps(predictions_)

    @staticmethod
    def _predictions_decode(msg: str) -> List[Prediction]:
        predictions_ = json.loads(msg)
        for prediction in predictions_:
            prediction["horizon"] = timedelta(seconds=prediction["horizon"])
        return [Prediction(**prediction) for prediction in predictions_]

    async def set(self, key, predictions):
        await cache_util.set_redis(
            redis=self.redis,
            key_arg=key,
            value=predictions,
            key_fun=self._predictions_key,
            redis_encode=self._predictions_encode,
        )

    async def get(self, key):
        return await cache_util.get_redis(
            redis=self.redis,
            key_arg=key,
            key_fun=self._predictions_key,
            redis_decode=self._predictions_decode,
            default_value=[],
        )

    async def reset(self, key):
        await self.redis.delete(self._predictions_key(key))


class ClimateMetric:
    def __init__(
        self, device_id: str, state: ApplianceState, store: ClimatePredictionStore
    ) -> None:
        self.device_id = device_id
        self.state = get_basic_state_from_appliance_state(state)
        self.store = store

    async def add_prediction(self, prediction: Prediction) -> None:
        if await self.has_state_changed:
            await self.store.reset(await self.last_key)
        await self.store.set_last_state(self.device_id, self.state)
        await self.store.add(self.key, prediction)

    @property
    async def has_state_changed(self) -> bool:
        last_state = await self.store.get_last_state(self.device_id)
        if last_state is None:
            return False
        return self.state != last_state

    async def get_errors(self, current_condition: Sensors):
        predictions = await self.store.get(self.key)
        await self.clean_store(predictions, current_condition)
        return [
            self.compute_error(current_condition, prediction)
            for prediction in predictions
            if self.can_compute_error(prediction, current_condition["created_on"])
        ]

    async def clean_store(self, predictions, current_condition):
        filtered_predictions = [
            prediction
            for prediction in predictions
            if self.cannot_compute_error_yet(
                prediction, current_condition["created_on"]
            )
        ]
        await self.store.set(self.key, filtered_predictions)

    def compute_error(
        self, current_condition: Sensors, prediction: Prediction
    ) -> Dict[str, Any]:
        d = dict(
            temperature_error=current_condition["temperature"] - prediction.temperature,
            humidity_error=current_condition["humidity"] - prediction.humidity,
            humidex_error=current_condition["humidex"] - prediction.humidex,
            humidex_sensors_error=current_condition["humidex"]
            - humidex(prediction.temperature, prediction.humidity),
        )
        for key in list(d):
            d["absolute_" + key] = abs(d[key])
        d["mode"] = self.state.mode
        d["temperature"] = self.state.temperature
        d["minute_horizon"] = prediction.horizon.seconds // 60
        d["created_on"] = prediction.created_on
        d["minutes_since_last_state"] = (
            datetime.utcnow() - self.state.created_on
        ).total_seconds() // 60

        return d

    @staticmethod
    def can_compute_error(prediction: Prediction, time: datetime) -> bool:
        time_since_prediction = time - prediction.created_on
        return (
            timedelta(0)
            < time_since_prediction - prediction.horizon
            < CLIMATE_MODEL_METRIC_INTERVAL
        )

    @staticmethod
    def cannot_compute_error_yet(prediction: Prediction, time: datetime) -> bool:
        time_since_prediction = time - prediction.created_on
        return time_since_prediction < prediction.horizon

    @property
    async def last_key(self) -> str:
        last_state = await self.store.get_last_state(self.device_id)
        assert last_state is not None
        # TODO: following a backend bug, some states had a mode=None, could
        # remove str when it is not an issue anymore
        return self.device_id + str(last_state.mode) + last_state.temperature

    @property
    def key(self) -> str:
        return self.device_id + str(self.state.mode) + self.state.temperature


def get_basic_state_from_appliance_state(state: ApplianceState) -> BasicState:
    mode = "off" if state["power"] == Power.OFF else state["mode"]
    if mode in ["off", "fan"]:
        temperature = str(24)
    else:
        temperature = str(state["temperature"])
    return BasicState(
        mode=mode, temperature=temperature, created_on=state["created_on"]
    )
