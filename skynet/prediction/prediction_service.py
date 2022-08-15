import abc
import time
from typing import Dict, List, Optional, Sequence

from ambi_utils.zmq_micro_service.actor import Actor, ActorContext
from ambi_utils.zmq_micro_service.micro_service import MicroService
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from voluptuous import Any, Invalid, Required, Schema

from ..user import comfort_model
from ..utils import data
from ..utils.async_util import PREDICTION_TIMEOUT, request_with_timeout
from ..utils.log_util import get_logger
from ..utils.status import StatusActor
from ..utils.storage import Loader, ModelReloadActor
from ..utils.types import ApplianceState, ModeSelection
from . import climate_model, mode_model, mode_model_util, predict

log = get_logger(__name__)

LOG_N_INITIAL = 10
LOG_INTERVAL = 10

MODELS = {
    "climate_model": {"climate_model": climate_model.ClimateModel.get_storage_key()},
    "mode_model": {"mode_model": mode_model.ModeModel.get_storage_key()},
    "comfort_model": {"comfort_model": comfort_model.ComfortModel.get_storage_key()},
}


class PredictionService(MicroService):
    def __init__(self, router_actor, prediction_actor):
        self.router_actor = router_actor
        self.prediction_actor = prediction_actor
        super().__init__(None, log)

    def setup_resources(self):
        log.debug("Setting up resources")
        self.actor_ctx = ActorContext(log)
        self.actor_ctx.add_actor("msger", self.router_actor)
        self.actor_ctx.add_actor(
            self.prediction_actor.method_name, self.prediction_actor
        )
        self.actor_ctx.add_actor("StatusActor", StatusActor())


class PredictionActor(Actor, metaclass=abc.ABCMeta):

    schema = Schema({})
    method_name = ""

    def __init__(self, models):
        self.models = models
        super().__init__(log)

    @abc.abstractmethod
    def _predict(self, params):
        """Prediction(s) for request.

        Returns
        -------
        dict:
            prediction list of floats
        """

    def _process(self, req):
        params = req.params
        try:
            self.schema(params)
        except Invalid as exc:
            log.error(exc)
            predicted_value = None
            status = 500
        else:
            predicted_value = self._predict(params)
            status = 200

        return {
            "data": {"prediction": predicted_value},
            "method": req.method,
            "status": status,
            "context_id": req.context_id,
            "message_id": req.message_id,
        }

    async def do_tell(self, req):
        self.counter = getattr(self, "counter", 0)
        tic = time.perf_counter()
        log_msg = {
            "context_id": req.context_id,
            "message_id": req.message_id,
            "event": self.method_name,
            "counter": self.counter,
        }
        data = self._process(req)
        msger = self.context.find_actor("msger")
        msger.tell(data)
        log_msg["response_time"] = 1000 * (time.perf_counter() - tic)
        if self.counter < LOG_N_INITIAL or self.counter % LOG_INTERVAL == 0:
            log.info("prediction", extra={"data": log_msg})
        self.counter += 1


class ClimateModelActor(PredictionActor):

    method_name = "climate_model_prediction"
    schema = Schema(
        {
            Required("history_features"): dict,
            Required("states"): list,
            Required("quantity"): Any(None, str),
        }
    )

    def _predict(self, params):
        predictor = predict.Predictor(self.models["climate_model"])
        y_pred = predictor.predict(**params)
        return list(y_pred)


class ModeModelActor(PredictionActor):

    method_name = "mode_model_prediction"
    schema = Schema({Required("mode_features"): dict, Required("mode_selection"): list})

    def _predict(self, params):
        model = self.models["mode_model"]
        mode_selection = params["mode_selection"]
        y_prob = model.predict_proba_one(params["mode_features"], mode_selection)
        return {"classes": mode_selection, "probas": y_prob}


class ComfortModelActor(PredictionActor):

    method_name = "comfort_model_prediction"
    schema = Schema({Required("features"): list})

    def _predict(self, params):
        comfort_model = self.models["comfort_model"]
        return comfort_model.predict(params["features"]).tolist()


class ClimateModelLoader(Loader):
    def __init__(self, storage, reload_seconds=climate_model.RELOAD_INTERVAL_SECONDS):
        super().__init__(storage, MODELS["climate_model"], reload_seconds)


class ModeModelLoader(Loader):
    def __init__(self, storage, reload_seconds=mode_model.RELOAD_INTERVAL_SECONDS):
        super().__init__(storage, MODELS["mode_model"], reload_seconds)


class ComfortModelLoader(Loader):
    def __init__(self, storage, reload_seconds=comfort_model.RELOAD_INTERVAL_SECONDS):
        super().__init__(storage, MODELS["comfort_model"], reload_seconds)


class ClimateModelActorReload(ModelReloadActor, ClimateModelActor):
    def __init__(self, loader):
        ClimateModelActor.__init__(self, {})
        ModelReloadActor.__init__(self, loader)


class ModeModelActorReload(ModelReloadActor, ModeModelActor):
    def __init__(self, loader):
        ModeModelActor.__init__(self, {})
        ModelReloadActor.__init__(self, loader)


class ComfortModelActorReload(ModelReloadActor, ComfortModelActor):
    def __init__(self, loader):
        ComfortModelActor.__init__(self, {})
        ModelReloadActor.__init__(self, loader)


def get_climate_model_request(
    history_features: Dict[str, Any],
    states: Sequence[ApplianceState],
    quantity: Optional[str],
) -> AircomRequest:
    params = {
        "history_features": history_features,
        "states": states,
        "quantity": quantity,
    }
    return AircomRequest.new(ClimateModelActor.method_name, params)


def get_mode_model_request(
    mode_features: Dict[str, Any], mode_selection: Sequence[str]
) -> AircomRequest:
    """AircomRequest for mode model prediction."""
    params = {"mode_features": mode_features, "mode_selection": mode_selection}
    return AircomRequest.new(ModeModelActor.method_name, params)


def get_comfort_model_request(features: List[Dict[str, Any]]):
    """AircomRequest for comfort model prediction."""
    params = {"features": features}
    return AircomRequest.new(ComfortModelActor.method_name, params)


async def call_mode_prediction_service(prediction_client, features, mode_selection):
    req = get_mode_model_request(features, mode_selection)
    resp = await request_with_timeout(PREDICTION_TIMEOUT, prediction_client, req)
    pred = resp["prediction"]
    mode_probas = pred["probas"]
    return mode_probas


async def get_mode_prediction(
    prediction_client,
    mode_selection: ModeSelection,
    scaled_target_delta: float,
    features,
    humidities,
    current_mode,
    log,
    prediction_service_client=call_mode_prediction_service,
):

    mode_hist = features["mode_hist"]
    power_hist = features["power_hist"]

    do_pred_modes = mode_model_util.using_mode_model(len(mode_selection))

    if do_pred_modes:
        mode_probas = await prediction_service_client(
            prediction_client, features, mode_selection
        )

        best_mode = mode_model_util.mode_model_adjustment_logic(
            mode_selection,
            mode_probas,
            mode_hist,
            power_hist,
            scaled_target_delta,
            features["humidity"],
            humidities,
        )

        mode_model_used = True
        highest_ranking_mode = data.argmax_dict(mode_probas, mode_model_util.MULTIMODES)
        best_mode_proba = mode_probas[best_mode]
        highest_ranking_proba = mode_probas[highest_ranking_mode]
    else:
        best_mode = mode_selection[0]
        mode_probas = {
            k: 1.0 if k == best_mode else 0.0 for k in mode_model_util.MODE_PROBAS_KEYS
        }
        mode_model_used = False
        highest_ranking_mode = best_mode
        best_mode_proba = 1.0
        highest_ranking_proba = 1.0

    log(
        "multi_mode_model",
        mode_selection=mode_selection,
        mode_features=features,
        mode_model_used=mode_model_used,
        mode_probas=mode_probas,
        highest_ranking_mode=highest_ranking_mode,
        best_mode=best_mode,
        best_mode_proba=best_mode_proba,
        highest_ranking_proba=highest_ranking_proba,
        mode_hist=mode_hist,
        scaled_target_delta=scaled_target_delta,
        current_humidity=features["humidity"],
        humidities=humidities,
        current_mode=current_mode,
    )

    return best_mode


async def get_climate_prediction(
    prediction_client, history_features, states, quantity
) -> List:
    req = get_climate_model_request(history_features, states, quantity)
    resp = await request_with_timeout(PREDICTION_TIMEOUT, prediction_client, req)
    predictions = resp["prediction"]
    return predictions
