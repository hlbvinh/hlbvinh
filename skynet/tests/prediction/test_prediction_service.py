import asyncio
from datetime import timedelta
from math import isnan

import pytest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor, RouterActor
from asynctest import mock

from ...prediction import mode_model, mode_model_util, predict, prediction_service
from ...utils.async_util import request_with_timeout
from ...utils.enums import Power
from ...utils.log_util import get_logger

log = get_logger(__name__)


def dict_to_list(dictionary, keys):
    return [dictionary[k] for k in keys if not isnan(dictionary[k])]


def _check_request(actor, request_fun, params):
    pred = actor._predict(params)
    req = request_fun(**params)
    resp = actor._process(req)
    if isinstance(pred, dict):
        modes = sorted(mode_model_util.MULTIMODES)
        assert dict_to_list(pred["probas"], modes) == dict_to_list(
            resp["data"]["prediction"]["probas"], modes
        )
        assert pred["classes"] == resp["data"]["prediction"]["classes"]
    else:
        assert pred == resp["data"]["prediction"]
    assert resp["context_id"] == req.context_id
    assert resp["message_id"] == req.message_id
    assert resp["status"] == 200


@pytest.fixture
def climate_params():
    states = predict.generate_on_signals()
    return {
        "history_features": {"appliance_id": "0", "temperature_set_last": 2},
        "quantity": "temperature",
        "states": states,
    }


@pytest.fixture
def climate_request(climate_params):
    return prediction_service.get_climate_model_request(**climate_params)


def test_climate_model_actor(trained_climate_model, climate_params):
    actor = prediction_service.ClimateModelActor(
        models={"climate_model": trained_climate_model}
    )
    params = climate_params
    pred = actor._predict(params)
    assert isinstance(pred, list)
    for item in pred:
        assert isinstance(item, float)

    _check_request(actor, prediction_service.get_climate_model_request, params)

    # test bad requests:
    for k in ["history_features", "states"]:
        new_params = params.copy()
        new_params[k] = None
        req = prediction_service.get_climate_model_request(**new_params)
        resp = actor._process(req)
        assert resp["status"] == 500


@pytest.fixture(params=[mode_model_util.MULTIMODES])
def mode_params(request):
    features = {
        "appliance_id": "app",
        "humidex": 25.0,
        "humidity": 50,
        "temperature": 25,
        "power_hist": Power.OFF,
        "mode_hist": "off",
        "temperature_out": 20.0,
        "humidity_out": 85,
        "humidex_out": 30,
        "temperature_out_mean_day": 21.0,
    }

    mode_features = mode_model.make_features(
        history_features=features, target_quantity="humidex", target_value=1.0
    )
    return {"mode_features": mode_features, "mode_selection": list(request.param)}


@pytest.fixture
def mode_request(mode_params):
    return prediction_service.get_mode_model_request(**mode_params)


def test_mode_model_actor(trained_mode_model, mode_params):
    models = {"mode_model": trained_mode_model}
    actor = prediction_service.ModeModelActor(models)
    params = mode_params
    pred = actor._predict(params)
    classes = pred["classes"]
    probas = mode_model_util.select_probas(
        pred["probas"], mode_model_util.MULTIMODES
    ).values()

    assert classes
    assert probas
    assert sorted(pred["probas"]) == sorted(
        mode_model_util.MULTIMODES + ["first_layer_cool", "first_layer_heat"]
    )
    for class_, proba in zip(classes, probas):
        assert isinstance(class_, str)
        assert isinstance(proba, float)

    _check_request(actor, prediction_service.get_mode_model_request, mode_params)


def _get_comfort_params(features):
    return dict(features=features)


@pytest.fixture
def comfort_params():
    # test minimial input
    features = {"device_id": "a", "user_id": "b"}
    return _get_comfort_params([features])


@pytest.fixture
def comfort_request(comfort_params):
    return prediction_service.get_comfort_model_request(**comfort_params)


def test_comfort_model_actor(trained_comfort_model, comfort_params):
    models = {"comfort_model": trained_comfort_model}
    actor = prediction_service.ComfortModelActor(models)

    # minimal params
    features = {"device_id": "a", "user_id": "b"}
    params = _get_comfort_params([features])

    comfort = actor._predict(params)
    assert isinstance(comfort[0], float)

    # check with valid parameters
    params = comfort_params
    _check_request(actor, prediction_service.get_comfort_model_request, params)


@pytest.fixture(params=["mode", "climate", "comfort"])
async def service(
    request,
    port,
    model_store,
    trained_comfort_model,
    trained_mode_model,
    trained_climate_model,
    climate_request,
    mode_request,
    comfort_request,
):

    dealer = DealerActor(ip="127.0.0.1", port=port, log=log)
    router = RouterActor(ip="127.0.0.1", port=port, log=log)

    if request.param == "climate":
        name = "climate_model"
        models = {"climate_model": trained_climate_model}
        actor_class = prediction_service.ClimateModelActorReload
        loader_class = prediction_service.ClimateModelLoader
        request = climate_request

    elif request.param == "mode":
        name = "mode_model"
        models = {"mode_model": trained_mode_model}
        actor_class = prediction_service.ModeModelActorReload
        loader_class = prediction_service.ModeModelLoader
        request = mode_request

    elif request.param == "comfort":
        name = "comfort_model"
        models = {"comfort_model": trained_comfort_model}
        actor_class = prediction_service.ComfortModelActorReload
        loader_class = prediction_service.ComfortModelLoader
        request = comfort_request

    for model_name, model_key in prediction_service.MODELS[name].items():
        model_store.save(model_key, models[model_name])

    loader = loader_class(model_store)
    pred_actor = actor_class(loader)
    service = prediction_service.PredictionService(router, pred_actor)

    await asyncio.sleep(0)

    return dealer, service, request


@pytest.mark.asyncio
async def test_prediction_service(service):
    dealer, _, request = service
    resp = await dealer.ask(request)
    assert "prediction" in resp

    # test with timeout

    async def check_raises():
        with pytest.raises(asyncio.TimeoutError):
            await request_with_timeout(timedelta(seconds=1e-5), dealer, request)

    await check_raises()

    # test success
    resp = await request_with_timeout(timedelta(seconds=1.0), dealer, request)
    assert "prediction" in resp


@pytest.fixture
def test_params():
    return {
        "prediction_client": None,
        "scaled_target_delta": -2.0,
        "features": {
            "temperature_out": 30.94,
            "power_hist": Power.ON,
            "temperature_set_last": 17,
            "humidity_out": 68.99999976158142,
            "appliance_id": "f3b9b5c8-d144-41f2-b0ed-55273579c36f",
            "humidex_out": 42.534491790354124,
            "humidex": 31.645695617501126,
            "quantity": "temperature",
            "temperature": 29.63,
            "temperature_delta": -0.00033333333333347524,
            "humidity": 32.83,
            "temperature_out_mean_day": 28.890833333333333,
            "target": -0.129999999999999,
            "mode_hist": "cool",
        },
        "humidities": [35.02, 35.16],
        "current_mode": "cool",
        "log": mock.Mock(),
    }


prediction_service_client_error = mock.CoroutineMock(
    return_value="Mode Model Should Not Be Called"
)

prediction_service_client_test = mock.CoroutineMock(
    return_value={
        "first_layer_cool": 1.0,
        "first_layer_heat": 0.0,
        "cool": 1.0,
        "heat": 0.0,
    }
)


@pytest.mark.parametrize(
    "extra_params, result",
    [
        pytest.param(
            {
                "mode_selection": ["cool"],
                "prediction_service_client": prediction_service_client_error,
            },
            "cool",
            id="only one mode is selected",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "heat"],
                "prediction_service_client": prediction_service_client_test,
            },
            "cool",
            id="Mode model prediction is used",
        ),
        pytest.param(
            {
                "scaled_target_delta": 3,
                "mode_selection": ["cool", "dry"],
                "prediction_service_client": prediction_service_client_test,
            },
            "cool",
            id="Dry mode is selected and it is getting cold",
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_mode_prediction(test_params, extra_params, result):

    params = test_params.copy()
    params.update(extra_params)
    best_mode = await prediction_service.get_mode_prediction(**params)
    assert best_mode == result
