import asyncio
import logging
from datetime import datetime, timedelta

import click
import numpy as np
import yaml
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor, RouterActor

from skynet.prediction import mode_model, predict, prediction_service
from skynet.utils.async_util import run_sync
from skynet.utils.enums import Power
from skynet.utils.log_util import get_logger, init_logging
from skynet.utils.storage import get_storage

np.random.seed(1)
log = get_logger("skynet")


async def benchmark_comfort(actor):

    features = {
        "device_id": "EDCFFF303433463443206509",
        "humidity": 55.072500228881836,
        "humidity_out": 82.04166640837987,
        "luminosity": 13.0,
        "temperature": 32.93249988555908,
        "temperature_out": 28.00083333333333,
        "tod_cos": -0.782608156852414,
        "tod_sin": 0.6225146366376193,
        "tow_cos": 0.9953961983671789,
        "tow_sin": -0.09584575252022395,
        "toy_cos": 0.3772774848710263,
        "toy_sin": -0.9261002642313587,
        "user_id": "1e5aa52a-9302-11e3-a724-0683ac059bd8",
    }

    futures = []
    import uuid

    for i in range(1000):
        features["device_id"] = str(uuid.uuid4())
        features["user_id"] = str(uuid.uuid4())
        features["temperature_out"] = np.random.rand() * 10 + 20
        msg = {
            "method": "comfort_model_prediction",
            "params": {"features": features},
            "session_id": None,
        }
        req = AircomRequest.from_dict(msg)
        futures.append(actor.ask(req))

    for i, f in enumerate(futures):
        res = await f
        print(i, res)

    return None


async def benchmark_climate(actor):

    features = {
        "feedback": 2.0,
        "humidity": 55.072500228881836,
        "humidity_out": 82.04166640837987,
        "humidex": 36,
        "humidex_out": 35,
        "luminosity": 13.0,
        "pircount": 1.5,
        "pirload": 3.0,
        "temperature": 35.93249988555908,
        "temperature_set_last": 27,
    }
    futures = []
    # import uuid
    states = [s for s in predict.generate_on_signals() if s["mode"] == "cool"]
    for i in range(1000):
        print("sending ...")
        features["appliance_id"] = "251cc694-97b7-11e3-a724-0683ac059bd8"
        features["temperature_out"] = np.random.rand() * 10 + 25
        msg = {
            "method": "climate_model_prediction",
            "params": {
                "history_features": features,
                "states": states,
                "quantity": "temperature",
            },
            "session_id": None,
        }
        req = AircomRequest.from_dict(msg)
        futures.append(actor.ask(req))

    for i, f in enumerate(futures):
        res = await f
        print(i, res)

    return None


async def benchmark_mode(actor):

    features = {
        "feedback": 2.0,
        "humidity": 55.072500228881836,
        "humidity_out": 82.04166640837987,
        "humidex": 36,
        "humidex_out": 35,
        "luminosity": 13.0,
        "pircount": 1.5,
        "pirload": 3.0,
        "temperature": 35.93249988555908,
        "temperature_out_mean_day": 20.0,
        "temperature_set_last": 27,
        "power_hist": Power.OFF,
        "mode_hist": "cool",
    }
    futures = []
    # import uuid
    for i in range(1000):
        features["appliance_id"] = "251cc694-97b7-11e3-a724-0683ac059bd8"
        features["temperature_out"] = np.random.rand() * 10 + 25
        mode_features = mode_model.make_features(features, "humidex", 0.0)
        msg = {
            "method": "mode_model_prediction",
            "params": {"mode_features": mode_features},
            "session_id": None,
        }
        req = AircomRequest.from_dict(msg)
        futures.append(actor.ask(req))

    for i, f in enumerate(futures):
        res = await f
        print(i, res)

    return None


@click.command()
@click.option("--config", default="config.yml")
@click.option("--client", is_flag=True)
@click.option(
    "--model_type", type=click.Choice(["climate", "mode", "comfort"]), default="climate"
)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
def main(config, client, model_type, storage):

    cnf = yaml.safe_load(open(config))
    init_logging(
        "prediction_service_{}".format(model_type), loglevel=logging.INFO, log_json=True
    )

    if model_type == "climate":
        service_cnf = cnf["prediction_services"]["climate_model"]
        Loader = prediction_service.ClimateModelLoader
        ModelActor = prediction_service.ClimateModelActorReload
        benchmark_fun = benchmark_climate

    elif model_type == "mode":
        service_cnf = cnf["prediction_services"]["mode_model"]
        Loader = prediction_service.ModeModelLoader
        ModelActor = prediction_service.ModeModelActorReload
        benchmark_fun = benchmark_mode

    elif model_type == "comfort":
        service_cnf = cnf["prediction_services"]["comfort_model"]
        Loader = prediction_service.ComfortModelLoader
        ModelActor = prediction_service.ComfortModelActorReload
        benchmark_fun = benchmark_comfort

    if client:
        actor = DealerActor(**service_cnf, log=log)
        run_sync(benchmark_fun, actor)

    else:
        model_store = get_storage(
            storage, **cnf["model_store"], directory="data/models"
        )

        # loader = Loader(storage, reload_seconds=10)
        loader = Loader(model_store)
        model_actor = ModelActor(loader)

        msger = RouterActor(**service_cnf, log=log)
        service = prediction_service.PredictionService(
            prediction_actor=model_actor, router_actor=msger
        )

        print(service)
        asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
