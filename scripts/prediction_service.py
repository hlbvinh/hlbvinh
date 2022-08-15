import asyncio
import logging

import click
import yaml
from ambi_utils.zmq_micro_service.zmq_actor import RouterActor

import uvloop
from skynet.prediction import prediction_service
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.storage import get_storage

log = get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--model_type", type=click.Choice(["climate", "mode", "comfort"]), default="climate"
)
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
def main(config, model_type, storage):
    uvloop.install()

    cnf = yaml.safe_load(open(config))

    init_logging_from_config(
        f"prediction_service_{model_type}",
        cnf=cnf,
        loglevel=logging.INFO,
        log_json=True,
    )

    if model_type == "climate":
        service_cnf = cnf["prediction_services"]["climate_model"]
        Loader = prediction_service.ClimateModelLoader
        ModelActor = prediction_service.ClimateModelActorReload

    elif model_type == "mode":
        service_cnf = cnf["prediction_services"]["mode_model"]
        Loader = prediction_service.ModeModelLoader
        ModelActor = prediction_service.ModeModelActorReload

    elif model_type == "comfort":
        service_cnf = cnf["prediction_services"]["comfort_model"]
        Loader = prediction_service.ComfortModelLoader
        ModelActor = prediction_service.ComfortModelActorReload

    model_store = get_storage(storage, **cnf["model_store"], directory="data/models")

    loader = Loader(storage=model_store)
    model_actor = ModelActor(loader)

    msger = RouterActor(log=log, **service_cnf)
    service = prediction_service.PredictionService(
        prediction_actor=model_actor, router_actor=msger
    )

    log.debug(f"STARTING {service}")
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
