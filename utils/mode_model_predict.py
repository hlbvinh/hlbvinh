from datetime import datetime

import click
import numpy as np
import pandas as pd
import yaml

from skynet.prediction import mode_model
from skynet.sample import sample
from skynet.utils.async_util import run_sync
from skynet.utils.database.cassandra import CassandraSession
from skynet.utils.database.dbconnection import get_pool
from skynet.utils.enums import Power
from skynet.utils.log_util import get_logger, init_logging
from skynet.utils.storage import get_storage

log = get_logger("skynet")

RANGES = {"temperature": np.arange(-5, 6), "humidex": np.arange(-6, 7)}


@click.command()
@click.option("--config", default="config.yml")
@click.option("--device_id", default="05E0FF303830594143208530")
@click.option("--mysql", default="viewer")
@click.option("--cassandra", default="viewer")
@click.option("--log_directory", default="log")
@click.option("--storage", type=click.Choice(["s3", "file"]), default="file")
def main(config, device_id, mysql, cassandra, log_directory, storage):
    """
    train, score: load dataset from disk if possible otherwise fetch
                  from db

    run: always fetch from DB after training upload to mongodb,

    train: store model locally using storage.Storage object
           but don't upload to mongodb
    """
    init_logging("mode_model", log_directory=log_directory)
    cnf = yaml.safe_load(open(config))
    db_cnf = cnf[mysql]

    model_store = get_storage(storage, **cnf["model_store"], directory="data/models")
    fitted_model = model_store.load(mode_model.ModeModel.get_storage_key())
    smp = sample.PredictionSample(device_id, datetime.utcnow())
    pool = get_pool(**db_cnf)
    session = CassandraSession(**cnf["cassandra"][cassandra])
    run_sync(smp.fetch, pool, session)
    history_features = smp.get_history_feature()

    feats = [
        mode_model.make_features(history_features, quantity, value)
        for quantity, values in RANGES.items()
        for value in values
    ]
    feats = pd.DataFrame(feats)

    print(feats.iloc[0])
    feats["mode_hist"] = Power.OFF

    preds = fitted_model.predict_proba(feats)
    print(pd.DataFrame.from_records(preds))


# print(feats[[
#            'p_heat', 'p_cool', 'target_temperature', 'target_humidex']])
# print(fitted_model.estimator.steps[-1][-1].classes_)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
