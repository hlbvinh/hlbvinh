from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from dateutil.relativedelta import relativedelta

from . import log_util, redis_util
from .database.cassandra import CassandraSession
from .database.dbconnection import get_pool
from .mongo import Client
from .types import Config, Connections

log = log_util.get_logger("skynet")


def get_connections(
    config: Config,
    mysql: Optional[str] = None,
    cassandra: Optional[str] = None,
    mongo: Optional[str] = None,
    redis: bool = False,
    db_service_msger: Optional[str] = None,
) -> Connections:
    connections = {}
    if mysql:
        connections["pool"] = get_pool(**config[mysql])
    if cassandra:
        connections["session"] = CassandraSession(**config["cassandra"][cassandra])
    if mongo:
        connections["mongo"] = Client(**config["mongo"][mongo])
    if redis:
        connections["redis"] = redis_util.get_redis(config)
    if db_service_msger:
        connections["db_service_msger"] = DealerActor(
            log=log, **config[db_service_msger]
        )

    return Connections(**connections)


def get_sample_limit_according_to_number_of_months(
    samples: pd.DataFrame, month_limit: int, validation_interval: int
) -> int:
    samples = samples.reset_index()
    most_recent_date_after_excluding_samples_within_validation_interval = samples.iloc[
        -1
    ]["timestamp"] - relativedelta(days=validation_interval)
    samples_excluding_validation_interval_samples = samples.loc[
        samples["timestamp"]
        < most_recent_date_after_excluding_samples_within_validation_interval
    ]
    X = samples_excluding_validation_interval_samples.copy()
    X["end_time"] = X.iloc[-1]["timestamp"]

    if (
        datetime.utcnow()
        - relativedelta(months=month_limit)
        - relativedelta(days=validation_interval)
        < X.iloc[0]["timestamp"]
    ):
        log.warning(
            "Old dataset. Use recent dataset or add month equivalent "
            "to current date - most recent date on the dataset to the FETCH_MONTH_BUFFER_OFFSET"
        )
    X["timedelta"] = X["end_time"] - X["timestamp"]

    return len(X.loc[X["timedelta"] < timedelta(days=31 * month_limit)])
