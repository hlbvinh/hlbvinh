import asyncio
import os
import socket
import string
from datetime import datetime, timedelta

import numpy as np
import pytest
from asynctest import mock
from cryptography.fernet import Fernet

from ..utils.database import queries
from ..utils.database.dbconnection import DBConnection, Pool
from ..utils.enums import Power

# pylint: disable=import-outside-toplevel


@pytest.fixture()
def user_id():
    return "user0"


@pytest.fixture
def port():
    """Get a free port."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def db():
    with DBConnection(**DBConnection.get_config()) as dbcon:
        yield dbcon


@pytest.fixture
def cassandra_session():

    from ..utils.database import cassandra

    with cassandra.CassandraSession(
        **cassandra.CassandraSession.get_config()
    ) as session:
        session.execute(
            "CREATE KEYSPACE IF NOT EXISTS test WITH replication = "
            "{'class':'SimpleStrategy', 'replication_factor' : 3}"
        )
        session.set_keyspace("test")
        yield session


@pytest.fixture
def device_intervals(db):
    intervals = queries.get_device_online_intervals(db)
    assert intervals
    return intervals


@pytest.fixture
def device_id(db):
    return queries.execute(db, "SELECT device_id FROM Device")[-1]["device_id"]


@pytest.fixture
async def appliance_id(pool, device_id):
    return await queries.get_appliance(pool, device_id)


@pytest.fixture
def user_db(db, device_id, user_id):
    def clean():
        queries.execute(db, "DELETE From UserFeedback WHERE user_id = %s", (user_id,))
        queries.execute(db, "DELETE From User WHERE user_id = %s", (user_id,))
        queries.execute(db, "DELETE From ModeFeedback WHERE user_id = %s", (user_id,))

    queries.execute(
        db,
        "INSERT INTO User (user_id, created_on, email)" "VALUES (%s, NOW(), 'a@b.com')",
        (user_id,),
    )
    queries.execute(
        db,
        "INSERT INTO UserFeedback " "VALUES (NULL, %s, %s, 1.0, NOW(), 'Climate')",
        (device_id, user_id),
    )

    queries.execute(
        db,
        "INSERT INTO ModeFeedback VALUES (NULL, %s, %s, 'cool', NOW(), 'Climate')",
        (device_id, user_id),
    )
    yield db

    clean()


@pytest.fixture
def feedback_db(device_intervals, user_db, user_id):
    import pandas as pd

    def insert(params):
        queries.execute(
            user_db,
            """
            INSERT INTO UserFeedback (row_id, device_id, user_id,
                                      feedback, created_on)
            VALUES (NULL, %(device_id)s, %(user_id)s,
                    %(feedback)s, %(created_on)s)""",
            params,
        )

    rng = np.random.RandomState(0)
    for device in device_intervals:
        start = device["start"] + timedelta(hours=1)
        end = device["end"] - timedelta(hours=1)

        dt = (end - start).total_seconds() / 20
        for ts in pd.date_range(start, end, freq=pd.DateOffset(seconds=dt)):
            timestamp = ts.to_pydatetime()
            fb_row = {
                "device_id": device["device_id"],
                "user_id": user_id,
                "created_on": timestamp,
                "feedback": rng.rand() * 10 - 5,
            }
            insert(fb_row)

    def clear():
        queries.execute(user_db, "DELETE FROM UserFeedback")

    yield user_db

    clear()


@pytest.fixture
def device_appliance_rows(db):
    rows = queries.execute(db, *queries.query_device_appliance_list())
    assert rows, "no device appliance rows"
    return rows


@pytest.fixture
async def pool(db):
    pool = Pool.from_dbconnection(db)
    yield pool

    pool.close()
    await pool.wait_closed()


@pytest.fixture
def mongo_client():

    from ..utils.mongo import Client

    return Client(**Client.get_config())


def _random_string(n, chars=string.ascii_lowercase):
    return "".join(np.random.choice(list(chars), size=n))


@pytest.fixture
def s3_model_store(worker_id):

    from moto import mock_s3
    from ..utils.storage import S3Storage

    # The latest moto/boto/botocore requires dummy credentials to function. It
    # is unclear if this is a bug or intended behavior.
    # https://github.com/spulec/moto/issues/1793#issuecomment-431459262
    # https://github.com/spulec/moto/issues/1924
    os.environ["AWS_ACCESS_KEY_ID"] = "access"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secret"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    mock_s3().start()
    key = Fernet.generate_key()
    storage = S3Storage("bucket" + worker_id, encryption_key=key)
    yield storage

    storage.drop()
    mock_s3().stop()


@pytest.fixture
def filestorage(tmpdir):
    from ..utils.storage import FileStorage

    return FileStorage(tmpdir)


@pytest.fixture(params=["s3_model_store", "filestorage"])
def model_store(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def user_sample_store(mongo_client):

    from ..user.store import UserSampleStore

    s_col = _random_string(10)
    wm_col = _random_string(10)

    def drop():
        mongo_client._col(s_col).drop()
        mongo_client._col(wm_col).drop()

    yield UserSampleStore(
        mongo_client, sample_collection=s_col, watermark_collection=wm_col
    )

    drop()


@pytest.fixture
def climate_sample_store(mongo_client):
    from ..sample.climate_sample_store import ClimateSampleStore

    s_col = _random_string(10)
    wm_col = _random_string(10)

    def drop():
        mongo_client._col(s_col).drop()
        mongo_client._col(wm_col).drop()

    yield ClimateSampleStore(
        mongo_client, sample_collection=s_col, watermark_collection=wm_col
    )

    drop()


@pytest.fixture(params=["user", "climate"])
def sample_store(request, climate_sample_store, user_sample_store):
    if request.param == "user":
        return user_sample_store
    if request.param == "climate":
        return climate_sample_store
    raise ValueError


@pytest.fixture
def worker_id(request):
    if hasattr(request.config, "slaveinput"):
        return request.config.slaveinput["slaveid"]
    return "master"


@pytest.fixture
async def rediscon(worker_id):

    from ..utils import redis_util

    # TODO make work locally and on circleci
    cnf = redis_util.get_config()

    # to prevent bad flushing when using xdist, by default up to 16 databases
    # for redis
    if worker_id == "master":
        redis_db = 1
    else:
        redis_db = int("".join(x for x in worker_id if x.isdigit())) + 1
    if redis_db > 15:
        raise RuntimeError(
            "Not enough potential redis DBs for parallel tests."
            "Increase number of redis databases."
        )

    cnf["db"] = redis_db
    redis = redis_util.get_redis(**cnf)
    yield redis

    await redis.flushdb()
    redis.connection_pool.disconnect()


@pytest.fixture
def ir_feature():
    ir_feature = {
        "cool": {
            "temperature": {
                "ftype": "select_option",
                "value": [
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "27",
                    "28",
                    "29",
                    "30",
                ],
            },
            "fan": {"ftype": "select_option", "value": ["auto", "high", "med", "low"]},
        },
        "fan": {
            "fan": {"ftype": "select_option", "value": ["low", "med", "high", "auto"]}
        },
    }
    return ir_feature


@pytest.fixture
def monitor_interval():
    return 0.01


@pytest.fixture
def my_monitor(monitor_interval):

    from ..utils import monitor

    return monitor.Monitor(interval=monitor_interval)


@pytest.fixture
def weather():
    return {"humidity_out": 10, "temperature_out": 10, "timestamp": datetime.utcnow()}


@pytest.fixture
async def device_mode_preference_db_pool(
    user_db, device_id, pool  # pylint:disable=unused-argument
):
    await pool.execute(
        *insert_device_mode_preference(
            "u", device_id, "temperature", True, True, True, False, False
        )
    )

    await pool.execute(
        *insert_device_mode_preference(
            "u", device_id, "temperature", True, True, False, False, False
        )
    )

    yield pool

    await pool.execute("TRUNCATE TABLE DeviceModePreference")


def insert_device_mode_preference(
    user_id, device_id, quantity, cool, heat, dry, fan, auto
):
    q = """
    INSERT INTO DeviceModePreference
        (id, user_id, device_id, created_on, quantity, cool, heat, dry, fan,
         auto)
    VALUES
        (NULL, %s, %s, NOW(), %s, %s, %s, %s, %s, %s)"""
    return q, (user_id, device_id, quantity, cool, heat, dry, fan, auto)


@pytest.fixture
async def get_response():
    """
    sent out in message and receive responese from actor
    """

    from ambi_utils.zmq_micro_service.msg_util import AircomRequest

    async def wrap(client, msg):
        req = AircomRequest.from_dict(msg)
        return await client.ask(req)

    return wrap


@pytest.fixture
def sensors():
    return {
        "created_on": datetime.utcnow(),
        "temperature": 10.1,
        "humidity": 10.2,
        "luminosity": 10.0,
    }


@pytest.fixture
def control_target(device_id):
    return dict(
        device_id=device_id,
        quantity="comfort",
        value=None,
        origin="user",
        created_on=datetime.utcnow(),
    )


@pytest.fixture
def feedback(user_id):
    return {"feedback": 0, "created_on": datetime.utcnow(), "user_id": user_id}


@pytest.fixture
def state(appliance_id):
    return {
        "power": Power.ON,
        "created_on": datetime.utcnow(),
        "mode": "cool",
        "temperature": "24",
        "appliance_id": appliance_id,
        "swing": "off",
        "louver": "up",
        "fan": "low",
        "ventilation": None,
    }


@pytest.fixture
def get_db_service_msger():
    def _get_db_service_msger(code=None, response=None):
        db_service_msger = mock.MagicMock()
        db_service_result = (code, response)
        db_service_msger.ask = mock.CoroutineMock(return_value=db_service_result)
        return db_service_msger

    return _get_db_service_msger


@pytest.fixture
def db_service_msger():
    async def ask(req):
        if req.method == "DaikinVentilationOptionRead":
            return (404, {})
        return (200, mock.MagicMock())

    msger = mock.Mock()
    msger.ask = ask
    return msger


@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    tasks = asyncio.gather(*asyncio.Task.all_tasks())
    tasks.cancel()
    try:
        loop.run_until_complete(tasks)
    except asyncio.CancelledError:
        pass
