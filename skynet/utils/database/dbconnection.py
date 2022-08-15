import asyncio
from datetime import timedelta

import aiomysql
import pymysql

from ..log_util import get_logger
from ..misc import get_config_parameter

log = get_logger(__name__)

MYSQL_QUERY_TIMEOUT = timedelta(seconds=10)


class DBConnection:
    """
    use with 'with' statement:

        with DBConnection() as dm:
            do some stuff

    This makes sure that the connection is closed at the end.
    """

    @classmethod
    def get_config(cls, **kwargs):
        return {
            "user": get_config_parameter("user", kwargs, "MYSQL_USER", "test"),
            "passwd": get_config_parameter("passwd", kwargs, "MYSQL_PASSWORD", "test"),
            "db": get_config_parameter("db", kwargs, "MYSQL_DB", "TestAmbiNet"),
            "host": get_config_parameter("host", kwargs, "MYSQL_HOST", "127.0.0.1"),
        }

    def __init__(self, **kwargs):
        self.db = pymysql.connect(
            cursorclass=pymysql.cursors.DictCursor, charset="utf8", **kwargs
        )
        self.kwargs = kwargs

    def cursor(self):
        return self.db.cursor()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.db.close()


class Pool(aiomysql.Pool):
    @classmethod
    def from_dbconnection(cls, dbcon: DBConnection):
        pool = get_pool(**dbcon.kwargs)
        return pool

    async def execute(self, query, args=()):
        try:
            return await asyncio.wait_for(
                self.execute_(query, args), MYSQL_QUERY_TIMEOUT.total_seconds()
            )

        except asyncio.TimeoutError as exc:
            raise asyncio.TimeoutError(
                f"mysql query {query} timed out. Exception - {exc}"
            )

    async def execute_(self, query, args):
        async with self.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, args)
                result = await cur.fetchall()
                return result


def get_pool(**kwargs):
    kwargs = kwargs.copy()
    kwargs["password"] = kwargs.pop("passwd")
    loop = kwargs.pop("loop", None) or asyncio.get_event_loop()

    return Pool(
        minsize=1,
        maxsize=10,
        echo=False,
        loop=loop,
        cursorclass=aiomysql.DictCursor,
        pool_recycle=3,
        **kwargs,
        autocommit=True,
    )
