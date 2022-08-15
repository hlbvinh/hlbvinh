import pymongo
from pymongo import MongoClient
from pymongo.operations import ReplaceOne

from .data import _to_native
from .log_util import get_logger
from .misc import get_config_parameter

log = get_logger(__name__)
_connection = None

__all__ = ["get_connection"]

MONGO_TIMEOUT = 10 * 1000


def get_connection(uri):
    global _connection  # pylint: disable=global-statement
    if not _connection:
        log.info("making new connection to MongoDB")
        _connection = MongoClient(uri, serverSelectionTimeoutMS=MONGO_TIMEOUT)
    return _connection


class Client:
    @classmethod
    def get_config(cls, **kwargs):
        return {
            "user": get_config_parameter("user", kwargs, "MONGO_USER", "test"),
            "password": get_config_parameter("password", kwargs, "MONGO_PASSWORD", ""),
            "db": get_config_parameter("db", kwargs, "MONGO_DB", "test"),
            "host": get_config_parameter("host", kwargs, "MONGO_HOST", "localhost"),
        }

    def __init__(
        self, host="localhost", db="samples", port=27017, user=None, password=None
    ):
        """PyMongo Storage Module.

        Parameters
        ----------
        host: str (default='localhost')

        db: str (default='samples')

        port: int (default=27017)

        user: str (default=None)

        password: str (default=None)
        """
        self.host = host
        self.db = db
        self.port = port
        self.user = user
        self.password = password if password != "" else None
        log.debug("created MongoClient object")
        log.debug("testing MongoClient connection")
        self._test_connection()

    def _test_connection(self):
        for i in range(3):
            try:
                self._client().server_info()
            except Exception as e:
                log.exception(f"MongoClient failed connection: {e}")
                if i == 2:
                    raise

            else:
                break

    def _col(self, collection):
        return self._client()[self.db][collection]

    def _client(self):

        if self.password is not None:
            uri = "mongodb://{}:{}@{}:{}/{}".format(
                self.user, self.password, self.host, self.port, self.db
            )
        else:
            uri = "mongodb://{}/{}".format(self.host, self.db)

        return get_connection(uri)

    def create_index(self, collection, keys):
        return self._col(collection).create_index(keys)

    def upsert(self, collection, record, key=None):
        """Save/replace python dict in mongodb collection.

        Parameters
        ----------
        collection: str
            mongodb collection

        record: dict
            object to insert into mongodb

        key: dict (default=None)
            If specified a record where this key matches is replaced.

        Raises
        ------
        ValueError
            If record is not a dict.
        """

        if not isinstance(record, dict):
            raise ValueError(
                "Can only insert dict record, got {}." "".format(type(dict))
            )

        record = _to_native(record)

        if key is None:
            key = record

        res = self._col(collection).replace_one(key, record, upsert=True)
        log.debug("saved {} {}".format(key, list(record)))
        return res

    def upsert_many(self, collection, records, keys=None):
        if not keys:
            keys = records
        requests = [ReplaceOne(k, r, upsert=True) for k, r in zip(keys, records)]
        ret = self._col(collection).bulk_write(requests)
        log.info("upserted {} records into {}" "".format(len(records), collection))
        return ret

    def get(self, collection, key={}, sort=None, limit=0, direction=pymongo.ASCENDING):
        """Load collection from mongodb.

        Parameters
        ----------
        collection: str
            mongodb collection

        key: dict (default={})
            Return only matching records when given.
        """
        if sort is None:
            return list(self._col(collection).find(key))
        if isinstance(sort, str):
            return list(
                self._col(collection)
                .find(key)
                .sort(sort, direction=direction)
                .limit(limit)
            )
        return list(self._col(collection).find(key).sort(sort).limit(limit))

    def get_one(self, collection, key={}):
        """Load collection from mongodb.

        Parameters
        ----------
        collection: str
            mongodb collection

        key: dict (default={})
            Return only matching records when given.
        """
        return self._col(collection).find_one(key)

    def remove(self, collection, key):
        """Remove items from collection in mongodb.

        Parameters
        ----------
        collection: str
            mongodb collection

        key: dict
            Remove matching records.
        """
        return self._client()[self.db][collection].delete_many(key)
