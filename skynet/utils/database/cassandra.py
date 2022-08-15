import asyncio
from asyncio import Future
from typing import List

# pylint: disable=no-name-in-module
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

# pylint: enable=no-name-in-module

from ..misc import get_config_parameter


class PagedResultHandler:
    """Handle paged results from cassandra.
    https://datastax.github.io/python-driver/query_paging.html#handling-paged-results-with-callbacks

    Note: If the asyncio ioloop is not passed as arguments but fetched via
          get_event_loop, get_event_loop hangs.

          The handle_page method runs in another thread. I think this is why
          the event loop needs to passed explicitly here.
    """

    def __init__(self, future):
        self.loop = asyncio.get_event_loop()
        self.future = future
        self.future.add_callbacks(callback=self.handle_page, errback=self.handle_error)
        self.final_future = Future()
        self._rows = []

    def handle_page(self, rows):
        self._rows.extend(rows)

        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
        else:
            self.loop.call_soon_threadsafe(self.final_future.set_result, self._rows)

    def handle_error(self, exc):
        self.loop.call_soon_threadsafe(self.final_future.set_exception, exc)


class CassandraSession:
    @classmethod
    def get_config(cls, **kwargs):
        return {
            "host": get_config_parameter("host", kwargs, "CASSANDRA_HOST", "127.0.0.1"),
            "port": get_config_parameter("port", kwargs, "CASSANDRA_DOCKER_PORT", 9042),
            "username": get_config_parameter(
                "username", kwargs, "CASSANDRA_USER", None
            ),
            "password": get_config_parameter(
                "password", kwargs, "CASSANDRA_PASSWORD", None
            ),
        }

    def __init__(self, host, port, username, password, keyspace=None):
        auth_provider = PlainTextAuthProvider(username=username, password=password)
        self._cluster = Cluster(
            [host],
            port=port,
            auth_provider=auth_provider,
            control_connection_timeout=10,
        )
        self._session = self._cluster.connect()
        self.set_keyspace(keyspace)

    def set_keyspace(self, keyspace):
        if keyspace is not None:
            return self._session.set_keyspace(keyspace)
        return self

    def execute(self, *args, **kwargs):
        return self._session.execute(*args, **kwargs)

    async def execute_async(self, *args, **kwargs) -> List:
        cassandra_future = self._session.execute_async(*args, **kwargs)
        return await PagedResultHandler(cassandra_future).final_future

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self._cluster.shutdown()
