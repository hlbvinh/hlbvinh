from unittest import mock

import pymysql
import pytest

from ..utils.database import queries

# silence the logger
queries.log = mock.MagicMock()


def test_mysql_reconnect(db):
    def query(retry):
        return queries.execute(db, "SHOW DATABASES", retry=retry)

    assert query(retry=False)

    db.db.kill(db.db.thread_id())

    with pytest.raises(pymysql.err.InterfaceError):
        query(retry=False)

    resp = query(retry=True)
    assert resp
