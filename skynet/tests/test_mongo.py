from unittest import mock

import pytest
from pymongo import errors

from ..utils.mongo import Client


def test_mongo_timeout():
    with mock.patch("skynet.utils.mongo.MONGO_TIMEOUT", 1):
        # disable global connection
        with mock.patch("skynet.utils.mongo._connection", None):
            with pytest.raises(errors.ConnectionFailure):
                Client(host="no_host")
