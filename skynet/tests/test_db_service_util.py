from unittest import mock

import numpy as np
import pytest

from ..utils.db_service import db_service_util


def test_get_irorigin():
    data = [
        ("User", "skynet"),
        ("Geo", "skynet_geo"),
        ("Timer", "skynet_timer"),
        ("Reverse", "skynet"),
    ]

    for origin, actor in data:
        assert db_service_util.get_irorigin({"origin": origin}) == actor


def test_get_aircom_request():
    req = db_service_util.get_aircom_request("AircomRequest", {"test": "test"})
    assert req.method == "AircomRequest"
    assert req.params == {"test": "test"}


@pytest.mark.parametrize(
    "temperature, result",
    [(14, 14), ("14", "14"), (np.int64(14), 14), (14.5, 14), (np.float32(14.5), 14)],
)
def test_ensure_temperature_is_serialisable(temperature, result):
    assert (
        db_service_util.ensure_temperature_is_serialisable(temperature, mock.Mock())
        == result
    )
