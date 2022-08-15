import logging
import os
import tempfile
from unittest import mock

import pytest
from raven.handlers.logging import SentryHandler

from ..utils import log_util

SENTRY_DSN = "https://XXX:YYY@sentry.io/ZZZ"


def test_kv_format():
    runs = [
        (1, 1),
        ("hello", "hello"),
        ({"key": "value"}, "key=value"),
        ({"key": 1.1111}, "key=1.11"),
    ]
    for arg, result in runs:
        assert log_util.kv_format(arg, digits=2) == result


@pytest.fixture(params=["json", "text"])
def log_json(request):
    return request.param == "json"


@pytest.fixture(params=[None, SENTRY_DSN])
def sentry_dsn(request):
    return request.param


@pytest.fixture()
def patched_sentry_handler():
    def patch(dsn):  # pylint: disable=unused-argument
        handler = mock.Mock()
        handler.level = logging.ERROR
        return handler

    return patch


def test_init_logging(log_json, sentry_dsn, patched_sentry_handler):
    # potential race conditions, but should be okay just for this test
    log = log_util.get_logger("test")

    fname = next(tempfile._get_candidate_names())
    tmpdir = tempfile._get_default_tempdir()
    log_file = os.path.join(tmpdir, fname)
    log_file_path = f"{log_file}.log"

    with mock.patch("skynet.utils.log_util.SentryHandler", patched_sentry_handler):
        log_util.init_logging(
            log_file=log_file,
            log_names=["test"],
            log_json=log_json,
            sentry_dsn=sentry_dsn,
        )

    log.error("a message")
    if log_json:
        log.error("another message", extra={"data": {"some": "data"}})

    with open(log_file_path, "r") as f:
        text = f.read()
        assert "a message" in text
        if log_json:
            assert "another message" in text
            assert "some" in text
            assert "data" in text
    try:
        os.remove(log_file_path)
    except OSError:
        pass


def test_extract_loggin_params_from_config():
    logging_name = "control"
    cnf = {"logging": {"control": {"backup_count": 7}}}
    params = log_util.extract_logging_params_from_config(
        logging_name, cnf, log_json=False
    )
    assert params == {"log_file": "control", "backup_count": 7, "log_json": False}

    cnf = {"logging": {"default": {"backup_count": 7}}}
    params = log_util.extract_logging_params_from_config(
        logging_name, cnf, log_json=False
    )
    assert params == {"log_file": "control", "backup_count": 7, "log_json": False}

    cnf = {}
    params = log_util.extract_logging_params_from_config(
        logging_name, cnf, log_json=False
    )
    assert params == {"log_file": "control", "log_json": False}


def test_init_logging_from_config():
    logging_name = "control"
    cnf = {"logging": {"control": {"backup_count": 7}}}
    with mock.patch("skynet.utils.log_util.init_logging") as m:
        log_util.init_logging_from_config(logging_name, cnf)
        params = log_util.extract_logging_params_from_config(logging_name, cnf)
        m.assert_called_once_with(**params)


def test_sentry_setup():
    logger = log_util.get_logger("sentry")
    log_util.add_sentry_handler(logger, SENTRY_DSN)
    handler = logger.handlers[0]
    assert isinstance(handler, SentryHandler)
    assert handler.level == logging.ERROR
