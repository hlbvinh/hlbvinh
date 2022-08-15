import collections
import logging
import os
from logging.handlers import TimedRotatingFileHandler as FileHandler
from typing import Any, Dict, TypeVar

from pythonjsonlogger import jsonlogger
from raven.handlers.logging import SentryHandler

CONFIG_FILE_AUTHORIZED_PARAMS = ["backup_count"]
T = TypeVar("T")


def kv_format(d, digits=0):
    if isinstance(d, collections.Mapping):
        keys = sorted(d)
        parts = []
        for k in keys:
            v = d[k]
            if isinstance(v, float):
                parts.append("{0}={1:.{2}f}".format(k, v, digits))
            else:
                parts.append("{}={}".format(k, v))
        return " ".join(parts)
    return d


class KVFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        data = getattr(record, "data", None)
        if data:
            extra_msg = " |".join(
                "{}: {}".format(k, kv_format(v, digits=3)) for k, v in data.items()
            )
            return "{} | {}".format(msg, extra_msg)
        return msg


def init_logging(
    log_file,
    log_names=["skynet"],
    loglevel=logging.DEBUG,
    when="midnight",
    log_directory="log",
    backup_count=14,
    log_to_file=True,
    log_json=False,
    sentry_dsn=None,
):
    loggers = [logging.getLogger(log_name) for log_name in log_names]

    for logger in loggers:
        logger.propagate = False
        logger.setLevel(loglevel)
        if sentry_dsn is not None:
            add_sentry_handler(logger, dsn=sentry_dsn)

    fmt = "%(asctime)s %(name)s:%(lineno)d %(levelname)s " "%(process)d %(message)s"

    if log_json:
        stream_formatter = KVFormatter(fmt)
        file_formatter = jsonlogger.JsonFormatter(fmt)
    else:
        stream_formatter = logging.Formatter(fmt)
        file_formatter = stream_formatter

    # the stream handler logs to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    for logger in loggers:
        logger.addHandler(stream_handler)

    # try to create the logging directory for the file handler
    try:
        os.mkdir(log_directory)
    except OSError:
        pass

    if log_to_file:
        file_handler = FileHandler(
            os.path.join(log_directory, "{}.log".format(log_file)),
            when=when,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        for logger in loggers:
            logger.addHandler(file_handler)


def add_sentry_handler(logger, dsn):
    handler = SentryHandler(dsn)
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)


def get_logger(name=None):
    return logging.getLogger(name)


def extract_logging_params_from_config(
    logging_name: str, cnf: Dict["str", Any], **logging_params
) -> Dict["str", Any]:

    params: Dict["str", Any] = {}
    if "logging" in cnf:
        if logging_name in cnf["logging"]:
            params = cnf["logging"][logging_name]
        elif "default" in cnf["logging"]:
            params = cnf["logging"]["default"]

    filtered_params = {
        k: v for k, v in params.items() if k in CONFIG_FILE_AUTHORIZED_PARAMS
    }

    # overwrite params using values in config file
    logging_params.update(filtered_params)
    logging_params.update({"log_file": logging_name})

    return logging_params


def init_logging_from_config(
    logging_name: str, cnf: Dict["str", Any], **logging_params
) -> None:

    logging_params = extract_logging_params_from_config(
        logging_name, cnf, **logging_params
    )
    init_logging(**logging_params)
