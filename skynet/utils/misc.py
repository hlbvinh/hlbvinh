import functools
import os
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

import skynet

from .log_util import get_logger

log = get_logger(__name__)


def get_config_parameter(name, kwargs, environment_variable, default):
    return kwargs.get(name, os.environ.get(environment_variable, default))


def base_dir():
    return os.path.split(os.path.dirname(os.path.abspath(skynet.__file__)))[0]


def get_dir(ext):
    return os.path.join(base_dir(), ext)


def timeit():
    def wrap(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            t_in = datetime.utcnow()
            ret_val = f(*args, **kwargs)
            msg = "{}.{} took {:.3f} ms".format(
                args[0].__class__.__name__ if args else "unbound",
                f.__name__,
                1000 * (datetime.utcnow() - t_in).total_seconds(),
            )
            log.debug(msg)
            return ret_val

        return wrapped_f

    return wrap


@contextmanager
def timeblock(label):
    start = time.perf_counter()
    try:
        yield

    finally:
        end = time.perf_counter()
        log.info(f"{label} : {end - start}")


def json_timeit(service, event):
    def wrap(f):
        @functools.wraps(f)
        async def wrapped_f(*args, **kwargs):
            tic = time.perf_counter()
            ret_val = await f(*args, **kwargs)
            log_msg = {
                "service": service,
                "event": event,
                "response_time": 1000 * (time.perf_counter() - tic),
            }
            log.info(event, extra={"data": log_msg})
            return ret_val

        return wrapped_f

    return wrap


def na_columns(x):
    """Return corresponding column labels of columns with nan values."""
    return tuple(x.columns[x.isna().any()])


def na_index(x):
    """Return corresponding index labels for nan values of TimeSeries."""
    return tuple(x.index[x.isna()])


def check(check_fun, error):
    def wrap(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):  # pylint:disable=R1710
            ret = f(*args, **kwargs)
            if check_fun(ret):
                return ret
            raise error

        return wrapped_f

    return wrap


def check_na(exc, msg):
    def wrap(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):  # pylint:disable=R1710
            ret = f(*args, **kwargs)
            if isinstance(ret, dict):
                nas = [
                    k
                    for k, v in ret.items()
                    if not isinstance(v, str) and not np.isfinite(v)
                ]
            elif isinstance(ret, pd.DataFrame):
                nas = na_columns(ret)
            elif isinstance(ret, pd.Series):
                nas = na_index(ret)
            else:
                raise ValueError("return value must be Series or DataFrame")

            if nas:
                raise exc("{} {}".format(msg, nas))
            return ret

        return wrapped_f

    return wrap


def log_format(d):
    if hasattr(d, "items"):
        return {k: log_format(v) for k, v in d.items()}
    if isinstance(d, str):
        return d
    if isinstance(d, (list, tuple)):
        return type(d)([log_format(x) for x in d])
    return "{:.1f}".format(d) if isinstance(d, float) else d


class UpdateCounter:
    """Simple Counter that calculates the rate of counts."""

    def __init__(self):
        self.start_timestamp = None
        self.counter = 0

    def start(self):
        self.start_timestamp = datetime.utcnow()

    def update(self):
        self.counter += 1

    def read(self, reset=False):
        """Calculate number of counts per minute then resets the counter if required.

        Args:
            reset:

        Returns:

        """
        now = datetime.utcnow()
        minutes = (now - self.start_timestamp).total_seconds() / 60
        updates_per_minute = self.counter / minutes
        if reset:
            self.counter = 0
            self.start()
        return updates_per_minute


def elapsed(timestamp, period):
    """Returns time elapsed since timestamp.

    Parameters
    ----------
    timestamp : datetime or None

    period : timedelta

    Returns
    -------
    bool
    """
    if timestamp is None:
        return True
    return (datetime.utcnow() - timestamp) > period
