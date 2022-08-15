import asyncio
import functools
import inspect
import types
from datetime import timedelta
from typing import Dict, List

from .log_util import get_logger

log = get_logger(__name__)

PREDICTION_TIMEOUT = timedelta(seconds=2)


def run_sync(fun, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(fun(*args, **kwargs))


def run_every(seconds, fun):
    """Runs async task and then sleep in a infinite loop.

    Note that the running interval might vary depending on the elapsed time of func().

    Args:
        seconds:
        fun:

    Returns:

    """

    async def loop():
        while True:
            try:
                if inspect.iscoroutinefunction(fun):
                    await fun()
                else:
                    fun()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.exception(e)

            await asyncio.sleep(seconds)

    asyncio.ensure_future(loop())


def add_callback(fun, *args, **kwargs):
    asyncio.ensure_future(fun(*args, **kwargs))


@functools.singledispatch
async def multi(tasks):
    """Gathers multiple async tasks.

    Tasks can be passed in several types: in a List, as a values in Dict, and in a Generator.
    Results is returned in an "intuitive" type.

    Args:
        tasks (Union[Dict, List, Generator]):

    Returns:

    Warnings:
        asyncio.gather - Deprecated since version 3.8,
            will be removed in version 3.10: The loop parameter.

    """
    raise NotImplementedError(f"{type(tasks)} not supported")


@multi.register(Dict)
async def multi_dict(tasks: Dict) -> Dict:
    return dict(
        zip(tasks.keys(), await asyncio.gather(*tasks.values()))
    )  # does order match?


@multi.register(List)
async def multi_list(tasks: List) -> List:
    return list(await asyncio.gather(*tasks))


@multi.register(types.GeneratorType)
async def multi_generator(tasks: types.GeneratorType) -> List:
    return list(await asyncio.gather(*tasks))


def run_in_executor(fun):
    def wrap(*args, **kwargs):
        asyncio.get_event_loop().run_in_executor(None, fun, *args, **kwargs)

    return wrap


async def request_with_timeout(timeout, actor, req):
    try:
        return await asyncio.wait_for(actor.ask(req), timeout.total_seconds())
    except asyncio.TimeoutError as exc:
        raise asyncio.TimeoutError("request {} timed out".format(req.method)) from exc
