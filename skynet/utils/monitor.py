import collections
import functools

from .async_util import run_every

from .log_util import get_logger


log = get_logger(__name__)


def monitored(name):
    def wrapped(fun):
        # Count number of calls of particular method
        @functools.wraps(fun)
        def wrap(self, *args, **kwargs):
            self.monitor.tick(
                name
            )  # The class must contains an attribute called monitor
            return fun(self, *args, **kwargs)

        return wrap

    return wrapped


def _status():
    return {"status": None, "counter": 0}


class Monitor:
    """Monitor function calls.

    Attributes:
        interval:

    Examples:
        >>> class Monitored:
        >>>    def __init__(self, monitor):
        >>>        self.monitor = monitor
        >>>
        >>>    @monitor.monitored('some_name')
        >>>    def fun(self):
        >>>        pass
        >>>
        >>> my_monitor = monitor.Monitor(interval=60)
        >>> my_monitored = Monitored(monitor)
        >>> my_monitored.fun()
        >>> monitored.maybe_message('some_name')
        None
        >>> monitored.maybe_message('some_name')
        some_name DOWN

    """

    def __init__(self, interval):
        self.stats = collections.defaultdict(_status)
        self.interval = interval
        self.is_staging = False

    def monitored(self, name):  # not using?
        def wrapped(fun):
            @functools.wraps(fun)
            def wrap(self, *args, **kwargs):
                self.tick(name)
                return fun(self, *args, **kwargs)

            return wrap

        return wrapped

    def check(self, name):  # rename to check_and_reset?
        """Checks number of calls of monitored functions and then resets the counter.

        Args:
            name:

        Returns:

        """
        current = self.stats[name]["counter"]
        self.stats[name]["counter"] = 0
        return current

    def tick(self, name):
        """Advances number of calls when monitored function is called.

        Args:
            name:

        Returns:

        """
        self.stats[name]["counter"] += 1

    def maybe_message(self, name):
        stat = self.stats[name]
        if self.check(name) > 0:
            new_status = "up"
        else:
            new_status = "down"

        if stat["status"] is None:
            if new_status == "down":
                msg = "{} DOWN".format(name)
            else:
                msg = None

        elif new_status != stat["status"]:
            if new_status == "up":
                msg = "{} recovered".format(name)
            else:
                msg = "{} DOWN".format(name)

        else:
            msg = None

        stat["status"] = new_status

        return msg

    def start(self):
        run_every(self.interval, self.log_error)

    def log_error(self):
        for name in self.stats:
            msg = self.maybe_message(name)
            if msg is not None:
                if self.is_staging:
                    log.info(f"monitor alert: {msg}")
                else:
                    log.error(f"monitor alert: {msg}")
