import asyncio
from asyncio import Future, Task
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import zmq
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from aredis import LockError
from aredis import StrictRedis as Redis
from tenacity import RetryError

from ..utils import cache_util, event_parsing, events, metrics, monitor
from ..utils.async_util import add_callback, multi, run_every
from ..utils.log_util import get_logger
from ..utils.misc import UpdateCounter
from ..utils.types import Connections
from .controller import Controller
from .util import AMBI_SENSOR_INTERVAL_SECONDS

log = get_logger(__name__)
CONTROL_CONCURRENCY = 10
CONTROLLER_LOCK_SECONDS = 10

# make sure that LISTENER_TIMEOUT and MONITOR_INTERVAL are coprimes to avoid
# resetting the 'incoming events' counter at the same time
LISTENER_TIMEOUT = 29
MONITOR_INTERVAL = 180
CONTROLLER_UPDATE_LIMIT = 10
COMFORT_PREDICTION_SECOND = 10 * 60
MAX_SENSOR_PROCESSING_DELAY = timedelta(seconds=30)
SENSOR_TRIM_INTERVAL = timedelta(seconds=10)
# computing those mode live metrics take a fair share of cpu, we
# disable it for now as we are trying to reduce AWS server cost
COMPUTE_METRICS = False


class Control:
    """
    Routes event messages from event_service and creates tasks for control worker periodically.

    Routing event messages: ListenerQueue
    Creating tasks: PeriodicalUpdate

    Attributes:
        redis: Redis client
        listener: zeroMQ client
    """

    def __init__(self, redis: Redis, listener: zmq.Socket) -> None:

        self.redis = redis
        self.listener_queue = ListenerQueue(listener, redis)
        self.periodical_update = PeriodicalUpdate(redis)

    def start(self) -> None:
        """Sets up listener and task creator.

        Returns:

        """
        log.info("starting control")
        self.listener_queue.setup_listener()
        self.periodical_update.start()
        run_every(10, self.log_stats)
        log.info("control started")

    async def log_stats(self) -> None:
        log.info(
            "controlling {} devices".format(
                len(await cache_util.get_online_devices(self.redis))
            )
        )


class ControlWorker:
    """Subscribes to topic events (in Redis) and starts corresponding controllers to handle events.

    Attributes:
        connections: connection clients
        prediction_clients:
        testing:
    """

    def __init__(
        self,
        connections: Connections,
        prediction_clients: Dict[str, DealerActor],
        testing: bool,
    ) -> None:
        self.connections = connections
        self.monitor = monitor.Monitor(interval=MONITOR_INTERVAL)
        self.prediction_clients = prediction_clients
        self.update_counter = UpdateCounter()
        self.testing = testing

    def start(self) -> None:
        self.monitor.start()
        self.update_counter.start()
        run_every(10, self.log_stats)
        add_callback(self.spawn_workers)

    def log_stats(self) -> None:
        """Configures Monitor and UpdateCounter to log status and stats of control worker.

        Returns:

        """
        updates_per_minute = self.update_counter.read()
        self.monitor.is_staging = updates_per_minute < CONTROLLER_UPDATE_LIMIT
        log.info(f"{updates_per_minute} updates per minute")

    async def spawn_workers(self) -> None:
        """Creates workers for event handling.

        Returns:

        """
        log.info("setting up workers for control events")
        coroutines = []
        for worker_id in range(CONTROL_CONCURRENCY):
            coroutines.append(self.events_worker())
            log.info(f"started control worker {worker_id}")
        await multi(coroutines)

    async def events_worker(self) -> None:
        """Picks an event and then handles it one by one.

        Events are enqueued by Control into Redis.
        In each loop, we pick an event according to its priority.
        And then create the corresponding controller to handle that event.

        Returns:

        """
        while True:
            try:
                msg = await cache_util.pick_event(self.connections.redis)
                await self.handle_listener(msg)
            except asyncio.CancelledError:
                return
            except RetryError as e:
                # now that redis connections have timeout, waiting to pick an
                # event on staging can time out
                log.info(f"worker failed: {e}")
            except Exception as e:
                log.exception(f"worker failed: {e}")
                await asyncio.sleep(0)  # why

    async def handle_listener(self, msg: Tuple[str, Any]) -> None:
        """Creates event instance for event data, cache required data, process the event if needed.

        Args:
            msg:

        Returns:

        """
        event = await events.create_event_from_listener_message(msg, self.connections)
        if not event or event.has_expired:
            return

        coroutines = [event.cache()]
        if event.is_from_event_service:
            coroutines.append(
                cache_util.set_online_device(self.connections.redis, event.device_id)
            )
        await multi(coroutines)

        if await event.need_processing():
            await self.process_event(event)

    async def process_event(self, event: events.Event) -> None:
        """Processes event by creating controller.

        Devices can be controlled by one controller only (archived using lock in Redis)

        Args:
            event:

        Returns:

        """
        lock = self.connections.redis.lock(
            event.device_id, timeout=CONTROLLER_LOCK_SECONDS
        )
        await lock.acquire()
        try:
            controller = await self.get_controller(event.device_id)
            if controller:
                await event.update_controller(controller)
        finally:
            try:
                await lock.release()
            except LockError as e:
                log.info(
                    f"could not properly release the lock for {event.device_id}: {e}"
                )

    async def get_controller(self, device_id: str) -> Optional[Controller]:
        """Returns controller instance for a particular device.

        Args:
            device_id:

        Returns:

        """
        try:
            return await Controller.from_redis(
                self.connections,
                device_id,
                self.prediction_clients,
                self.monitor,
                self.update_counter,
                self.testing,
            )

        except LookupError as exc:
            log.info(f"could not create controller for {device_id}: {exc}")
        return None


class PeriodicalUpdate:
    """Creates tasks for ControlWorker periodically in the background.

    Attributes:
        redis
    """

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    def start(self) -> None:
        """Sets up and runs tasks periodically based on predefined interval.

        Returns:

        """
        if COMPUTE_METRICS:
            run_every(COMFORT_PREDICTION_SECOND, self.predict_comfort)
            run_every(
                metrics.CLIMATE_MODEL_METRIC_INTERVAL.seconds,
                self.log_climate_model_metric,
            )
        run_every(SENSOR_TRIM_INTERVAL.total_seconds(), self.trim_sensor_event_queue)

    async def predict_comfort(self) -> None:
        log.info("enqueueing comfort predictions")
        await cache_util.create_events_for_online_devices(
            events.ComfortPredictionEvent.topic
        )(self.redis)

    async def log_climate_model_metric(self) -> None:
        log.info("enqueueing climate predictions")
        await cache_util.create_events_for_online_devices(
            events.ClimateModelMetricEvent.topic
        )(self.redis)

    async def trim_sensor_event_queue(self) -> None:
        """

        Returns:

        """
        # FIXME: getting its length by querying the whole device list??
        online_device_count = len(await cache_util.get_online_devices(self.redis))
        # trim unhandled, "expired" sensor event.
        # (I don't understand why it is necessary.
        # Is it because the control instance is not scalable?)
        max_queue_length = int(
            MAX_SENSOR_PROCESSING_DELAY.total_seconds()
            * online_device_count
            / AMBI_SENSOR_INTERVAL_SECONDS
        )
        await cache_util.trim_queue(
            self.redis, events.SensorEvent.topic, 0, max_queue_length
        )


class ListenerQueue:
    """Listens to event_service (zeroMQ) and pushes all events received into Redis.

    Attributes:
        listener:
        redis:
    """

    def __init__(self, listener: zmq.Socket, redis: Redis) -> None:
        self.listener = listener
        self.redis = redis
        self.monitor = monitor.Monitor(interval=MONITOR_INTERVAL)
        self.monitor.start()
        self.event_listener_future: Future
        run_every(LISTENER_TIMEOUT, self.check_listener)

    def setup_listener(self) -> None:
        """Creates a task for enqueuing events.

        Returns:

        """
        self.event_listener_future = Task(self.enqueue())

    async def enqueue(self) -> None:
        log.info("setting up enqueueing of incoming events")
        while True:
            try:
                await self._enqueue()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.exception(f"enqueueing failed: {e}")
                await asyncio.sleep(0)

    @monitor.monitored("incoming events")
    async def _enqueue(self) -> None:
        msg = await self.listener.recv_multipart()
        await cache_util.enqueue_event(self.redis, msg)

    def check_listener(self) -> None:
        n_events = self.monitor.check("incoming events")
        if n_events == 0 and hasattr(self, "event_listener_future"):
            log.info("no incoming events, reconnect socket")
            # pylint: disable=no-member
            endpoint = self.listener.getsockopt_string(zmq.LAST_ENDPOINT)
            host, port = event_parsing.parse_host_port(endpoint)
            # to make sure we do not hang when closing
            self.listener.setsockopt(zmq.LINGER, 0)
            # pylint: enable=no-member
            self.event_listener_future.cancel()
            self.listener.close()
            self.listener = events.get_event_listener(host, port)
            self.setup_listener()

    def __del__(self):
        """Cancels enqueuing task and aborts connection to zeroMQ before gc.

        Returns:

        """
        self.listener.close()
        if hasattr(self, "event_listener_future"):
            self.event_listener_future.cancel()
