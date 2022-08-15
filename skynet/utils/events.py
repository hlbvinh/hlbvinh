from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import zmq
import zmq.asyncio
from voluptuous import Invalid

from ..control import util

# from ..control.controller import Controller
from . import cache_util, event_parsing, parse
from .async_util import multi
from .database import queries
from .enums import EventPriority
from .log_util import get_logger
from .misc import elapsed
from .types import Connections

log = get_logger(__name__)

EVENT_REGISTRY: Dict[
    str, "Event"
] = {}  # storing the {Topic name -> Event class} mapping
EVENT_EXPIRED = timedelta(minutes=15)


def get_event_listener(ip: str, port: str) -> zmq.Socket:
    ctx = zmq.asyncio.Context()
    # pylint: disable=no-member
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{ip}:{port}")
    for topic in EVENT_REGISTRY:
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    # pylint: enable=no-member

    return sock


class Event:
    """Event base class. Providing several method to act on event data.

    Attributes:
        data: event data queried from Redis
        connections: connection clients that may or may not required by child classes

    """

    # class attributes that should be overridden by child class
    topic = ""
    is_from_event_service = True
    priority = EventPriority.HIGH

    def __init_subclass__(cls):
        super().__init_subclass__()
        EVENT_REGISTRY[cls.topic] = cls

    def __init__(self, data: Dict[str, Any], connections: Connections) -> None:
        self.raw_data = data
        self.msg = (self.topic, self.raw_data)
        self.connections = connections
        self.device_id = self.raw_data["device_id"]

    @classmethod
    async def from_listener_message(
        cls, data: Dict[str, Any], connections: Connections
    ) -> Optional["Event"]:
        """Returns corresponding event instance if the event data can be correctly parsed.

        Args:
            data:
            connections:

        Returns:

        """
        self = cls(data, connections)

        try:
            await self.data_parser()
        except Invalid:
            return None
        else:
            return self

    async def data_parser(self) -> None:
        """Parses event data

        Returns:

        Raises:
            Invalid: If event data is not possible to parse

        """
        try:
            self._parse()
        except Invalid as exc:
            await self._handle_invalid(exc)
            raise

    async def cache(self) -> None:
        """Writes transformed data into cache (Redis).

        Returns:

        """

    async def need_processing(self) -> bool:
        """Gives information to control worker that whether further action(s) is needed.

        Returns:

        """
        return True

    async def update_controller(self, controller) -> None:
        await self._update_controller(controller)
        await controller.store_state()

    async def _update_controller(self, controller) -> None:
        pass

    def _parse(self):
        """Preprocesses raw event data.

        Returns:

        """
        self.data = self.raw_data

    async def _handle_invalid(self, exc: Exception) -> None:
        """Handles data that failed to preprocess.

        Args:
            exc:

        Returns:

        """
        log.info(
            f"invalid event service data {exc}",
            extra={"data": {"device_id": self.device_id, "msg": self.msg}},
        )

    @property
    def has_expired(self):
        if isinstance(self.data, dict) and "created_on" in self.data:
            return elapsed(self.data["created_on"], EVENT_EXPIRED)
        return False


async def create_event_from_listener_message(
    msg: Tuple[str, Dict[str, Any]], connections: Connections
) -> Optional[Event]:
    """Creates an event instance based on event payload.

    Topic name determines which event class is used and

    Args:
        msg:
        connections:

    Returns:

    """
    topic, data = msg
    return await EVENT_REGISTRY[topic].from_listener_message(data, connections)


class SensorEvent(Event):
    topic = "sensor_data"
    priority = EventPriority.MEDIUM

    def _parse(self) -> None:
        self.data = event_parsing.parse_sensor_event(self.raw_data)

    async def _handle_invalid(self, exc: Exception) -> None:
        await util.handle_invalid_sensors(
            self.connections.redis, self.device_id, exc, self.msg
        )

    async def cache(self) -> None:
        await cache_util.set_sensors(
            redis=self.connections.redis, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        try:
            coroutines = {
                "control_mode": cache_util.fetch_control_mode(
                    connections=self.connections, key_arg=self.device_id
                ),
                "automated_demand_response": cache_util.get_automated_demand_response(
                    redis=self.connections.redis, key_arg=self.device_id
                ),
                "last_state_update": cache_util.get_last_state_update(
                    redis=self.connections.redis, key_arg=self.device_id
                ),
                "last_deployment": cache_util.get_last_deployment(
                    redis=self.connections.redis, key_arg=self.device_id
                ),
            }
            data = await multi(coroutines)
        except LookupError as e:
            log.info(e)
            return False

        return util.needs_sensors_update(**data)

    async def _update_controller(self, controller) -> None:
        await controller.sensors_update()


class StateEvent(Event):
    topic = "appliance_state"

    def _parse(self) -> None:
        self.data = parse.lower_dict(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_appliance_state_all(
            connections=self.connections, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        return False


class NearbyUserEvent(Event):
    topic = "nearby_users"

    def _parse(self) -> None:
        self.data = event_parsing.parse_nearby_user_event(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_nearby_users(
            redis=self.connections.redis, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        return False


class ControlEvent(Event):
    topic = "control_target"

    def _parse(self) -> None:
        self.data = event_parsing.parse_control_target(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_control_target_all(
            connections=self.connections, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        try:
            coroutines = {
                "control_mode": cache_util.fetch_control_mode(
                    connections=self.connections, key_arg=self.device_id
                ),
                "automated_demand_response": cache_util.get_automated_demand_response(
                    redis=self.connections.redis, key_arg=self.device_id
                ),
            }
            data = await multi(coroutines)
        except LookupError as e:
            log.info(e)
            return False

        return util.needs_active_control(**data)

    async def _update_controller(self, controller) -> None:
        await controller.control_target_update()


class FeedbackEvent(Event):
    topic = "user_feedback"

    def _parse(self) -> None:
        self.data = event_parsing.parse_feedback_event(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_latest_feedback(
            redis=self.connections.redis, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        try:
            control_mode = await cache_util.fetch_control_mode(
                connections=self.connections, key_arg=self.device_id
            )
        except LookupError as e:
            log.info(e)
            return False

        self.need_feedback_update = util.needs_feedback_control(control_mode)
        self.need_comfort_prediction = True
        return self.need_feedback_update or self.need_comfort_prediction

    async def _update_controller(self, controller) -> None:
        if self.need_comfort_prediction:
            await controller.predict_comfort_for_feedback()
        if self.need_feedback_update:
            await controller.feedback_update()
            await controller.log_feedback_set_point_variation()


class IREvent(Event):
    topic = "irprofile_change"

    def _parse(self) -> None:
        self.data = event_parsing.parse_irprofile_event(self.raw_data)

    async def _get_cache_key(self) -> str:
        appliance_id = await queries.get_appliance(
            self.connections.pool, self.device_id
        )
        if appliance_id:
            return appliance_id
        raise LookupError(f"no appliance_id for {self.device_id}")

    async def cache(self) -> None:
        key_arg = await self._get_cache_key()
        await cache_util.set_ir_feature(
            redis=self.connections.redis, key_arg=key_arg, value=self.data
        )

    async def need_processing(self) -> bool:
        return False


class ModePreferenceEvent(Event):
    topic = "device_mode_preference"

    def _parse(self) -> None:
        self.data = event_parsing.parse_mode_prefs(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_mode_preference_all(
            connections=self.connections, key_arg=self.device_id, value=self.data
        )

    async def need_processing(self) -> bool:
        try:

            mode_pref_key = await cache_util.fetch_mode_preference_key(
                connections=self.connections, key_arg=self.device_id
            )
            coroutines = {
                "last_mode_pref": cache_util.get_last_mode_preference(
                    self.connections.redis, self.device_id, mode_pref_key
                ),
                "new_mode_pref": cache_util.fetch_mode_preference(
                    self.connections, self.device_id, mode_pref_key
                ),
            }

            data = await multi(coroutines)
        except LookupError as e:
            log.info(e)
            return False

        return util.needs_mode_pref_update(**data)

    async def _update_controller(self, controller) -> None:
        await controller.mode_preferences_update()


class ModeFeedbackEvent(Event):
    topic = "mode_feedback"

    def _parse(self) -> None:
        self.data = event_parsing.parse_mode_feedback_event(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_mode_feedback(
            redis=self.connections.redis, key_arg=self.device_id, value=self.data
        )

    async def _update_controller(self, controller) -> None:
        await controller.mode_feedback_update()


class ComfortPredictionEvent(Event):
    topic = "comfort_prediction"
    is_from_event_service = False
    priority = EventPriority.LOW

    async def _update_controller(self, controller) -> None:
        await controller.log_comfort()


class ClimateModelMetricEvent(Event):
    topic = "climate_model_metric"
    is_from_event_service = False
    priority = EventPriority.LOW

    async def need_processing(self) -> bool:
        try:
            control_mode = await cache_util.fetch_control_mode(
                connections=self.connections, key_arg=self.device_id
            )
        except LookupError as e:
            log.info(e)
            return False

        return util.needs_climate_model_metric(control_mode)

    async def _update_controller(self, controller) -> None:
        await controller.log_climate_model_metric()


class AutomatedDemandResponseEvent(Event):
    topic = "automated_demand_response"

    def _parse(self) -> None:
        self.data = event_parsing.parse_automated_demand_response(self.raw_data)

    async def cache(self) -> None:
        await cache_util.set_automated_demand_response(
            redis=self.connections.redis, key_arg=self.device_id, value=self.data
        )

    async def _update_controller(self, controller) -> None:
        await controller.automated_demand_response_update()
