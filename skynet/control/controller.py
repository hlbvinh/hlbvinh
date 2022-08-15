import asyncio
import time
from datetime import datetime, timedelta, tzinfo
from operator import itemgetter
from typing import Dict, List, Optional, Union

import aredis
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from . import (
    comfort,
    deploy,
    feedback_adjustment,
    managed_manual,
    penalty,
    prediction,
    setting_selection,
    target,
    tracking,
    util,
)
from ..sample import feature_cache
from ..user.sample import local_timestamp
from ..utils import cache_util, ir_feature, metrics
from ..utils.async_util import multi
from ..utils.database import result_format
from ..utils.enums import Power
from ..utils.log_util import get_logger
from ..utils.misc import UpdateCounter, elapsed
from ..utils.monitor import Monitor
from ..utils.progressive_rollout import Experiment
from ..utils.thermo import fix_temperature
from ..utils.types import (
    ApplianceState,
    AutomatedDemandResponse,
    BasicDeployment,
    ComfortPrediction,
    Connections,
    ControlTarget,
    Feedback,
    IRFeature,
    ModeFeedback,
    ModePref,
    ModePreferences,
    Sensors,
    UserIdSet,
)

log = get_logger(__name__)

AWAY_MODE_SIGNAL_DURATION = timedelta(minutes=100)
MODE_FEEDBACK_DURATION = timedelta(minutes=30)


class Controller(util.LogMixin):
    """Business logic and simple routing logic is contained here.

    Attributes:

    """

    def __init__(
        self,
        connections: Connections,
        device_id: str,
        control_target: ControlTarget,
        appliance_id: str,
        ir_feature: IRFeature,
        timezone: tzinfo,
        latest_feedbacks: List[Feedback],
        nearby_users: UserIdSet,
        automated_demand_response: Optional[AutomatedDemandResponse],
        prediction_clients: Dict[str, DealerActor],
        monitor: Monitor,
        update_counter: UpdateCounter,
        testing: bool = True,
    ) -> None:

        self.feature_data = feature_cache.RedisFeatureData(
            connections=connections, device_id=device_id, log_fun=self.log
        )
        self.device_id = device_id
        self.control_target = control_target
        self.connections = connections
        self.prediction_clients = prediction_clients
        self.monitor = monitor
        self.appliance_id = appliance_id
        self.ir_feature = ir_feature
        self.timezone = timezone
        self.latest_feedbacks = latest_feedbacks
        self.nearby_users = nearby_users
        self.automated_demand_response = automated_demand_response
        self.feedback_humidex: Optional[float] = None
        self.force_away_mode_update = False
        self.testing = testing
        self.sensors: Sensors = {}
        self.state: ApplianceState = {}
        self.mode_preferences: ModePreferences = {}
        self.mode_feedback: ModeFeedback = {}
        self.update_counter = update_counter
        self.deploy: deploy.Deploy
        self.prediction: prediction.Prediction
        self.comfort: comfort.Comfort
        self.managed_manual: managed_manual.ManagedManual
        self.feedback_adjustment: feedback_adjustment.FeedbackAdjustment
        self.target: target.Target
        self.tracking: tracking.Tracking
        self.deployed_settings: Optional[BasicDeployment] = None
        self.experiments: List[Experiment] = [
            Experiment(self.connections.redis, self.device_id, "maintain_trend"),
            Experiment(self.connections.redis, self.device_id, "average_humidity"),
            Experiment(self.connections.redis, self.device_id, "maintain_threshold"),
        ]

    @classmethod
    async def from_redis(
        cls,
        connections: Connections,
        device_id: str,
        prediction_clients: Dict[str, DealerActor],
        monitor: Monitor,
        update_counter: UpdateCounter,
        testing: bool = True,
    ) -> "Controller":
        """Queries required information and then instantiates a controller for the specified device.

        Args:
            connections:
            device_id:
            prediction_clients:
            monitor:
            update_counter:
            testing:

        Returns:

        """
        data_queries = {
            "control_target": cache_util.fetch_control_target(connections, device_id),
            "sensors": cache_util.get_sensors(
                redis=connections.redis, key_arg=device_id
            ),
            "state": cache_util.fetch_appliance_state(connections, device_id),
            "mode_preferences": cache_util.fetch_mode_preferences(
                connections, device_id
            ),
            "timezone": cache_util.fetch_timezone(connections, device_id),
            "mode_feedback": cache_util.fetch_mode_feedback(connections, device_id),
            "latest_feedbacks": cache_util.fetch_latest_feedbacks(
                connections, device_id
            ),
            "nearby_users": cache_util.fetch_nearby_users(connections, device_id),
            "automated_demand_response": cache_util.get_automated_demand_response(
                redis=connections.redis, key_arg=device_id
            ),
        }
        controller_data = await multi(data_queries)

        controller_data["timezone"] = result_format.parse_timezone(
            controller_data["timezone"]
        )
        appliance_id = controller_data["state"]["appliance_id"]
        controller_data["ir_feature"] = await cache_util.fetch_ir_feature(
            connections, appliance_id
        )
        self = await cls.create(
            connections=connections,
            device_id=device_id,
            appliance_id=appliance_id,
            prediction_clients=prediction_clients,
            monitor=monitor,
            update_counter=update_counter,
            testing=testing,
            **controller_data,
        )

        return self

    @classmethod
    async def create(
        cls,
        connections: Connections,
        device_id: str,
        appliance_id: str,
        ir_feature: IRFeature,
        control_target: ControlTarget,
        mode_feedback: ModeFeedback,
        sensors: Sensors,
        state: ApplianceState,
        mode_preferences: List[ModePref],
        timezone: tzinfo,
        latest_feedbacks: List[Feedback],
        nearby_users: UserIdSet,
        automated_demand_response: Optional[AutomatedDemandResponse],
        prediction_clients: Dict[str, DealerActor],
        monitor: Monitor,
        update_counter: UpdateCounter,
        testing: bool = True,
    ) -> "Controller":

        if state["power"] == Power.ON:
            if util.needs_managed_manual(
                control_target["quantity"], automated_demand_response
            ):
                control_target["quantity"] = "managed_manual"

        self = cls(
            connections,
            device_id,
            control_target,
            appliance_id,
            ir_feature,
            timezone,
            latest_feedbacks,
            nearby_users,
            automated_demand_response,
            prediction_clients,
            monitor,
            update_counter,
            testing,
        )

        await multi([experiment.update() for experiment in self.experiments])
        self.set_sensors(sensors)
        self.set_state(state, update=False)
        self.set_mode_preferences(mode_preferences)
        self.set_mode_feedback(mode_feedback)

        # variables (functionalities) are strongly coupled here.
        # Is it inevitable?
        self.deploy = deploy.Deploy(
            self.connections,
            self.device_id,
            self.appliance_id,
            self.state,
            self.ir_feature,
            self.control_target,
            self.latest_feedback,
            self.testing,
            self.monitor,
        )
        self.comfort = comfort.Comfort(
            self.device_id,
            self.timezone,
            self.prediction_clients,
            self.feature_data,
            self.nearby_users,
            self.latest_feedbacks,
            self.connections,
            self.experiments,
        )
        self.managed_manual = managed_manual.ManagedManual(
            self.device_id, self.state, self.connections
        )
        self.target = target.Target(
            self.device_id,
            self.sensors,
            self.comfort,
            self.managed_manual,
            self.control_target,
            self.automated_demand_response,
            self.experiments,
            self.connections.redis,
        )

        self.prediction = prediction.Prediction(
            self.device_id,
            self.feature_data,
            self.mode_feedback,
            self.mode_preferences,
            self.latest_feedbacks,
            self.ir_feature,
            self.prediction_clients,
            self.state,
            self.target,
        )

        self.feedback_adjustment = feedback_adjustment.FeedbackAdjustment(
            self.state["temperature"], self.latest_feedback
        )

        self.tracking = tracking.Tracking(
            self.state,
            self.device_id,
            self.connections,
            self.prediction,
            self.target,
            self.feedback_adjustment,
            self.control_target,
            self.experiments,
        )
        await multi([self.load_state(), self.update_feedback_humidex()])

        return self

    async def sensors_update(self) -> None:
        self.log("updating state with new sensor readings")
        await self.update_state_wrap()

    async def control_target_update(self) -> None:
        self.log("control target", control_target=self.control_target)
        last_control_mode = await cache_util.get_last_control_mode(
            redis=self.connections.redis, key_arg=self.device_id
        )
        self.log(
            f"updating with control mode {last_control_mode} -> {self.target.control_mode}"
        )
        self.tracking.penalty_factor = penalty.control_target_change_penalty_factor(
            last_control_mode, self.target.control_mode
        )
        await self.update_state_wrap()

    async def update_state_wrap(self):
        if self.ir_feature is None:
            self.log("ir feature set to None, do not perform state update")
            return

        tic = time.perf_counter()
        await self.target.update_value()
        try:
            await self.update_state()
        except (asyncio.TimeoutError, aredis.TimeoutError) as exc:
            self.log(exc, level="info")
        except ir_feature.NoButtonError as exc:
            self.log(exc, level="info")
        except ValueError as exc:
            log.exception(exc)
            self.log(exc, level="error")
        except Exception as exc:
            self.log(
                "{}, uncaught exception in update_state, traceback "
                "following".format(exc),
                level="error",
            )
            log.exception(exc)
        toc = time.perf_counter()
        self.log("state_update", state_update_ms=1000 * (toc - tic))
        self.update_counter.update()

    async def update_state(self):
        """Updates the state.

        It performs a few rest calls and if necessary updates the state of the
        AC. Then schedules a control update of the controller.

        Returns:

        """
        await cache_util.set_last_state_update(
            redis=self.connections.redis,
            key_arg=self.device_id,
            value=datetime.utcnow(),
        )

        log_state = self.state.copy()
        try:
            log_state["temperature_float"] = float(log_state["temperature"])
        except (KeyError, TypeError, ValueError):
            pass
        self.log("current appliance state", appliance_state=log_state)

        # start routing and get AC setting for deployment
        ai_suggested_settings = None  # "None" means no deployment will be made
        if self.target.control_mode == "off" or self.target.turn_off_appliance:
            if self.state["power"] != Power.OFF:
                ai_suggested_settings = self.deploy.off_deployment

        elif self.target.control_mode == "away":
            ai_suggested_settings = await self.away_mode_update_settings()
            self.log("away mode settings", away_mode_settings=ai_suggested_settings)

        elif self.target.control_mode in util.TRACKING_QUANTITIES:
            ai_suggested_settings = await self.tracking.get_best_setting()
            self.deployed_settings = ai_suggested_settings
            self.log(
                f"tracking mode {self.target.control_mode} settings",
                best_settings_for_update=ai_suggested_settings,
            )

        elif self.target.control_mode == "managed_manual":
            ai_suggested_settings = await self.get_managed_manual_settings()

        elif self.target.control_mode == "manuel":
            # won't make a deployment in manuel mode
            # exception: Daikin Urusara model
            #   link for reference: https://echogroup.atlassian.net/browse/DMW-186 (and DMW-185)
            ai_suggested_settings = None

        if ai_suggested_settings is not None:
            await self.deploy.deploy_settings(ai_suggested_settings)

    async def away_mode_update_settings(self) -> Optional[BasicDeployment]:
        # assembling away mode's state
        new = self.state["created_on"] < self.control_target["created_on"]
        away_mode_state = {
            "quantity": self.target.quantity,
            "threshold_type": self.target.threshold_type,
            "threshold": self.target.threshold,
            "current": self.target.current_value,
        }
        condition = util.away_mode_conditions(**away_mode_state)  # type: ignore
        timed_out = elapsed(
            self.last_deployment, AWAY_MODE_SIGNAL_DURATION
        )  # timed out for what?
        away_mode_state["condition"] = condition
        away_mode_state["new"] = new

        self.log("away mode state", away_mode_state=away_mode_state)

        # get actions by given state
        action = util.away_mode_action(self.is_ac_on, condition, timed_out, new)
        if action == "update" or self.force_away_mode_update:
            return await self.get_away_mode_settings()

        if action == "off":
            return self.deploy.off_deployment
        return None

    async def get_away_mode_settings(self) -> BasicDeployment:
        states, predictions = await self.prediction.get_predictions()

        assert self.target.threshold_type is not None
        best_setting = setting_selection.select_away_mode_setting(
            predictions, states, self.target.threshold_type, self.target.quantity
        )

        return BasicDeployment(**best_setting)

    async def get_managed_manual_settings(self) -> Optional[BasicDeployment]:
        self.target.set_predicted_mode(await self.managed_manual.get_mode())
        assert self.target.target_value is not None
        deployment = await self.managed_manual.get_deployment(
            self.target.target_value, self.ir_feature
        )
        self.log("control_state", control_state=self.target.control_state)
        return deployment

    async def update_feedback_humidex(self) -> None:
        if self.has_feedback:
            self.feedback_humidex = await cache_util.fetch_feedback_humidex(
                self.connections, self.device_id, self.latest_feedback["created_on"]
            )

    async def log_comfort(self) -> None:
        comfort_pred = await self.comfort.predict_adjusted()
        self.log(
            "comfort prediction",
            comfort_prediction={
                "prediction": comfort_pred,
                "absolute_prediction": abs(comfort_pred),
                "control_mode": self.target.control_mode,
                "quantity": self.target.quantity,
                "minute_since_last_control_target": self.target.minute_since_last_control_target,
            },
        )

    async def log_climate_model_metric(self) -> None:
        await metrics.log_climate_model_metric(
            self.prediction_clients["climate_model"],
            self.feature_data.get_history_features(),
            self.state,
            self.sensors,
            self.device_id,
            self.connections.redis,
            self.log,
        )

    @property
    def latest_feedback(self) -> Feedback:
        latest_feedbacks_sorted_on_created_on = sorted(
            self.latest_feedbacks, key=itemgetter("created_on")
        )
        if latest_feedbacks_sorted_on_created_on:
            return latest_feedbacks_sorted_on_created_on[-1]
        return {}

    @property
    def has_feedback(self) -> bool:
        return bool(self.latest_feedback)

    async def feedback_update(self):
        self.feedback_adjustment.is_feedback_update = True
        self.log("feedback", feedback=self.latest_feedback)

        self.log("feedback_humidex", feedback_humidex=self.feedback_humidex)

        self.log("updating state with new feedback")
        self.tracking.penalty_factor = penalty.FEEDBACK_UPDATE_FACTOR
        await self.update_state_wrap()

    async def predict_comfort_for_feedback(self) -> None:
        predicted_comfort = await self.comfort.predict()
        await multi(
            [
                self.save_comfort_prediction(predicted_comfort),
                self.log_comfort_model_metric(predicted_comfort),
            ]
        )

    async def automated_demand_response_update(self):
        self.log("automated_demand_response", adr=self.automated_demand_response)
        self.tracking.penalty_factor = penalty.ADR_UPDATE_FACTOR
        await self.update_state_wrap()

    async def save_comfort_prediction(self, predicted_comfort) -> None:
        """cache feedback data needed by the comfort model to adjust the comfort
        prediction"""
        assert self.feedback_humidex is not None
        assert self.has_feedback
        feedback_prediction = ComfortPrediction(
            self.latest_feedback["feedback"],
            predicted_comfort,
            self.feedback_humidex,
            self.latest_feedback["created_on"],
        )
        await cache_util.set_comfort_prediction(
            redis=self.connections.redis,
            key_arg=(self.device_id, self.latest_feedback["user_id"]),
            value=feedback_prediction,
        )

    async def log_comfort_model_metric(self, predicted_comfort: float) -> None:
        assert self.has_feedback
        true_feedback = self.latest_feedback["feedback"]
        self.log(
            "feedback prediction",
            feedback={
                "comfort_prediction": predicted_comfort,
                "comfort_feedback": true_feedback,
                "comfort_error": predicted_comfort - true_feedback,
                "comfort_absolute_error": abs(predicted_comfort - true_feedback),
                "hour_of_day": local_timestamp(
                    self.timezone, self.latest_feedback["created_on"]
                ).hour,
            },
        )

    async def log_feedback_set_point_variation(self) -> None:
        if self.deployed_settings:
            try:
                variation = abs(
                    fix_temperature(self.state["temperature"])
                    - fix_temperature(self.deployed_settings.temperature)
                )
                self.log(
                    "feedback set point variation",
                    feedback=self.latest_feedback["feedback"],
                    variation=variation,
                    quantity=self.target.quantity,
                )
            except Exception:
                pass

    def set_mode_feedback(self, mode_feedback: ModeFeedback) -> None:
        if not elapsed(mode_feedback.get("created_on"), MODE_FEEDBACK_DURATION):
            self.mode_feedback = mode_feedback
        else:
            self.mode_feedback = {}

    async def mode_feedback_update(self) -> None:
        self.log(
            "mode feedback",
            feedback=self.mode_feedback,
            current_mode=self.state["mode"],
        )
        self.log("updating state with new mode feedback")
        self.tracking.penalty_factor = penalty.FEEDBACK_UPDATE_FACTOR
        await self.update_state_wrap()

    def set_mode_preferences(self, mode_prefs: Union[List[ModePref], ModePref]) -> None:

        if isinstance(mode_prefs, ModePref):
            mode_prefs = [mode_prefs]

        for mode_pref in mode_prefs:
            self.mode_preferences[mode_pref.key] = mode_pref.modes

    async def mode_preferences_update(self) -> None:
        self.log("updating with mode_preferences change")
        self.force_away_mode_update = self.target.control_mode == "away"
        await self.update_state_wrap()

    def set_sensors(self, sensors: Sensors) -> None:
        self.sensors = sensors

    def set_state(self, state: ApplianceState, update: bool = True) -> None:
        self.state = state
        if update:
            self.log("appliance_state", appliance_state=state)

    @property
    def last_deployment(self) -> datetime:  # bad naming
        return self.state["created_on"]

    async def load_state(self) -> None:
        coroutines = [self.feature_data.load_state()]
        if self.is_ac_on and self.target.quantity in util.TRACKING_QUANTITIES:
            coroutines.extend(
                [
                    self.tracking.deviation_tracker.load_state(),
                    self.tracking.maintain.load_state(),
                ]
            )

        await multi(coroutines)

    async def store_state(self) -> None:
        if self.target.quantity in util.TRACKING_QUANTITIES:
            await multi(
                [
                    self.tracking.deviation_tracker.store_state(),
                    self.tracking.maintain.store_state(),
                ]
            )

    @property
    def is_ac_on(self) -> bool:
        return self.state["power"] == Power.ON
