from typing import Dict, List, Tuple

from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from ..prediction import mode_model, prediction_service
from ..prediction.mode_selection import ModeSelection
from ..sample.feature_cache import RedisFeatureData
from ..utils.log_util import get_logger
from ..utils.types import (
    ApplianceState,
    Feedback,
    IRFeature,
    ModeFeedback,
    ModePreferences,
)
from . import prediction_util, util
from .target import Target

log = get_logger(__name__)

QUANTITIES_INCREASING_WITH_SET_POINT = ["temperature", "comfort"]
SORT_PREDICTIONS = False


class Prediction(util.LogMixin):
    def __init__(
        self,
        device_id: str,
        feature_data: RedisFeatureData,
        mode_feedback: ModeFeedback,
        mode_preferences: ModePreferences,
        latest_feedbacks: List[Feedback],
        ir_feature: IRFeature,
        prediction_clients: Dict[str, DealerActor],
        state: ApplianceState,
        target: Target,
    ):
        self.device_id = device_id
        self.feature_data = feature_data
        self.ir_feature = ir_feature
        self.prediction_clients = prediction_clients
        self.state = state
        self.target = target
        self.mode_selection = ModeSelection(
            mode_preferences, ir_feature, mode_feedback, latest_feedbacks, target
        )

    async def get_predictions(self) -> Tuple[List[ApplianceState], List[float]]:
        """Gets predicted states and its comfort scores from available set temperatures.

        Available states are also determined by predicted best mode chosen. It is a hierarchy model.

        Returns:

        """
        # prepare features
        history_features = self.feature_data.get_history_features()
        mode_features = mode_model.make_features(
            history_features,
            self.target.mode_model_quantity,
            self.target.mode_model_target_delta,
        )

        # pick an AC mode
        best_mode = await prediction_service.get_mode_prediction(
            self.prediction_clients["mode_model"],
            self.mode_selection.get_mode_selection(
                self.current_mode,
                history_features["temperature"],
                history_features["temperature_out"],
            ),
            self.target.scaled_mode_model_target_delta,
            mode_features,
            self.feature_data.get_historical_sensor("humidity"),
            self.current_mode,
            self.log,
        )
        # currently ADR penalty is done by changing the target to be reached,
        # in the case of tracking modes the target actually depends on the mode
        # we want to deploy
        self.target.set_predicted_mode(best_mode)

        # predict state given selected mode
        states_to_predict = prediction_util.get_states_to_predict(
            best_mode,
            self.ir_feature,
            self.target.sensors["temperature"],
            self.state["temperature"],
        )

        predictions = await prediction_service.get_climate_prediction(
            self.prediction_clients["climate_model"],
            history_features,
            states_to_predict,
            self.target.climate_model_quantity,
        )

        if SORT_PREDICTIONS:
            assert self.target.quantity is not None
            return sort_predictions(
                states_to_predict,
                await self.target.from_climate_predictions(predictions),
                self.target.quantity,
            )
        return (
            states_to_predict,
            await self.target.from_climate_predictions(predictions),
        )

    @property
    def current_mode(self) -> str:
        return self.state["mode"]


def sort_predictions(
    states: List[ApplianceState], predictions: List[float], quantity: str
) -> Tuple[List[ApplianceState], List[float]]:
    assert is_from_the_same_mode(states)

    if quantity in QUANTITIES_INCREASING_WITH_SET_POINT:
        try:
            return (
                sorted(states, key=lambda x: float(x["temperature"])),
                sorted(predictions),
            )
        except (TypeError, ValueError):
            pass
    return states, predictions


def is_from_the_same_mode(states: List[ApplianceState]) -> bool:
    return (len(set(state["mode"] for state in states)) == 1) and (
        len(set(state["power"] for state in states)) == 1
    )
