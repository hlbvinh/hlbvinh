from typing import List, Optional

import numpy as np

from .. import utils
from ..utils import log_util
from ..utils.enums import Power
from ..utils.types import ApplianceState
from .climate_model import QUANTITY_MAP, TRAIN_USING_RELATIVE_TARGET, ClimateModel

log = log_util.get_logger(__name__)


def generate_on_signals():
    """Generate all AC states where power is 'on'.

    Returns
    -------
    list of dict
        all possible combinations of AC state for alpha trial
    """
    s = []
    for m in utils.config.ALL_SIGNALS["on"]:
        for t in utils.config.ALL_SIGNALS["on"][m]:
            s.append({"power": Power.ON, "mode": m, "temperature": t})
    return s


class Predictor:
    @classmethod
    def load(cls, storage):
        model = storage.load(ClimateModel.get_storage_key())
        return cls(model)

    def __init__(self, model):
        self.model = model

    def predict(
        self,
        history_features,
        states: List[ApplianceState],
        quantity: Optional[str] = None,
    ) -> List:
        features = [s.copy() for s in states]

        for f in features:
            # "fixing" a few fields
            f["temperature_set"] = f.pop("temperature")
            f.update(history_features)
            if f["power"] == Power.OFF:
                f["mode"] = "off"
        predictions = self.model.predict(features)
        if TRAIN_USING_RELATIVE_TARGET:
            predictions = np.add(
                predictions,
                np.array(
                    [history_features.get(quantity, 0) for quantity in QUANTITY_MAP]
                ),
            )
        if quantity is not None:
            predictions = predictions[:, QUANTITY_MAP[quantity]]

        return predictions.tolist()
