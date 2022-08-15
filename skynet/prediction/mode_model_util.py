import math
from ast import literal_eval as make_tuple
from itertools import chain, combinations
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from ..utils import data
from ..utils.enums import Power
from ..utils.types import ModePreferences, ModePrefKey, ModeProbas, ModeSelection

GOOD_ROOM_CONDITION_THRESHOLD = 1
FIRST_LAYER_MODES = ["cool", "heat"]

# some user A/C do not has fan mode nor they only selected cool and dry mode
# if this happens, we consider only using dry mode when if it is predicted
# and scaled_target_delta is positive
MODES_BY_COOLING_STRENGTH_ASCENDING = ["heat", "fan", "dry", "auto", "cool"]
MULTIMODES = sorted(["cool", "heat", "auto", "fan", "dry"])

AC_MODES = {
    ("away", "temperature", "upper"): ["cool"],
    ("away", "temperature", "lower"): ["heat"],
    ("away", "humidity", "upper"): ["cool", "dry"],
}

COOL_MODES = ["cool", "dry", "fan"]
HEAT_MODES = ["heat"]

# When > 1, more uncertain predictions are allowed.
# when < 1 less uncertain predictions are allowed.
# This value is multiplied with with std of last 20 minutes
# of humidity readings.
# Then different between current and mean of last 20 minutes of humidity
# is divided by the product.
HUMIDITY_TUNING_FACTOR = 1.5

MODE_PROBAS_KEYS = ["first_layer_cool", "first_layer_heat"] + MULTIMODES

FIRST_LAYER_CERTAINTY = 0.65
SECOND_LAYER_CERTAINTY = {2: 0.65, 3: 0.40, 4: 0.35, 5: 0.30}
MAKE_MODE_CHANGE_HARDER = 0.05


def certainty_threshold(n_modes_selected):
    return SECOND_LAYER_CERTAINTY[n_modes_selected]


def using_mode_model(mode_selection_length):
    if mode_selection_length == 0:
        raise ValueError("No modes is selected")
    return mode_selection_length > 1


def get_all_possible_mode_selections(modes=MULTIMODES):
    return list(
        chain.from_iterable(combinations(modes, n) for n in range(2, len(modes) + 1))
    )


def get_all_mini_groups(mode_selections):
    all_mini_groups = []
    for mode_selection in mode_selections:
        for _, v in get_mini_groups(mode_selection).items():
            if len(v) > 1 and v not in all_mini_groups:
                all_mini_groups.append(v)
    return all_mini_groups


def get_mini_groups(mode_selection):
    mini_groups = {}
    if set(mode_selection) & set(HEAT_MODES):
        mini_groups.update(
            {"heat": tuple(sorted(set(mode_selection) & (set(HEAT_MODES) | {"auto"})))}
        )
    if set(mode_selection) & set(COOL_MODES):
        mini_groups.update(
            {"cool": tuple(sorted(set(mode_selection) & (set(COOL_MODES) | {"auto"})))}
        )
    if "heat" in mini_groups and "cool" in mini_groups:
        mini_groups.update({"first_layer": tuple(sorted({"cool", "heat"}))})

    return mini_groups


def certainty(
    first_layer_adjustment: bool, proba: ModeProbas, mode_selection: ModeSelection
) -> float:

    if first_layer_adjustment:
        return np.max([proba["first_layer_cool"], proba["first_layer_heat"]])
    return np.max(
        [v for k, v in proba.items() if k in mode_selection and not math.isnan(v)]
    )


def features_quality(
    current_humidity: float, humidities: Iterable[float]
) -> Tuple[bool, float, float]:
    mu = np.mean(humidities)
    humidity_stddev = np.std(humidities)
    current_hum_dev = np.abs(mu - current_humidity)
    features_qualified = current_hum_dev >= (humidity_stddev * HUMIDITY_TUNING_FACTOR)
    return features_qualified


def select_probas(mode_probas: ModeProbas, mode_selection: ModeSelection) -> ModeProbas:
    return {k: max(mode_probas[k], 0) for k in set(mode_probas) & set(mode_selection)}


def sort_modes_by_proba(mode_probas: ModeProbas) -> ModeSelection:
    return sorted(mode_probas, key=mode_probas.get, reverse=True)


def rank_modes(ordered_modes: ModeSelection) -> Dict[str, int]:
    return {m: r for r, m in enumerate(ordered_modes)}


def select_best_mode(
    modes: ModeSelection,
    mode_hist: str,
    scaled_target_delta: float,
    mode_order: ModeSelection = MODES_BY_COOLING_STRENGTH_ASCENDING,
) -> str:

    # delta is: target - current
    # heating => target > current
    #         => target - current > 0
    #         => delta > 0
    heating = scaled_target_delta >= 0
    if heating:
        mode_order = mode_order[::-1]

    # ranking's value greater => more agree with target
    modes_strength_rankings = rank_modes(mode_order)

    # pick the first mode (mode with highest proba)
    # with higher strength ranking than mode_hist
    if mode_hist in modes:
        mode_hist_rank = modes_strength_rankings[mode_hist]
        stronger_modes = [
            mode for mode in modes if modes_strength_rankings[mode] >= mode_hist_rank
        ]
    else:
        stronger_modes = modes

    if stronger_modes:
        return stronger_modes[0]

    return modes[0]


def prediction_valid_threshold(
    first_layer_adjust, features_qualified, num_mode_selection
):
    if first_layer_adjust:
        threshold = FIRST_LAYER_CERTAINTY
    else:
        threshold = certainty_threshold(n_modes_selected=num_mode_selection)

    if not features_qualified:
        threshold += MAKE_MODE_CHANGE_HARDER

    return threshold


def mode_model_adjustment_logic(
    mode_selection: ModeSelection,
    mode_probas: ModeProbas,
    mode_hist: str,
    power_hist: str,
    scaled_target_delta: float,
    current_humidity: float,
    humidities: list,
) -> str:

    best_mode = get_best_mode(
        mode_probas, mode_selection, mode_hist, scaled_target_delta
    )

    first_layer_adjust = get_first_layer_adjust(mode_hist, best_mode, mode_selection)

    can_use_prediction = is_prediction_valid(
        first_layer_adjust, mode_probas, mode_selection, current_humidity, humidities
    )

    if can_we_choose_best_mode(
        mode_hist,
        power_hist,
        mode_selection,
        first_layer_adjust,
        scaled_target_delta,
        can_use_prediction,
    ):
        return best_mode
    return mode_hist


def can_we_choose_best_mode(
    mode_hist: str,
    power_hist: str,
    mode_selection: ModeSelection,
    first_layer_adjust: bool,
    scaled_target_delta: float,
    can_use_prediction: bool,
) -> bool:
    """Determines whether we change to best mode with few regulation rules.

    Args:
        mode_hist:
        power_hist:
        mode_selection:
        first_layer_adjust:
        scaled_target_delta:
        can_use_prediction:

    Returns:

    """

    # existing mode is invalid OR irrelevant
    if mode_hist not in mode_selection or is_the_previous_mode_irrelevant(power_hist):
        return True
    # the opportunities cost of changing mode is small
    if first_layer_adjust and abs(scaled_target_delta) < GOOD_ROOM_CONDITION_THRESHOLD:
        return False
    # else
    return can_use_prediction


def is_the_previous_mode_irrelevant(power_hist: str) -> bool:
    return power_hist == Power.OFF


def is_prediction_valid(
    first_layer_adjust: bool,
    mode_probas: ModeProbas,
    mode_selection: ModeSelection,
    current_humidity: float,
    humidities: Iterable[float],
) -> bool:
    prediction_certainty = certainty(first_layer_adjust, mode_probas, mode_selection)

    features_qualified = features_quality(current_humidity, humidities)
    threshold = prediction_valid_threshold(
        first_layer_adjust, features_qualified, len(mode_selection)
    )

    can_use_prediction = prediction_certainty >= threshold
    return can_use_prediction


def get_first_layer_adjust(
    mode_hist: str, best_mode: str, mode_selection: ModeSelection
) -> bool:
    from_heat_to_cool = mode_hist in HEAT_MODES and best_mode in COOL_MODES
    from_cool_to_heat = mode_hist in COOL_MODES and best_mode in HEAT_MODES
    first_layer_change = from_heat_to_cool or from_cool_to_heat
    cool_picked = bool(set(mode_selection).intersection(COOL_MODES))
    heat_picked = "heat" in mode_selection
    first_layer_adjust = first_layer_change and cool_picked and heat_picked
    return first_layer_adjust


def get_best_mode(
    mode_probas: ModeProbas,
    mode_selection: ModeSelection,
    mode_hist: str,
    scaled_target_delta: float,
) -> str:
    filtered_mode_probas = select_probas(mode_probas, mode_selection)
    modes_sorted_by_proba = sort_modes_by_proba(filtered_mode_probas)
    best_mode = select_best_mode(modes_sorted_by_proba, mode_hist, scaled_target_delta)
    return best_mode


def select_modes(
    device_mode_preference: ModePreferences, mode_pref_key: ModePrefKey
) -> ModeSelection:
    default_modes = AC_MODES.get(mode_pref_key, FIRST_LAYER_MODES)  # type: ignore
    mode_selection = device_mode_preference.get(mode_pref_key, default_modes)
    return mode_selection


def mode_from_probas(probas: ModeProbas) -> Optional[str]:
    return data.argmax_dict(select_probas(probas, mode_selection=MULTIMODES))


class ModeModelParameterGrid:
    def __init__(self, grid_param):
        self.grid_param = grid_param

    def __iter__(self):
        new_params = {}
        for mode_selection, params in self.grid_param.items():
            for p in ParameterGrid(params):
                key = str(mode_selection) + "___" + list(p)[0]
                val = list(p.values())[0]
                if key in new_params:
                    new_params[key].append(val)
                else:
                    new_params[key] = [val]
        for p in ParameterGrid(new_params):
            new_records = []
            new_key = []
            for k, v in p.items():
                new_p = {
                    "mode_selection": make_tuple(k.split("___", 1)[0]),
                    "params": {k.split("___", 1)[1]: v},
                }
                new_records.append(new_p)
                new_key.append(k)
            yield {"mini_params": new_records, "params": tuple(new_key)}


class MultiModesEstimator(BaseEstimator):
    def __init__(
        self,
        mode_selection,
        first_layer_estimator=None,
        second_layer_estimator_cool=None,
        second_layer_estimator_heat=None,
    ):
        self.mini_groups = get_mini_groups(mode_selection)
        self.estimators = {
            "first_layer": first_layer_estimator,
            "cool": second_layer_estimator_cool,
            "heat": second_layer_estimator_heat,
        }
        self.cool_modes = list(self.mini_groups.get("cool", tuple()))
        self.heat_modes = list(self.mini_groups.get("heat", tuple()))

    def insert_estimator(self, layer, estimator):
        self.estimators[layer] = estimator

    def predict_proba(self, X):
        return [
            self._from_layers_proba(*layers_proba)
            for layers_proba in zip(
                *(
                    self._layer_probas(X, layer)
                    for layer in "first_layer cool heat".split()
                )
            )
        ]

    def _from_layers_proba(self, first_layer, second_layer_cool, second_layer_heat):
        first_layer_cool, first_layer_heat = first_layer
        proba = {
            mode: 0.0
            for mode in "first_layer_cool first_layer_heat auto cool dry fan heat".split()
        }
        proba["first_layer_cool"] = first_layer_cool
        proba["first_layer_heat"] = first_layer_heat
        if first_layer_cool >= first_layer_heat:
            proba.update(dict(zip(self.cool_modes, second_layer_cool)))
        if first_layer_cool <= first_layer_heat:
            proba.update(dict(zip(self.heat_modes, second_layer_heat)))
        return proba

    def _layer_probas(self, X, layer):
        if layer == "first_layer":
            if "first_layer" not in self.mini_groups:
                return np.zeros((len(X), 2))
        else:
            if len(self.mini_groups.get(layer, tuple())) < 2:
                return np.ones((len(X), 1))

        pred = self.estimators[layer].predict_proba(X)
        if pred.ndim == 1:
            pred = np.column_stack((pred, 1 - pred))
        return pred

    def predict(self, X):
        probas = self.predict_proba(X)
        return [mode_from_probas(p) for p in probas]
