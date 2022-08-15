import numpy as np
import pytest

from ...prediction import mode_model_util
from ...prediction.estimators.mode_model import get_mini_pipeline
from ...utils.enums import Power
from ...utils.types import ModePreferences, ModePrefKey

np.random.seed(7252014)


def random_num_mode_selection():
    return np.random.randint(2, len(mode_model_util.MULTIMODES) + 1, dtype=int)


def get_trained_pipeline(mode_selection, X, y):
    feat = list(X.columns)
    est = get_mini_pipeline(features=feat)
    mask = np.in1d(y, mode_selection)
    est.fit(X[mask], y[mask])
    return est


def test_select_modes():
    """

    get_selected_modes(ir_feature, device_mode_preference, control_mode,
                       quantity)

    """

    mode_selection1: ModePreferences = {
        ModePrefKey("comfort", "comfort", None): ["cool", "auto"],
        ModePrefKey("away", "temperature", "upper"): ["cool"],
        ModePrefKey("away", "humidity", "upper"): ["dry", "fan"],
    }
    sel = mode_model_util.select_modes

    test_mode_selection1 = sel(mode_selection1, ModePrefKey("comfort", "comfort", None))
    assert sorted(["cool", "auto"]) == sorted(test_mode_selection1)

    test_mode_selection2 = sel(
        mode_selection1, ModePrefKey("away", "temperature", "upper")
    )
    assert sorted(test_mode_selection2) == ["cool"]

    test_mode_selection3 = sel(
        mode_selection1, ModePrefKey("away", "humidity", "upper")
    )
    assert sorted(test_mode_selection3) == sorted(["dry", "fan"])

    test_mode_selection4 = sel(
        mode_selection1, ModePrefKey("temperature", "temperature")
    )
    assert sorted(test_mode_selection4) == sorted(["cool", "heat"])

    test_mode_selection5 = sel({}, ModePrefKey("temperature", "temperature", None))
    assert sorted(test_mode_selection5) == sorted(["heat", "cool"])

    test_mode_selection6 = sel({}, ModePrefKey("away", "temperature", "lower"))
    assert sorted(test_mode_selection6) == sorted(["heat"])


def test_selected_mode_filter():
    test_probas = {
        "cool": 0.1,
        "heat": 0.05,
        "fan": 0.2,
        "dry": 0.5,
        "auto": 0.25,
        "first_layer_cool": 0.6,
        "first_layer_heat": 0.4,
    }

    def order(selection):
        probas = mode_model_util.select_probas(test_probas, selection)
        return mode_model_util.sort_modes_by_proba(probas)

    selection = ["cool", "auto", "fan"]
    assert order(selection) == ["auto", "fan", "cool"]

    selection = ["cool", "dry"]
    assert order(selection) == ["dry", "cool"]


def test_humidity_features_quality():
    # create periodic function that oscillate between 40 to 80
    # each step is pi/6
    quality = mode_model_util.features_quality

    test_humidities = 20 * np.sin(np.arange(40) * 2 * np.pi / 6) + 60

    # pick a current humidity values far from the historical humidities
    test_current_humidity = 30.0
    feat_quality = quality(test_current_humidity, test_humidities)
    assert feat_quality

    # pick a current humidity values far from the historical humidities
    test_current_humidity_1 = 40.0
    feat_quality = quality(test_current_humidity_1, test_humidities)
    assert not feat_quality


def test_probas_certainty():

    certainty = mode_model_util.certainty

    test_proba_1 = {"cool": 0.6, "dry": 0.1, "fan": 0.1, "auto": 0.2}
    mode_selection_1 = ["auto", "cool", "dry", "fan"]
    first_layer_adj_1 = False
    assert certainty(first_layer_adj_1, test_proba_1, mode_selection_1) == 0.6

    test_proba_2 = {
        "first_layer_cool": 0.9,
        "first_layer_heat": 0.1,
        "cool": 1.0,
        "heat": 0.0,
    }
    mode_selection_2 = ["cool", "heat"]
    first_layer_adj_2 = False
    assert certainty(first_layer_adj_2, test_proba_2, mode_selection_2) == 1.0

    test_proba_3 = {
        "first_layer_cool": 0.9,
        "first_layer_heat": 0.1,
        "cool": 1.0,
        "heat": 0.0,
    }
    mode_selection_3 = ["cool", "heat"]
    first_layer_adj_3 = True
    assert certainty(first_layer_adj_3, test_proba_3, mode_selection_3) == 0.9

    test_proba_4 = {
        "first_layer_cool": 0.9,
        "first_layer_heat": 0.1,
        "cool": 1.0,
        "heat": 0.0,
    }
    mode_selection_4 = ["cool", "dry"]
    first_layer_adj_4 = False
    assert certainty(first_layer_adj_4, test_proba_4, mode_selection_4) == 1.0


def test_get_all_possible_mode_selections():
    all_possible = mode_model_util.get_all_possible_mode_selections([1, 2, 3])
    assert sorted(all_possible) == [(1, 2), (1, 2, 3), (1, 3), (2, 3)]


@pytest.mark.parametrize(
    "args, result",
    [
        (
            ("cool", "heat"),
            {"cool": ("cool",), "first_layer": ("cool", "heat"), "heat": ("heat",)},
        ),
        (
            ("cool", "fan", "heat"),
            {
                "cool": ("cool", "fan"),
                "first_layer": ("cool", "heat"),
                "heat": ("heat",),
            },
        ),
        (
            ("auto", "cool", "fan", "heat"),
            {
                "cool": ("auto", "cool", "fan"),
                "first_layer": ("cool", "heat"),
                "heat": ("auto", "heat"),
            },
        ),
    ],
)
def test_get_mini_groups(args, result):
    assert mode_model_util.get_mini_groups(args) == result


@pytest.mark.parametrize(
    "run",
    [
        {
            "mode_selection": ("auto", "cool", "heat"),
            "first_layer": ("cool", "heat"),
            "cool_layer": ("auto", "cool"),
            "heat_layer": ("auto", "heat"),
            "do_first_layer": True,
            "do_cool": True,
            "do_heat": True,
            "cool_modes": ["auto", "cool"],
            "heat_modes": ["auto", "heat"],
        },
        {
            "mode_selection": ("auto", "cool"),
            "first_layer": None,
            "cool_layer": ("auto", "cool"),
            "heat_layer": None,
            "do_first_layer": False,
            "do_cool": True,
            "do_heat": False,
            "cool_modes": ["auto", "cool"],
            "heat_modes": None,
        },
        {
            "mode_selection": ("fan", "heat"),
            "first_layer": ("cool", "heat"),
            "cool_layer": ("fan",),
            "heat_layer": None,
            "do_first_layer": True,
            "do_cool": False,
            "do_heat": False,
            "cool_modes": ["fan"],
            "heat_modes": None,
        },
        {
            "mode_selection": ("auto", "heat"),
            "first_layer": None,
            "cool_layer": None,
            "heat_layer": ("auto", "heat"),
            "do_first_layer": False,
            "do_cool": False,
            "do_heat": True,
            "cool_modes": None,
            "heat_modes": ["auto", "heat"],
        },
        {
            "mode_selection": ("heat",),
            "first_layer": None,
            "cool_layer": None,
            "heat_layer": None,
            "do_first_layer": False,
            "do_cool": False,
            "do_heat": False,
            "cool_modes": None,
            "heat_modes": ["heat"],
        },
        {
            "mode_selection": ("cool",),
            "first_layer": None,
            "cool_layer": None,
            "heat_layer": None,
            "do_first_layer": False,
            "do_cool": False,
            "do_heat": False,
            "cool_modes": None,
            "heat_modes": ["cool"],
        },
    ],
)
def test_multi_mode_estimator(mode_features, mode_targets, run):
    def gen_test_run_ests(
        mode_selection, first_layer, cool_layer, heat_layer, feat, tar
    ):
        if first_layer is not None:
            first_est = get_trained_pipeline(first_layer, feat, tar)
        else:
            first_est = None

        if heat_layer is not None:
            heat_est = get_trained_pipeline(("auto", "heat"), feat, tar)
        else:
            heat_est = None

        if cool_layer is not None:
            cool_est = get_trained_pipeline(("auto", "cool"), feat, tar)
        else:
            cool_est = None

        multi_mode_est = mode_model_util.MultiModesEstimator(
            mode_selection=mode_selection,
            first_layer_estimator=first_est,
            second_layer_estimator_cool=cool_est,
            second_layer_estimator_heat=heat_est,
        )

        return multi_mode_est

    multi_mode_est = gen_test_run_ests(
        run["mode_selection"],
        run["first_layer"],
        run["cool_layer"],
        run["heat_layer"],
        mode_features,
        mode_targets,
    )

    probas = multi_mode_est.predict_proba(mode_features)

    for proba in probas:
        assert (
            np.sum([v for k, v in proba.items() if k in run["mode_selection"]]) == 1.0
        )


def test_using_mode_model():
    runs = [(4, True), (5, True), (1, False), (3, True), (2, True)]

    for arg, res in runs:
        assert mode_model_util.using_mode_model(arg) == res

    try:
        mode_model_util.using_mode_model(0)
    except ValueError as e:
        message = e.args[0]
        assert message == "No modes is selected"


@pytest.mark.parametrize(
    "kwargs, result",
    [
        (
            {
                "modes": ["dry", "cool", "auto"],
                "mode_hist": "cool",
                "scaled_target_delta": 1.0,
                "mode_order": mode_model_util.MODES_BY_COOLING_STRENGTH_ASCENDING,
            },
            "dry",
        ),
        (
            {
                "modes": ["dry", "cool"],
                "mode_hist": "cool",
                "scaled_target_delta": 1.0,
                "mode_order": mode_model_util.MODES_BY_COOLING_STRENGTH_ASCENDING,
            },
            "dry",
        ),
        (
            {
                "modes": ["dry", "cool"],
                "mode_hist": "cool",
                "scaled_target_delta": 1.0,
                "mode_order": ["heat", "fan", "auto", "cool", "dry"],
            },
            "cool",
        ),
        (
            {
                "modes": ["heat", "cool"],
                "mode_hist": "cool",
                "scaled_target_delta": -1.0,
                "mode_order": mode_model_util.MODES_BY_COOLING_STRENGTH_ASCENDING,
            },
            "cool",
        ),
        (
            {
                "modes": ["cool", "auto", "dry", "fan"],
                "mode_hist": "heat",
                "scaled_target_delta": 1.0,
                "mode_order": mode_model_util.MODES_BY_COOLING_STRENGTH_ASCENDING,
            },
            "cool",
        ),
    ],
)
def test_select_best_mode(kwargs, result):
    assert mode_model_util.select_best_mode(**kwargs) == result


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {
                "mode_selection": ["cool", "dry", "auto"],
                "mode_probas": {"cool": 0.8, "dry": 0.1, "auto": 0.1},
                "mode_hist": "fan",
                "power_hist": Power.ON,
                "scaled_target_delta": -5,
                "current_humidity": 85,
                "humidities": list(range(80, 90)),
            },
            "cool",
            id="select highest ranked mode",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "fan"],
                "mode_probas": {"fan": 0.9, "cool": 0.1},
                "mode_hist": "dry",
                "power_hist": Power.ON,
                "scaled_target_delta": 0.01,
                "current_humidity": 85,
                "humidities": list(range(80, 90)),
            },
            "fan",
            id="if scaled_target_delta is close to 0 ok to use fan to maintain the temperature",
        ),
        pytest.param(
            {
                "mode_selection": mode_model_util.MULTIMODES,
                "mode_probas": {
                    "cool": 0.24,
                    "auto": 0.19,
                    "heat": 0.19,
                    "dry": 0.19,
                    "fan": 0.19,
                },
                "mode_hist": "dry",
                "power_hist": Power.ON,
                "scaled_target_delta": -2,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "dry",
            id="if no probability is high enough, keep the current mode",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "heat"],
                "mode_probas": {
                    "first_layer_cool": 0.9,
                    "first_layer_heat": 0.1,
                    "cool": 1.0,
                },
                "mode_hist": "heat",
                "power_hist": Power.ON,
                "scaled_target_delta": 0.1,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "heat",
            id="when in heat with wrong predictions towards "
            "cool but heat is working well, keep heating",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "heat"],
                "mode_probas": {
                    "first_layer_cool": 0.1,
                    "first_layer_heat": 0.90,
                    "heat": 1.0,
                },
                "mode_hist": "cool",
                "power_hist": Power.ON,
                "scaled_target_delta": 0.1,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "cool",
            id="when in cool with wrong predictions towards "
            "heat but cool is working well, keep cooling",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "fan"],
                "mode_probas": {"fan": 0.1, "cool": 0.9},
                "mode_hist": "fan",
                "power_hist": Power.ON,
                "scaled_target_delta": -0.1,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "cool",
            id="when not on a heat-cool first layer transition, "
            "just pick the highest mode prediction",
        ),
        pytest.param(
            {
                "mode_selection": ["cool", "dry"],
                "mode_probas": {"dry": 0.4, "cool": 0.6},
                "mode_hist": "fan",
                "power_hist": Power.ON,
                "scaled_target_delta": -3.5,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "cool",
            id="when the current mode is not selected, and the probas are "
            "not high enough, still pick the best mode",
        ),
        pytest.param(
            {
                "mode_selection": ["heat", "cool"],
                "mode_probas": {
                    "first_layer_cool": 0.6,
                    "first_layer_heat": 0.4,
                    "cool": 1.0,
                },
                "mode_hist": "heat",
                "power_hist": Power.OFF,
                "scaled_target_delta": -1,
                "current_humidity": 85,
                "humidities": [85] * 20,
            },
            "cool",
            id="when the AC is just switched on then we should choose the mode "
            "with the highest probability",
        ),
    ],
)
def test_mode_model_adjustment_logic(kwargs, result):
    assert mode_model_util.mode_model_adjustment_logic(**kwargs) == result


def test_mode_mode_parameter_grid():
    grid_params = {("cool", "heat"): {"estimator__fit__eta0": [0.01, 0.05]}}

    outs = [
        {
            "mini_params": [
                {
                    "mode_selection": ("cool", "heat"),
                    "params": {"estimator__fit__eta0": 0.01},
                }
            ],
            "params": ("('cool', 'heat')___estimator__fit__eta0",),
        },
        {
            "mini_params": [
                {
                    "mode_selection": ("cool", "heat"),
                    "params": {"estimator__fit__eta0": 0.05},
                }
            ],
            "params": ("('cool', 'heat')___estimator__fit__eta0",),
        },
    ]
    grids = mode_model_util.ModeModelParameterGrid(grid_params)
    for param, out in zip(grids, outs):
        assert param == out


@pytest.mark.parametrize("_n_times", range(10))
def test_prediction_valid_threshold(_n_times):

    # test when first_adjust is true, threshold depends on
    # features_qualified while num_mode_selection is a don't-care
    first_layer_adjust_1 = True
    features_qualified_1 = True
    num_mode_selection_1 = random_num_mode_selection()
    threshold_1 = mode_model_util.prediction_valid_threshold(
        first_layer_adjust_1, features_qualified_1, num_mode_selection_1
    )
    assert threshold_1 == mode_model_util.FIRST_LAYER_CERTAINTY

    features_qualified_2 = False
    num_mode_selection_2 = random_num_mode_selection()
    threshold_2 = mode_model_util.prediction_valid_threshold(
        first_layer_adjust_1, features_qualified_2, num_mode_selection_2
    )
    assert threshold_2 == threshold_1 + mode_model_util.MAKE_MODE_CHANGE_HARDER

    # test when first_adjust is false, threshold depends on both
    # features_qualified and num_mode_selection
    first_layer_adjust_3 = False
    features_qualified_3 = True
    num_mode_selection_3 = random_num_mode_selection()
    threshold_3 = mode_model_util.prediction_valid_threshold(
        first_layer_adjust_3, features_qualified_3, num_mode_selection_3
    )
    true_threshold = mode_model_util.SECOND_LAYER_CERTAINTY[num_mode_selection_3]
    assert threshold_3 == true_threshold

    features_qualified_4 = False
    true_threshold = mode_model_util.SECOND_LAYER_CERTAINTY[num_mode_selection_3]
    threshold_4 = mode_model_util.prediction_valid_threshold(
        first_layer_adjust_3, features_qualified_4, num_mode_selection_3
    )
    true_threshold += mode_model_util.MAKE_MODE_CHANGE_HARDER
    assert threshold_4 == true_threshold


@pytest.mark.parametrize(
    "kwargs, result",
    [
        pytest.param(
            {
                "mode_hist": "fan",
                "power_hist": Power.ON,
                "mode_selection": ["cool", "dry", "auto"],
                "first_layer_adjust": False,
                "scaled_target_delta": -5,
                "can_use_prediction": True,
            },
            True,
            id="choose the best mode if previous mode is not in mode_selection",
        ),
        pytest.param(
            {
                "mode_hist": "heat",
                "power_hist": Power.OFF,
                "mode_selection": ["cool", "dry", "auto", "heat"],
                "first_layer_adjust": True,
                "scaled_target_delta": -5,
                "can_use_prediction": True,
            },
            True,
            id="choose the best mode if previously power was off",
        ),
        pytest.param(
            {
                "mode_hist": "heat",
                "power_hist": Power.ON,
                "mode_selection": ["cool", "dry", "auto", "heat"],
                "first_layer_adjust": True,
                "scaled_target_delta": -0.5,
                "can_use_prediction": True,
            },
            False,
            id="choose previous mode if scaled_target_delta is not that bad",
        ),
        pytest.param(
            {
                "mode_hist": "heat",
                "power_hist": Power.ON,
                "mode_selection": ["cool", "dry", "auto", "heat"],
                "first_layer_adjust": True,
                "scaled_target_delta": -5,
                "can_use_prediction": True,
            },
            True,
            id="let can_use_prediction choose the mode if the conditions are bad and we are "
            "predicting some mode other than current mode",
        ),
    ],
)
def test_can_we_choose_best_mode(kwargs, result):
    assert mode_model_util.can_we_choose_best_mode(**kwargs) == result
