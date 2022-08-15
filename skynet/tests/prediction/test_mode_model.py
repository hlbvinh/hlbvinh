import math

import numpy as np
import pytest

from ...prediction import climate_model, mode_model, mode_model_util
from ...prediction.climate_model import QUANTITIES
from ...prediction.estimators.mode_model import get_mini_weights

np.random.seed(1)

MODEL_TYPE = "mode_model"


def assert_proba(p1, p2):
    assert len(p1) == len(p2)
    for k, v in p1.items():
        assert k in p2
        if not math.isnan(v):
            assert np.isclose(p2[k], v)


def test_make_mode_model_dataset(mode_features, mode_targets):
    assert len(mode_features) == len(mode_targets)
    target_values = set(mode_targets)
    assert "other" not in target_values
    assert "cool" in target_values
    assert "heat" in target_values


@pytest.fixture
def feat():
    return {"a": "b"}


def test_make_features(feat):
    orig = feat.copy()
    value = 1.0
    for q in mode_model.TARGET_FEATURE_COLUMNS:
        f = mode_model.make_features(feat, q, value)

        # test input unchanged
        assert feat == orig

        # test original features in output
        assert feat["a"] == f["a"]

        # test target value added to features
        assert f["target"] == value


def test_fit_predict(mode_features, mode_targets, trained_mode_model):
    # test a simple model and the default

    y_pred = trained_mode_model.predict(mode_features, mode_model_util.MULTIMODES)
    assert len(mode_targets) == len(y_pred)
    Xd = mode_features.to_dict("records")[:2]

    # test individual predictions
    for x, yp in zip(Xd, y_pred):
        yp_one = trained_mode_model.predict_one(x, mode_model_util.MULTIMODES)
        assert yp_one == yp

        # make sure the returned value of predict_one is a string
        # (not a numpy array)
        assert isinstance(yp_one, str)

    # test proba predictions
    y_pred_proba = trained_mode_model.predict_proba(mode_features)
    for x, yp in zip(Xd, y_pred_proba):
        assert_proba(
            trained_mode_model.predict_proba_one(x, mode_model_util.MULTIMODES), yp
        )


def test_predict_proba(mode_features, trained_mode_model):
    """Make sure predict_proba and predict are consistent."""
    y_pred = trained_mode_model.predict(mode_features)
    y_proba = trained_mode_model.predict_proba(mode_features)
    classes = sorted(mode_model_util.MULTIMODES)
    y_proba_second_layer = [
        {k: v for k, v in d.items() if k in classes} for d in y_proba
    ]
    y_pred_from_proba = [max(yp, key=yp.get) for yp in y_proba_second_layer]
    assert list(y_pred) == list(y_pred_from_proba)

    # make sure predict_proba and predict_proba_one are consistent
    Xd = mode_features.to_dict("records")[:2]
    for x, yp in zip(Xd, y_proba):
        assert_proba(trained_mode_model.predict_proba_one(x), yp)


def test_target_equal_mode_hist(mode_features):
    """Regression test to assure ModeModel.get_features is always called."""
    mode_targets = mode_features["mode_hist"]
    model = mode_model.ModeModel([tuple(mode_model_util.MULTIMODES)])
    y_pred = model.fit(mode_features, mode_targets).predict(mode_features)
    y_proba = model.predict_proba(mode_features)

    # make sure predict_proba and predict_proba_one are consistent
    Xd = mode_features.to_dict("records")[:2]
    for x, yp in zip(Xd, y_pred):
        assert model.predict_one(x) == yp

    # make sure predict_proba and predict_proba_one are consistent
    for x, yp in zip(Xd, y_proba):
        assert_proba(model.predict_proba_one(x), yp)


def test_score(mode_features, mode_targets):
    model = mode_model.ModeModel([tuple(mode_model_util.MULTIMODES)])
    for score in ["accuracy", "average_precision"]:
        scores1 = model.score(
            mode_features,
            mode_targets,
            score=score,
            n_folds=2,
            n_jobs=1,
            needs_confusion_matrix=True,
        )
        scores2 = model.score(
            mode_features,
            mode_targets,
            score=score,
            n_folds=2,
            n_jobs=1,
            needs_confusion_matrix=True,
        )
        for mode_selection, results in scores1.items():
            for score_method, s1 in results.items():
                s2 = scores2[mode_selection][score_method]
                if score_method != "confusion_matrix":
                    assert s1 == s2
                else:
                    for q in QUANTITIES:
                        assert np.array_equal(s1[q], s2[q])


def test_get_weights(trained_mode_model):
    weight = get_mini_weights(trained_mode_model)
    for _, w in weight.items():
        assert isinstance(w, dict)
        assert all([q in w for q in climate_model.QUANTITIES])
