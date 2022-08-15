import functools
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import model_selection
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import compute_sample_weight

from ..utils.compensation import ensure_compensated
from ..utils.log_util import get_logger
from ..utils.sklearn_utils import safe_indexing
from . import mode_model_util, model
from .climate_model import QUANTITIES
from .estimators.mode_model import get_mini_pipeline
from .mode_config import MODES

log = get_logger(__name__)


MODEL_TYPE = "mode_model"
MODE_MODEL_VERSION = 18
TRAINING_INTERVAL_SECONDS = 3600 * 3
RELOAD_INTERVAL_SECONDS = TRAINING_INTERVAL_SECONDS / 10

ALL_SCORES = ["accuracy", "average_precision"]

SCORING_VARIANTS = ["all"] + ALL_SCORES


# ## Mathis 2015 10 12 ###
# Optimized features for mode prediction using logistic regression
# balancing influence of groups of features.
# see skynet/predicton/estimators/mode_model.py
HISTORICAL_FEATURE_COLUMNS = [
    "appliance_id",
    "humidex",
    "humidity",
    "temperature",
    "temperature_out",
    "humidex_out",
    "humidity_out",
    "temperature_out_mean_day",
]

TARGET_FEATURE_COLUMNS = ["quantity", "target"]

FEATURE_COLUMNS = HISTORICAL_FEATURE_COLUMNS + TARGET_FEATURE_COLUMNS

# Lookup table for all mini modes features
# if not specified, feature from FEATURE_COLUMN will be used
MINI_MODELS_FEATURES = {
    ("cool", "heat"): ["humidex_out", "temperature_out_mean_day", "target"],
    ("dry", "fan"): ["temperature", "humidex", "humidity", "target"],
    ("cool", "fan"): [
        "temperature",
        "humidex",
        "humidity",
        "target",
        "temperature_out",
        "humidex_out",
    ],
    ("cool", "dry"): ["temperature", "humidity", "target"],
    ("auto", "cool", "fan"): ["temperature", "humidex", "humidity", "target"],
}


MINI_MODELS_FEATURES = {
    k: v + ["appliance_id", "quantity"] for k, v in MINI_MODELS_FEATURES.items()
}


def create_lookup_key():
    return {
        "origin": {"$in": ["irdeployment", "reverse"]},
        "timestamp": {"$gte": datetime.utcnow() - timedelta(days=365)},
    }


def make_features(history_features, target_quantity, target_value):
    features = history_features.copy()
    features["quantity"] = target_quantity
    features["target"] = target_value
    return features


def _fit_and_score(
    estimator,
    X,
    y,
    score_functions,
    train,
    test,
    needs_confusion_matrix,
    label_binarizer=None,
    sample_weights=None,
):

    estimator.fit(
        safe_indexing(X, train),
        safe_indexing(y, train),
        fit__sample_weight=sample_weights,
    )

    y_test = safe_indexing(y, test)
    X_test = safe_indexing(X, test)
    all_scores = defaultdict(dict)
    for score_function in score_functions:
        for quantity in QUANTITIES:
            idx = np.where(X_test["quantity"] == quantity)[0]
            truth, prediction = y_test.iloc[idx], estimator.predict(X_test.iloc[idx])
            all_scores[score_function["name"]][quantity] = score_function["scorer"](
                *map(
                    label_binarizer.transform
                    if score_function["needs_binarize"]
                    else lambda x: x,
                    [truth, prediction],
                )
            )
            if needs_confusion_matrix:
                all_scores["test_samples"][quantity] = (truth.values, prediction)
    return all_scores


class ModeModel(model.Model):

    """Learns best AC mode based on current and desired conditions."""

    @classmethod
    def get_storage_key(cls, model_type=MODEL_TYPE, model_version=MODE_MODEL_VERSION):
        return super().get_storage_key(model_type, model_version)

    def __init__(
        self,
        mode_selections_groups,
        n_jobs=1,
        model_type=MODEL_TYPE,
        model_version=MODE_MODEL_VERSION,
    ):
        super().__init__(model_type=model_type, model_version=model_version)
        self.mode_selections_groups = mode_selections_groups
        self.n_jobs = n_jobs
        self.mini_estimators = {}
        self.estimators = {}
        self.feature_columns = FEATURE_COLUMNS
        self._init_estimator()

    def set_params(self, **params):
        param = params["mini_params"]
        for p in param:
            mode_selection = p["mode_selection"]
            mini_params = p["params"]
            self.mini_estimators[mode_selection].set_params(**mini_params)

    def get_features(self, X):
        return X.copy()[set(self.feature_columns).intersection(X.columns)]

    def _init_estimator(self):
        for mode_selection in self.mode_selections_groups:

            self.estimators.update(
                {
                    mode_selection: {
                        q: mode_model_util.MultiModesEstimator(
                            mode_selection=mode_selection
                        )
                        for q in QUANTITIES
                    }
                }
            )
            mini_groups = mode_model_util.get_mini_groups(mode_selection)
            self.estimators[mode_selection]["mini_groups"] = mini_groups
            for _, v in mini_groups.items():
                if v not in self.mini_estimators:
                    if len(v) > 1:
                        self.mini_estimators[v] = get_mini_pipeline(
                            features=MINI_MODELS_FEATURES.get(v, FEATURE_COLUMNS)
                        )

    def _update_estimator(self, new_mini_group):
        for mode_selection, est in self.estimators.items():
            for layer, mini_group in est["mini_groups"].items():
                if mini_group == new_mini_group:
                    for q in QUANTITIES:
                        est = self.mini_estimators[mini_group].models[q]
                        self.estimators[mode_selection][q].insert_estimator(
                            layer=layer, estimator=est
                        )

    @ensure_compensated
    def fit(self, X, y, **fit_params):
        self.log_features(X)

        if "mode_selection" in fit_params:
            mode_selection = [fit_params["mode_selection"]]
        else:
            mode_selection = self.mode_selections_groups
        mini_groups = mode_model_util.get_all_mini_groups(mode_selection)

        log.debug(f"training mode model for {len(mode_selection)} mode selections")
        log.debug(f"training {len(mini_groups)} mini-estimator  ")

        # Reversing because the last models take the longest time to train.
        mini_groups = mini_groups[::-1]

        args = []
        estimators = []
        weights = []
        for mini_group in mini_groups:
            log.debug(f"training mini-estimator for the modes: {mini_group}.")
            mini_group_mask = np.in1d(y, mini_group)

            if mini_group is None:
                mini_group = tuple(np.unique(y))

            args.append((X[mini_group_mask], y[mini_group_mask]))
            estimators.append(self.mini_estimators[mini_group])
            weights.append(get_sample_weights(y[mini_group_mask]))

        fit_estimators = Parallel(n_jobs=self.n_jobs, verbose=50)(
            (
                delayed(est.fit)(X, y, fit__sample_weight=w)
                for est, (X, y), w in zip(estimators, args, weights)
            )
        )

        for estimator, mini_group in zip(fit_estimators, mini_groups):
            self.mini_estimators[mini_group] = estimator
            self._update_estimator(new_mini_group=mini_group)

        return self

    def predict(self, X, mode_selection=mode_model_util.MULTIMODES):
        probas = self._predict(self.get_features(X), mode_selection)
        return np.array([mode_model_util.mode_from_probas(p) for p in probas])

    def predict_proba(self, X, mode_selection=mode_model_util.MULTIMODES):
        return self._predict(self.get_features(X), mode_selection)

    def predict_one(self, x, mode_selection=mode_model_util.MULTIMODES):
        return self.predict(pd.DataFrame([x]), mode_selection)[0]

    def predict_proba_one(self, x, mode_selection=mode_model_util.MULTIMODES):
        return self.predict_proba(pd.DataFrame([x]), mode_selection)[0]

    @staticmethod
    def _empty_predictions(X):
        return np.empty(len(X), dtype=dict)

    def _predict(self, X, mode_selection):
        if isinstance(mode_selection, list):
            mode_selection = tuple(sorted(mode_selection))
        X = X.copy()
        X["indices"] = np.arange(len(X))
        y_pred = self._empty_predictions(X)
        for quantity, df in X.groupby("quantity"):
            model = self.estimators[mode_selection][quantity]
            y_pred[df.indices.values] = model.predict_proba(df)
        return y_pred

    def transform(self, X):
        return self.estimator.transform(X)

    def score(
        self,
        X,
        y,
        score="all",
        cv=None,
        n_folds=4,
        n_jobs=1,
        groups=None,
        needs_confusion_matrix=True,
        mode_selections=None,
    ):
        if mode_selections is None:
            mode_selections = self.mode_selections_groups
        else:
            mode_selections += mode_model_util.get_all_mini_groups(mode_selections)
        scores = {}
        for modes in mode_selections:
            mask = np.in1d(y, modes)
            scores[modes] = self._score(
                X[mask],
                y[mask],
                modes,
                score,
                cv,
                n_folds,
                n_jobs,
                groups[mask] if groups is not None else None,
                needs_confusion_matrix,
                get_sample_weights(y[mask]),
            )
        return scores

    def _score(
        self,
        X,
        y,
        mode_selection,
        score="all",
        cv=None,
        n_folds=4,
        n_jobs=1,
        groups=None,
        needs_confusion_matrix=True,
        sample_weights=None,
    ):
        score_types = ALL_SCORES if score == "all" else [score]
        score_function = {
            "accuracy": dict(
                name="accuracy", scorer=mode_accuracy_score, needs_binarize=False
            ),
            "average_precision": dict(
                name="average_precision",
                scorer=functools.partial(average_precision_score, average="micro"),
                needs_binarize=True,
            ),
        }
        fold_scores = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_fit_and_score)(
                clone(
                    get_mini_pipeline(
                        MINI_MODELS_FEATURES.get(mode_selection, FEATURE_COLUMNS)
                    )
                ),
                self.get_features(X),
                y,
                [score_function[score_type] for score_type in score_types],
                train,
                test,
                needs_confusion_matrix,
                LabelBinarizer().fit(y),
                sample_weights,
            )
            for train, test in (
                cv or model_selection.GroupKFold(n_splits=n_folds)
            ).split(X, groups=groups if groups is not None else X["appliance_id"])
        )

        scores = {
            score_type: {
                quantity: "{mean:.3g} +/- {std:.3g}".format(**agg)
                for quantity, agg in (
                    pd.DataFrame([fold_score[score_type] for fold_score in fold_scores])
                    .aggregate(["mean", "std"])
                    .to_dict()
                    .items()
                )
            }
            for score_type in score_types
        }

        if needs_confusion_matrix:
            scores["confusion_matrix"] = {
                quantity: confusion_matrix(
                    *map(
                        np.concatenate,
                        zip(
                            *[
                                fold_score["test_samples"][quantity]
                                for fold_score in fold_scores
                            ]
                        ),
                    )
                )
                for quantity in QUANTITIES
            }
        return scores


def mode_accuracy_score(y_true, y_pred, modes=MODES):
    true_positive = sum(len(y_true[(y_true == m) & (y_pred == m)]) for m in modes)
    return true_positive / len(y_true)


def make_mode_model_dataset(X, targets):
    dfs = []
    for quantity in QUANTITIES:
        X = X.copy()
        X["target"] = targets[quantity].values
        X["quantity"] = quantity
        dfs.append(X)
    X = pd.concat(dfs)
    y = X["mode"]
    return X, y


def get_sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight("balanced", y)
