from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from skynet.sample.selection import COMPENSATE_COLUMN
from ...prediction.climate_model import (
    QUANTITIES,
    FEATURE_COLUMNS,
    ClimateModel,
    climate_scoring,
    make_static_climate_dataset,
    make_static_mode_dataset,
    prepare_dataset,
)
from ...utils.testing import gen_feature_matrix


@pytest.fixture
def key1():
    return ClimateModel.get_storage_key(model_type="a", model_version=1)


def test_init(untrained_climate_model):
    assert untrained_climate_model.model_type == "a"
    assert untrained_climate_model.model_version == 1
    assert isinstance(untrained_climate_model.estimator, BaseEstimator)


def test_fit_predict(untrained_climate_model, climate_features, climate_targets):
    X = climate_features
    y = climate_targets
    untrained_climate_model.fit(X, y)
    y_pred = untrained_climate_model.predict(X)
    assert len(y) == len(y_pred)


def test_key(
    untrained_climate_model, untrained_climate_model_a, untrained_climate_model_b, key1
):
    assert untrained_climate_model.storage_key == untrained_climate_model_a.storage_key
    assert untrained_climate_model.storage_key == key1
    assert untrained_climate_model.storage_key != untrained_climate_model_b.storage_key


def test_score(untrained_climate_model, climate_features, split_score_sample_size):
    random_y = np.random.rand(split_score_sample_size)
    X = climate_features
    scores = untrained_climate_model.score(X, random_y, n_jobs=1)
    assert not scores.empty


@pytest.fixture
def climate_model():
    return ClimateModel()


@pytest.fixture
def minimal_climate_model_features():
    return {
        "temperature_set": 20,
        "temperature_set_last": 20,
        "mode": "cool",
        "previous_temperatures": np.array([1, 2, 3, 4], dtype=np.float32),
        "temperature_out": 22,
        "temperature_out_mean_day": 22,
        "temperature_set_diff": 22,
    }


@pytest.fixture
def sequence_climate_model_features(split_score_sample_size):
    cols = FEATURE_COLUMNS
    X = pd.DataFrame(np.random.rand(split_score_sample_size, len(cols)), columns=cols)
    X[COMPENSATE_COLUMN] = True
    X["previous_temperatures"] = X["previous_temperatures"].apply(
        lambda x: np.array([0, 1, 4], dtype=np.float32)
    )
    targets = np.random.rand(len(X), len(QUANTITIES))
    return X, targets


@pytest.mark.skip("somehow the training can be extremely slow on CI")
@pytest.mark.filterwarnings("ignore:Got `batch_size` less than:UserWarning")
def test_climate_model_pipeline(
    sequence_climate_model_features, climate_model, minimal_climate_model_features
):
    climate_features, climate_targets = sequence_climate_model_features
    climate_model.fit(climate_features, climate_targets)
    y_pred = climate_model.predict(climate_features)
    assert y_pred.shape == climate_targets.shape

    y_pred_empty = climate_model.predict([minimal_climate_model_features])
    assert y_pred_empty.shape == (1, 3)


@pytest.fixture
def raw_samples():

    n_targets = 90
    n_samples = 2
    features = []
    targets = []
    start = datetime(2018, 1, 1)

    # adding mode and temperature_set to test prepare_dataset function later
    feature_names = QUANTITIES + ["mode", "temperature_set"]

    for sample_id in range(n_samples):
        y = gen_feature_matrix(QUANTITIES, n_targets)
        y["sample_id"] = sample_id
        y["timestamp"] = pd.date_range(start, freq="5Min", periods=n_targets)
        X = gen_feature_matrix(feature_names, 1)
        X["sample_id"] = sample_id
        X["timestamp"] = start
        X["previous_temperatures"] = [np.array([1, 2, 3], dtype=np.float32)] * len(X)
        features.append(X)
        targets.append(y)

    return (
        pd.concat(features).set_index("sample_id"),
        pd.concat(targets).set_index("sample_id"),
    )


@pytest.fixture
def static_climate_dataset(raw_samples):
    return make_static_climate_dataset(*raw_samples)


def test_make_static_climate_dataset(static_climate_dataset):
    Xs, ys = static_climate_dataset
    assert Xs.shape[0] == ys.shape[0]


@pytest.fixture
def static_mode_dataset(raw_samples):
    return make_static_mode_dataset(*raw_samples)


def test_make_static_mode_dataset(static_mode_dataset):
    Xs, ys = static_mode_dataset
    assert Xs.shape[0] == ys.shape[0]


def test_prepare_dataset(static_climate_dataset):
    Xs, _ = static_climate_dataset
    X_prepared = prepare_dataset(Xs)
    assert X_prepared.shape == Xs.shape


@pytest.fixture
def dummy_estimator():
    class DummyEstimator:
        def fit(self, X, y):
            pass

        @staticmethod
        def predict(X):
            return X

    return DummyEstimator()


@pytest.fixture
def scoring():
    return climate_scoring(mean_absolute_error)


def test_scoring(dummy_estimator, scoring):
    """Scorer should select the corresponding column for the predictions.

    scoring[quantity] should use the i-th column of the predictions
        where i is its index in QUANTITIES.
    """
    X = np.array([[0, 0, 0]])
    y_true = np.array([[1, 2, 3]])
    errors = np.abs(y_true - X).ravel()

    for quantity, err in zip(QUANTITIES, errors):

        scorer = scoring[quantity]
        assert scorer(dummy_estimator, X=X, y_true=y_true) == err
