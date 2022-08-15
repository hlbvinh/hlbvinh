from datetime import datetime, timedelta

import numpy as np
import pytest
from sklearn import linear_model, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

from ..user.sample import COMFORT_FEATURES_WITH_TARGET
from ..utils import sklearn_utils
from ..utils.sklearn_wrappers import DFVectorizer
from ..utils.testing import gen_feature_datetime, gen_feature_matrix

N_SAMPLES = 100
START = datetime(2016, 1, 1)
TIME_FREQ = "1s"
TIME_VALIDATION_INTERVAL = timedelta(seconds=10)
TIME_VALIDATION_SPLIT_INTERVAL = "2s"
VALIDATION_INTERVAL = 10
VALIDATION_SPLIT_INTERVAL = 2
N_TRAIN = 80


@pytest.fixture
def features():
    dt = gen_feature_datetime("timestamp", N_SAMPLES, START, TIME_FREQ)
    X = gen_feature_matrix(features=COMFORT_FEATURES_WITH_TARGET, n_samples=N_SAMPLES)
    X["timestamp"] = dt["timestamp"].values
    return X


@pytest.fixture
def targets():
    y = gen_feature_matrix(features=["feedback"], n_samples=N_SAMPLES)["feedback"]
    return y


def test_base_series_fold():
    with pytest.raises(TypeError):
        # pylint: disable=abstract-class-instantiated
        sklearn_utils.BaseSeriesFold(
            n_train=N_TRAIN,
            validation_split_interval=VALIDATION_SPLIT_INTERVAL,
            validation_interval=VALIDATION_INTERVAL,
        )


# pylint: enable=abstract-class-instantiated


@pytest.fixture
def time_series_fold():
    return sklearn_utils.TimeSeriesFold(
        n_train=N_TRAIN,
        validation_split_interval=TIME_VALIDATION_SPLIT_INTERVAL,
        validation_interval=TIME_VALIDATION_INTERVAL,
    )


def test_time_series_fold(features, time_series_fold):

    groups = features["timestamp"]
    splits = list(time_series_fold.split(features, groups=groups))
    assert splits
    for tr, te in splits:
        assert len(tr) == N_TRAIN
        assert len(te) == 2

    # check if works with array as groups
    splits_arr = list(time_series_fold.split(features, groups=groups.values))
    assert splits_arr
    for (tr1, te1), (tr2, te2) in zip(splits, splits_arr):
        assert np.allclose(tr1, tr2)
        assert np.allclose(te1, te2)

    # check if test folds are ordered and non overlapping
    test_idxs = [test_idx for _, test_idx in splits]
    for last, current in zip(test_idxs[:-1], test_idxs[1:]):
        assert max(groups[last]) < min(groups[current])

    # shuffle and compare results
    features_shuffled = shuffle(features)
    splits2 = list(
        time_series_fold.split(features_shuffled, groups=features_shuffled["timestamp"])
    )
    assert splits2
    for (tr1, te1), (tr2, te2) in zip(splits, splits2):
        assert features.iloc[tr1].equals(features_shuffled.iloc[tr2])
        assert features.iloc[te1].equals(features_shuffled.iloc[te2])


def test_cross_val_score(time_series_fold, features, targets):
    estimator = make_pipeline(DFVectorizer(), linear_model.Ridge())
    groups = features.pop("timestamp")
    scores = model_selection.cross_val_score(
        estimator, features, targets, cv=time_series_fold, groups=groups
    )
    assert len(scores) == time_series_fold.get_n_splits(features, groups=groups)


def test_grid_search_cv(time_series_fold, features, targets):
    estimator = make_pipeline(DFVectorizer(), linear_model.Ridge())
    param_grid = {"ridge__alpha": [0.1, 1.0]}
    gs = model_selection.GridSearchCV(
        estimator, param_grid, refit=False, cv=time_series_fold
    )
    groups = features.pop("timestamp")
    gs.fit(features, targets, groups=groups)
    assert gs.n_splits_ == 5
