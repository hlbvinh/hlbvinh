from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from ..prediction.util import time_of_day
from ..utils import sklearn_utils, testing

np.random.seed(1)


def generate_samples(id_label, ids, hours, minutes):
    samples = [
        {
            id_label: i,
            "timestamp": datetime(2017, 1, 1, hour, minute),
            "tod_sin": time_of_day(datetime(2017, 1, 1, hour, minute))[0],
            "tod_cos": time_of_day(datetime(2017, 1, 1, hour, minute))[1],
        }
        for minute in minutes
        for hour in hours
        for i in ids
    ]
    return pd.DataFrame(samples)


def assert_cv(fold, runs, samples):
    for run in runs:
        cv = fold(**run[0])
        folds = cv.split(X=samples, groups=samples["timestamp"])
        min_ts = None
        max_ts = None
        for tr, ts in folds:
            tr_samples = samples.iloc[tr]
            ts_samples = samples.iloc[ts]
            assert len(tr_samples) == run[1]["sample_size"]
            assert np.max(tr_samples["timestamp"]) < np.min(ts_samples["timestamp"])
            assert (
                np.max(ts_samples["timestamp"]) - np.min(ts_samples["timestamp"])
                <= run[1]["sample_range"]
            )
            if min_ts is None or min_ts > np.min(ts_samples["timestamp"]):
                min_ts = np.min(ts_samples["timestamp"])
            if max_ts is None or max_ts < np.max(ts_samples["timestamp"]):
                max_ts = np.max(ts_samples["timestamp"])
        assert max_ts - min_ts <= run[1]["test_range"]


def test_grid_search_to_df():
    X = np.random.rand(100, 2)
    y = np.random.rand(100)
    grid = {"C": [1, 2], "gamma": [0.1, 0.2]}
    gs = GridSearchCV(SVR(), grid, cv=3)
    gs.fit(X, y)
    df = sklearn_utils.grid_search_score_dataframe(gs)
    assert len(gs.cv_results_["params"]) == len(df)


@pytest.fixture(params=[True, False])
def normalize(request):
    return request.param


def test_permutation_feature_importance(normalize):
    # without feature groups
    estimator = ExtraTreesRegressor(n_estimators=10)
    features = [
        "humidity",
        "temperature_out",
        "feelslike",
        "luminosity_mean",
        "pircount_mean",
        "feedback",
    ]
    X = testing.gen_feature_matrix(features, 100).drop(["feedback"], 1)
    y = testing.gen_feature_matrix(features, 100)["feedback"]
    test_PFI = sklearn_utils.permutation_feature_importance(estimator, X, y, 5)
    assert len(test_PFI) == 5
    assert bool(test_PFI.index.isin(features).all()) is True
    if normalize:
        assert bool(0.9999 <= np.around(np.sum(test_PFI)) <= 1) is True

    # with feature groups
    feature_groups = {
        "time_of_day": ["cos_time_of_day", "sin_time_of_day"],
        "humidity": "humidity",
        "temperature": "temperature",
    }
    cos_time_of_day = testing.gen_random_feature_normal(
        "cos_time_of_day", 100, mean=0.0, sigma=1.0
    )
    sin_time_of_day = testing.gen_random_feature_normal(
        "sin_time_of_day", 100, mean=0.0, sigma=1.0
    )
    humidity = testing.gen_random_feature_normal("humidity", 100, mean=60.0, sigma=12.0)
    temperature = testing.gen_random_feature_normal(
        "temperature", 100, mean=28.0, sigma=6.0
    )
    X2 = pd.concat([cos_time_of_day, sin_time_of_day, humidity, temperature], axis=1)
    test_PFI2 = sklearn_utils.permutation_feature_importance(
        estimator, X2, temperature["temperature"], 5, feature_groups
    )
    assert len(test_PFI2) == 3
    assert bool(test_PFI2.index.isin(feature_groups).all()) is True

    if normalize:
        assert bool(0.9999 <= np.around(np.sum(test_PFI2)) <= 1) is True

    assert test_PFI2.idxmax() == "temperature"


def test_scores_resample():
    scores = np.random.rand(1, 100)[0]
    resampled_scores_0 = sklearn_utils.scores_resample(
        scores, validation_split_interval="1H", average_interval="10H"
    )
    resampled_scores_1 = sklearn_utils.scores_resample(
        scores, validation_split_interval=1, average_interval=14
    )
    assert len(resampled_scores_0) == 10
    assert len(resampled_scores_1) == np.ceil(100.0 / 14)


def test_scores_rolling_mean():
    scores = np.random.rand(1, 100)[0]
    rolling_means = sklearn_utils.scores_rolling_mean(scores, 1, 10, return_error=False)
    rolling_means_1 = sklearn_utils.scores_rolling_mean(
        scores, "1H", "10H", return_error=False
    )
    assert len(scores) == len(rolling_means)

    scores_padded = np.append(scores[-9:], scores)
    scores_padded = pd.Series(scores_padded)
    test_rolling_scores = scores_padded.rolling(10).mean().dropna()
    rolling_means_2, rolling_errors = sklearn_utils.scores_rolling_mean(
        scores, 1, 10, return_error=True
    )
    test_rolling_errors = scores_padded.rolling(10).std().dropna()
    for ts_0, ts_1, ts_2, ts_3 in zip(
        test_rolling_scores.values, rolling_means, rolling_means_1, rolling_means_2
    ):
        assert ts_0 == ts_1
        assert ts_0 == ts_2
        assert ts_0 == ts_3
    for es_0, es_1 in zip(test_rolling_errors, rolling_errors):
        assert es_0 == es_1


def test_score_avg():
    scores = np.random.rand(1, 100)[0]
    avg_scores_test = sklearn_utils.scores_avg(scores, "1H", "2H", "4H")
    resampled_scores = sklearn_utils.scores_resample(
        scores, validation_split_interval="1H", average_interval="2H"
    )
    avg_scores = sklearn_utils.scores_rolling_mean(
        resampled_scores, validation_split_interval="1H", average_interval="2H"
    )

    assert len(avg_scores) == len(avg_scores_test)
    assert len(avg_scores_test) == 50

    for s0, s1 in zip(avg_scores, avg_scores_test):
        assert s0 == s1


def test_reseample_grid_scores():
    scores = np.random.rand(1, 100)[0]
    avg_scores = sklearn_utils.scores_avg(scores, "1H", "2H", "4H")
    mean_score = np.mean(avg_scores)
    grid_scores_dict = {"cv_validation_scores": scores}
    new_grid_dict = sklearn_utils.reseample_grid_scores(
        grid_scores_dict, "1H", "2H", "4H"
    )
    avg_scores_test = new_grid_dict["cv_validation_scores"]
    mean_score_test = new_grid_dict["mean_validation_score"]

    assert len(avg_scores) == len(avg_scores_test)
    for s0, s1 in zip(avg_scores, avg_scores_test):
        assert s0 == s1

    assert mean_score == mean_score_test


def test_time_series_fold():
    samples = generate_samples("device_id", range(10), range(0, 24), range(0, 60))
    runs = [
        (
            {
                "n_train": 50,
                "validation_interval": timedelta(days=1),
                "validation_split_interval": "3H",
            },
            {
                "sample_size": 50,
                "sample_range": timedelta(hours=3),
                "test_range": timedelta(days=1),
            },
        ),
        (
            {
                "n_train": 80,
                "validation_interval": timedelta(days=1),
                "validation_split_interval": "3H",
            },
            {
                "sample_size": 80,
                "sample_range": timedelta(hours=3),
                "test_range": timedelta(days=1),
            },
        ),
    ]

    assert_cv(sklearn_utils.TimeSeriesFold, runs, samples)


def test_time_series_fold_0_sample_limit():
    samples = generate_samples("device_id", range(10), range(0, 24), range(0, 60))
    params = {
        "n_train": 0,
        "validation_interval": timedelta(days=1),
        "validation_split_interval": "3H",
    }
    cv = sklearn_utils.TimeSeriesFold(**params)
    folds = cv.split(X=samples, groups=samples["timestamp"])
    tr_sample_size = 0
    sample_generated_per_hour = 3 * 60 * 10
    for tr, ts in folds:
        tr_sample_size += sample_generated_per_hour
        assert len(tr) == tr_sample_size
        assert len(ts) == sample_generated_per_hour
