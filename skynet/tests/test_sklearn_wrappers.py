import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.utils import check_random_state

from ..utils import sklearn_wrappers, testing, thermo

N = 20


class NonRepeatableRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, noise, random_state=None):
        self.estimator = estimator
        self.noise = noise
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        self._random_state = check_random_state(self.random_state)
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator.predict(X) + self._random_state.uniform(
            -self.noise / 2.0, self.noise / 2.0
        )


def test_individual():
    X = testing.gen_feature_matrix(["humidex", "temperature", "mode", "device_id"], 100)
    Xd = X.to_dict("records")
    y = X.pop("humidex")
    estimator = make_pipeline(sklearn_wrappers.DFVectorizer(), SVR(gamma="scale"))
    for groups in [["mode"], ["mode", "device_id"]]:
        est = sklearn_wrappers.Individual(
            estimator,
            groups,
            max_samples=100,
            min_samples=5,
            fallback_estimator=estimator,
        )
        est.fit(X, y)
        y_p_0 = est.predict(X)
        y_p_1 = est.predict(Xd)
        assert len(y_p_0) == len(y_p_1)
        for y0, y1 in zip(y_p_0, y_p_1):
            assert np.isclose(y0, y1), "failed group by {}, {} != {}".format(
                groups, y0, y1
            )


# On Ubuntu 18.04 test was failing due to RuntimeWarning
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_set_temperature_converter():
    ts = [[np.nan, 100, "0"], [20, None, "10"], [0, 10, "20"], ["blah", "blah2", "100"]]
    df = pd.DataFrame(ts, columns=["a", "b", "c"])
    orig_df = df.copy()
    dicts = df.to_dict("records")
    converter = sklearn_wrappers.SetTemperatureConverter(["a", "b", "c"])
    df_conv = converter.fit_transform(df)
    dicts_conv = converter.transform(dicts)

    assert df_conv.equals(pd.DataFrame.from_records(dicts_conv))

    # check if input was left unchanged
    assert orig_df.equals(df), "df input modified"
    assert orig_df.equals(pd.DataFrame(dicts)), "dicts input modified"

    # check if all converted finite values are within the range
    assert np.all(df_conv.fillna(thermo.MAX_CELSIUS) <= thermo.MAX_CELSIUS)
    assert np.all(df_conv.fillna(thermo.MIN_CELSIUS) >= thermo.MIN_CELSIUS)


@pytest.mark.parametrize(
    "X,result",
    [
        pytest.param(
            [{"temperature_set": 24, "temperature": 26, "temperature_set_last": 30}],
            [
                {
                    "temperature_set": 24,
                    "temperature": 26,
                    "temperature_set_last": 30,
                    "temperature_set_difference": -6,
                }
            ],
            id="Checking for base case",
        )
    ],
)
def test_set_temperature_difference(X, result):
    difference = sklearn_wrappers.TemperatureSetDifference()
    assert result == difference.fit_transform(X)
    assert pd.DataFrame(result).equals(difference.fit_transform(pd.DataFrame(X)))


def gen_samples_with_outliers(samples_size=500, mu=0.0, stdev=1.0):
    X = pd.DataFrame({"A": np.ones(samples_size), "B": np.zeros(samples_size)})
    y = np.random.normal(loc=mu, scale=stdev, size=samples_size)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return X, y, y_mean, y_std


def test_filter_outliers():
    X, y, y_mean, y_std = gen_samples_with_outliers()
    len_inliers = len(y[(y > y_mean - y_std) & (y <= y_mean + y_std)])
    y_series = pd.Series(y)
    for target in [y, y_series]:
        inliers_mask = sklearn_wrappers.filter_outliers(
            LinearRegression(), X, target, std_confident_level=1.0
        )
        assert len(X[inliers_mask]) == len_inliers
        assert all(
            (target[inliers_mask] >= y_mean - y_std)
            & (target[inliers_mask] < y_mean + y_std)
        )
        assert np.mean(y) == y_mean


def do_test_std_based_filter_wrapper(filtr, X, y, inliers_mask):

    assert len(filtr.fit(X, y).inlier_mask) == len(inliers_mask)
    for i, j in zip(filtr.fit(X, y).inlier_mask, inliers_mask):
        assert i == j


def test_std_based_filter_wrapper():
    filtr = sklearn_wrappers.StdBasedFeatureAndTargetFilter(
        estimator=LinearRegression(), columns=["A"], std_confident_level=1.0
    )

    # generate test sample with outliers, std is set to 1.0, mean =0.0
    X, y, _, _ = gen_samples_with_outliers()

    # generate the records and dict form of the test samples
    X_records = X.to_dict(orient="records")
    X_dict = X.to_dict(orient="list")

    # get inlier make from the test samples
    inliers_mask = sklearn_wrappers.filter_outliers(
        LinearRegression(), X, y, std_confident_level=1.0
    )

    # test wrapper with data in DataFrame, records and dict
    for Xt in [X, X_records, X_dict]:
        filtr.fit(X_records, y)
        do_test_std_based_filter_wrapper(clone(filtr), Xt, y, inliers_mask)

    # test wrapper return valid error messages with wrong types of data
    test_X = (1, 2, 3, 4)
    test_y = (1, 2, 3, 4)
    msg = (
        "Selector works on DataFrame, dict and iterables of dicts, "
        "{} passed.".format(type(test_X))
    )
    try:
        filtr.fit(test_X, test_y)
    except Exception as exp:
        assert exp.args[0] == msg

    # test filter do not filter samples when only X is passed
    res = filtr.transform(X)
    assert len(res) == len(X)
    res = filtr.fit_transform(X)
    assert len(res) == len(X)
    res2 = filtr.fit(X).transform(X)
    assert all([i == j for i, j in zip(res, res2)])

    # test wrapper filter samples correctly when X and y and passed
    res = filtr.transform(X, y)
    assert len(res) == 3
    res = filtr.fit_transform(X, y)
    assert len(res) == 3
    assert len(res[0]) == len(res[1])
    res2 = filtr.fit(X, y).transform(X, y)
    assert len(res2) == 3
    filtr2 = sklearn_wrappers.StdBasedFeatureAndTargetFilter(
        estimator=LinearRegression(),
        columns=["A"],
        std_confident_level=1.0,
        bypass=True,
    )
    res = filtr2.transform(X, y)
    assert len(res) == len(X)
    res = filtr2.fit_transform(X, y)
    assert len(res) == len(X)
    res = filtr2.fit(X, y).transform(X, y)
    assert len(res) == len(X)

    # test filter filter sample weight correctly
    res = filtr.fit_transform(X, y, sample_weight=np.arange(len(X)))
    assert len(res) == 3
    assert len(res[0]) == len(res[1]) == len(res[2]["sample_weight"])

    # test pass no specified column to filter
    filtr3 = sklearn_wrappers.StdBasedFeatureAndTargetFilter(
        estimator=LinearRegression(), std_confident_level=1.0, bypass=True
    )
    res = filtr3.transform(X, y)
    assert len(res) == len(X)

    # test when self.inlier_mask is None, it return original X, y
    filtr4 = sklearn_wrappers.StdBasedFeatureAndTargetFilter(
        estimator=LinearRegression(), columns=["A"], std_confident_level=1.0
    )
    assert filtr4.inlier_mask is None
    res4 = filtr4.transform(X, y)
    assert len(res4) == 2
    assert all([i == j for i, j in zip(res4[0], X)])

    # test fit_predict do nothing
    res5 = filtr4.fit_predict(X, y)
    assert len(res5) == len(X)
    assert all([i == j for i, j in zip(res5, X)])


def test_average_prediction_filter():
    def gen_samples(n_samples):
        X = testing.gen_feature_matrix(
            ["humidex", "temperature", "humidity"], n_samples
        )
        y = X["humidex"]
        return X, y

    # number of estimators used in the wrapper
    n_estimator = 2
    est = NonRepeatableRegressor(estimator=LinearRegression(), noise=1)
    avg_est = sklearn_wrappers.AverageEstimator(
        estimator=est, n_estimators=n_estimator, n_jobs=1
    )
    X_train, y_train = gen_samples(n_samples=50)
    X_test, y_test = gen_samples(n_samples=20)

    raw_y_preds = []
    averaged_y_preds = []
    raw_scores = []
    averaged_scores = []
    for _ in range(2):
        raw_y_pred = est.fit(X_train, y_train).predict(X_test)
        raw_y_preds.append(raw_y_pred)
        avg_est.fit(X_train, y_train)
        averaged_y_pred = avg_est.fit(X_train, y_train).predict(X_test)
        averaged_y_preds.append(averaged_y_pred)
        raw_scores.append(mean_absolute_error(y_test, raw_y_pred))
        averaged_scores.append(mean_absolute_error(y_test, averaged_y_pred))

    raw_y_std = np.std(raw_y_preds, axis=0)
    averaged_y_std = np.std(averaged_y_preds, axis=0)

    assert np.mean(averaged_y_std) < np.mean(raw_y_std)
    assert np.mean(averaged_scores) < np.mean(raw_scores)
