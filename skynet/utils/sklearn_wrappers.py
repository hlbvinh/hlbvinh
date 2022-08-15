import abc
import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

from ..utils import log_util
from . import data, thermo
from .types import SparseFeature, Target

log = log_util.get_logger(__name__)


def fit(args):
    """Fit and return estimator. To use with pool."""
    estimator, X, y, fit_params = args
    try:
        return clone(estimator).fit(X, y, **fit_params)

    except ZeroDivisionError:
        return None


def filter_outliers(estimator, X, y, std_confident_level):
    y_pred = estimator.fit(X, y).predict(X)
    if isinstance(y, pd.Series):
        y = y.values
    error = y_pred - y
    mu = np.mean(error)
    sigma = std_confident_level * np.std(error)
    inlier_mask = (error > (mu - sigma)) & (error <= (mu + sigma))
    return inlier_mask


class TransformDataFrameOrDicts(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    def fit(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        return self

    @abc.abstractmethod
    def _transform_df(self, X, y=None):
        """Transform DataFrame."""

    @abc.abstractmethod
    def _transform_dicts(self, X, y=None):
        """Transform sequence of dicts."""

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            Xt = self._transform_df(X, y)
        else:
            Xt = self._transform_dicts(X, y)
        return Xt


class PredictDataFrameOrDicts(BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _predict_df(self, X, proba=False):
        """Predict DataFrame"""

    @abc.abstractmethod
    def _predict_dicts(self, X, proba=False):
        """Predict sequence of dicts."""

    def predict(self, X, proba=False):
        if isinstance(X, pd.DataFrame):
            y = self._predict_df(X, proba=proba)
        else:
            y = self._predict_dicts(X, proba=proba)
        return y

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            y = self._predict_df(X, proba=True)
        else:
            y = self._predict_dicts(X, proba=True)
        return y


#   TODO: Mathis Feb 12 2015
#
#   Fahrenheit and other non-celsius Set Temperature Hack
#


class SetTemperatureConverter(TransformDataFrameOrDicts):
    def __init__(self, columns=["temperature_set", "temperature_set_last"]):
        """Fixes temperature values in specified columns as well as possible.

        Converts very high temperature from Fahrenheit to Celsius and adds an
        offset to very low temperatures.

        Parameters
        ----------
        columns: list of str (default=['temperature_set',
                                       'temperature_set_last'])
            Columns to fix in feature matrix DataFrame.
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def _transform_df(self, X, y=None):
        X = X.copy()
        cols = [col for col in self.columns if col in X]
        X.loc[:, cols] = thermo.fix_temperatures(X[cols].values)
        return X

    def _transform_dicts(self, X, y=None):
        X = [x.copy() for x in X]
        ts = np.array([[x[col] for col in self.columns] for x in X])
        fixed = thermo.fix_temperatures(ts)
        for x, tt in zip(X, fixed):
            for key, t in zip(self.columns, tt):
                x[key] = t
        return X


class TemperatureSetDifference(TransformDataFrameOrDicts):
    def fit(self, X, y=None):
        return self

    def _transform_df(self, X, y=None):
        X = X.copy()
        X["temperature_set_difference"] = (
            X["temperature_set"] - X["temperature_set_last"]
        )
        return X

    def _transform_dicts(self, X, y=None):
        X = [x.copy() for x in X]
        for x in X:
            x["temperature_set_difference"] = (
                x["temperature_set"] - x["temperature_set_last"]
            )
        return X


class DFVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize DataFrame or list of dicts via sklearn's DictVectorizer."""

    def __init__(self, sparse=True):
        self.sparse = sparse

    def fit(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        self.dv_ = DictVectorizer(sparse=self.sparse)
        if isinstance(X, pd.DataFrame):
            self.dv_.fit(X.to_dict("records"))
        else:
            self.dv_.fit(X)
        return self

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        if isinstance(X, pd.DataFrame):
            return self.dv_.transform(X.to_dict("records"))
        return self.dv_.transform(X)

    def get_feature_names(self):
        return self.dv_.get_feature_names()


class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """Select only specified columns in DataFrame."""
        self.columns = columns

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        if self.columns is None:
            return X
        if isinstance(X, pd.DataFrame):
            return X[self.columns]
        if isinstance(X, list):
            return [{k: d[k] for k in self.columns} for d in X]
        if isinstance(X, dict):
            return [{k: v for k, v in X.items() if k in self.columns}]
        raise ValueError(
            "Selector works on DataFrame, dict and iterables"
            "of dicts, {} passed.".format(type(X))
        )


class PipelineWithSampleFiltering(Pipeline):
    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            Xt, fit_params = self._pre_transform_x(X, y, **fit_params)
            yt = copy.copy(y)
        else:
            Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], "fit_transform"):
            return self.steps[-1][-1].fit_transform(Xt, yt, **fit_params)
        return self.steps[-1][-1].fit(Xt, yt, **fit_params).transform(Xt)

    def fit(self, X, y=None, **fit_params):
        if y is None:
            Xt, fit_params = self._pre_transform_x(X, y, **fit_params)
            yt = copy.copy(y)
        else:
            Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        self.steps[-1][-1].fit(Xt, yt, **fit_params)
        return self

    def fit_predict(self, X, y=None, **fit_params):
        if y is None:
            Xt, fit_params = self._pre_transform_x(X, y, **fit_params)
            yt = copy.copy(y)
        else:
            Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)

    def _update_fit_params_steps(self, new_params):
        for item_to_unpack in self.add_to_filter:
            if item_to_unpack[0] in new_params:
                self.fit_params_steps[item_to_unpack[2]][
                    item_to_unpack[0]
                ] = new_params[item_to_unpack[0]]
        return self

    def _pre_transform_x(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]).transform(Xt)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        add_to_filter = []
        for pname, pval in fit_params.items():
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval

            if isinstance(pval, (np.ndarray, list)):
                if len(pval) == len(X):
                    add_to_filter.append((param, pval, step))
        for item_to_add in add_to_filter:
            fit_params_steps[self.steps[0][0]].update({item_to_add[0]: item_to_add[1]})
        self.fit_params_steps = fit_params_steps
        self.add_to_filter = add_to_filter

        Xt = X
        yt = y
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                res = transform.fit_transform(Xt, yt, **self.fit_params_steps[name])
                if isinstance(res, tuple):
                    if len(res) == 2:
                        Xt, yt = res
                    elif len(res) == 3:
                        Xt, yt, new_params = res
                        self._update_fit_params_steps(new_params)
                    else:
                        raise ValueError("Too many values tp unpack")

                else:
                    Xt = res
            else:
                Xt = transform.fit(Xt, y, **self.fit_params_steps[name]).transform(Xt)
        return Xt, yt, fit_params_steps[self.steps[-1][0]]


class StdBasedFeatureAndTargetFilter(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, columns=None, std_confident_level=2.0, bypass=False):
        """Select only specified columns in DataFrame."""
        self.estimator = estimator
        self.columns = columns
        self.std_confident_level = std_confident_level
        self.bypass = bypass
        self.inlier_mask = None

    def fit(self, X, y=None):
        if y is not None:
            if self.columns is None:
                pass
            elif isinstance(X, pd.DataFrame):
                self.inlier_mask = filter_outliers(
                    self.estimator, X[self.columns], y, self.std_confident_level
                )
            elif isinstance(X, list):
                X = pd.DataFrame([{k: d[k] for k in self.columns} for d in X])
                self.inlier_mask = filter_outliers(
                    self.estimator, X, y, self.std_confident_level
                )

            elif isinstance(X, dict):
                X = pd.DataFrame.from_dict(
                    {k: v for k, v in X.items() if k in self.columns}
                )
                self.inlier_mask = filter_outliers(
                    self.estimator, X, y, self.std_confident_level
                )
            else:
                raise ValueError(
                    "Selector works on DataFrame, dict and iterables of dicts,"
                    " {} passed.".format(type(X))
                )

        return self

    def transform(
        self, X, y=None, copy=None, **fit_params
    ):  # pylint: disable=unused-argument
        if y is not None and not self.bypass:
            if self.inlier_mask is not None:
                for key, val in fit_params.items():
                    if isinstance(val, (list, np.ndarray)):
                        if len(val) == len(X):
                            fit_params[key] = val[self.inlier_mask]
                return X[self.inlier_mask], y[self.inlier_mask], fit_params
            return X, y
        return X

    @staticmethod
    def fit_predict(X, y=None):  # pylint: disable=unused-argument
        return X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None or self.bypass:
            return self.fit(X).transform(X)
        return self.fit(X, y).transform(X, y, **fit_params)


class Dropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """Drop columns in DataFrame."""
        self.columns = columns

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        return self

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        if self.columns is None:
            return X

        if isinstance(X, pd.DataFrame):
            return X.drop(self.columns, 1)

        if isinstance(X, list):
            return [{k: v for k, v in d.items() if k not in self.columns} for d in X]

        if isinstance(X, dict):
            return [{k: v for k, v in X.items() if k not in self.columns}]

        raise ValueError(
            "Dropper works on DataFrame, dict and iterables"
            "of dicts, {} passed.".format(type(X))
        )


class Individual(RegressorMixin, PredictDataFrameOrDicts):
    def __init__(
        self,
        estimator,
        column,
        max_samples=None,
        min_samples=200,
        fallback_estimator=None,
    ):
        """Make individual model for each value in a column.

        Parameters
        ----------
        estimator: sklearn estimator

        columns: string or list of strings
            will fit one model for each unique combination in columns

        max_samples: int (default=None)
            if the number of samples is larger, max_samples will be selected
            randomly
        """
        self.estimator = estimator
        self.column = column
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.fallback_estimator = fallback_estimator
        super().__init__()

    def fit(self, X, y, **fit_params):
        self._y_dtype = y.dtype
        # fit individual models
        self.models = {}
        args, model_ids = [], []
        cols = X[self.column].values
        for m, _ in X.groupby(self.column):

            # works, but slow
            ids = np.where((cols == np.array(m, dtype=object)).all(axis=1))[0]
            if len(ids) < self.min_samples:
                continue

            if self.max_samples is not None and len(ids) > self.max_samples:
                ids = np.random.permutation(ids)[: self.max_samples]

            args.append(
                (
                    self.estimator,
                    X.iloc[ids].drop(self.column, 1),
                    y.iloc[ids],
                    {k: v[ids] for k, v in fit_params.items()},
                )
            )
            # self.column to be excluded in micro models
            # 3/14/2016 Dominic
            model_ids.append(m)

        estimators = list(map(fit, args))

        for m, est in zip(model_ids, estimators):
            if est is not None:
                self.models[m] = est

        if self.fallback_estimator is not None:
            # pylint: disable=no-member
            self.models[None] = clone(self.fallback_estimator).fit(X, y)
        # pylint: enable=no-member

        return self

    def _empty_predictions(self, X, proba=False):
        if proba:
            return np.empty(len(X), dtype=np.float64)
        return np.zeros(len(X), dtype=self._y_dtype)

    def _predict_df(self, X, proba=False):
        X_with_indices = X.copy()
        # create a copy of X to be used as groupby to get the indice
        # then the original X that without the indices column is used in the
        # actual prediction
        # 3/14/2016 Dominic
        X_with_indices["indices"] = np.arange(len(X_with_indices))
        Xt = X.drop(self.column, 1)
        y_pred = self._empty_predictions(X, proba)
        for m, v in X_with_indices.groupby(self.column):
            if m in self.models:
                model = self.models[m]
                X_use = Xt
            else:
                model = self.models[None]
                X_use = X
            if proba:
                y_pred[v["indices"].values] = model.predict_proba(
                    X_use.iloc[v["indices"]]
                )[:, 1]
            else:
                y_pred[v["indices"].values] = model.predict(
                    X_use.iloc[v["indices"].values]
                )
        return y_pred

    def _predict_dicts(self, X, proba=False):
        model_ids = np.array([[x[c] for c in self.column] for x in X])
        y_pred = self._empty_predictions(X, proba)
        groups = data.group_by(self.column, X)
        for model_id, x in groups.items():
            selected = np.ones(y_pred.shape, dtype=np.bool)
            for i, id_ in enumerate(model_id):
                selected *= model_ids[:, i] == id_

            # pandas uses a string if we group by a single key
            if len(model_id) == 1:
                model_id = model_id[0]

            if model_id in self.models:
                model = self.models[model_id]
                X_use = x
            else:
                model = self.models[None]
                X_use = [X[i] for i in np.where(selected)[0]]

            if proba:
                y_pred[selected] = model.predict_proba(X_use)[:, 1]
            else:
                y_pred[selected] = model.predict(X_use)

        return y_pred


class AverageEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self, estimator: BaseEstimator, n_estimators: int, n_jobs: int, verbose: int = 1
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: SparseFeature, y: Target, **fit_params) -> "AverageEstimator":
        estimators = []
        for seed in range(self.n_estimators):
            estimator = clone(self.estimator)
            # Make sure each estimator uses a different seed for its
            # RandomState object.
            random_state = check_random_state(seed)
            estimator.set_params(random_state=random_state)
            estimators.append(estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        self.estimators_ = parallel(
            delayed(estimator.fit)(X, y, **fit_params) for estimator in estimators
        )
        return self

    def predict(self, X: SparseFeature) -> np.ndarray:
        y_preds = [est.predict(X) for est in self.estimators_]
        return np.mean(y_preds, axis=0)


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Prevent throwing an exception when a classifier is trained with only one class.
    This happen on staging as auto mode does not have enough samples.
    see https://github.com/scikit-learn/scikit-learn/issues/13001#issuecomment-455183782"""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, **fit_params):
        try:
            return self.base_estimator.fit(X, y, **fit_params)
        except ValueError as exc:
            log.exception(exc)
            if not str(exc).startswith(
                "This solver needs samples of at least 2 classes in the data"
            ):
                raise
        finally:
            self.classes_ = self.base_estimator.classes_

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.ones((X.shape[1], 1))
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        if len(self.classes_) == 1:
            return np.full_like(X, self.classes_[0])
        return self.base_estimator.predict(X)
