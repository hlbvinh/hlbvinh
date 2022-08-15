import os

from sklearn import model_selection
from sklearn.base import BaseEstimator, clone

from ..utils import misc
from ..utils.log_util import get_logger, kv_format
from .estimator import get_pipeline

log = get_logger(__name__)
DIR = misc.get_dir("data")

try:
    os.mkdir(DIR)
except OSError:
    pass


class Model(BaseEstimator):
    """Model base class, wrapper around sklearn estimators."""

    default_estimator = get_pipeline()

    @classmethod
    def get_storage_key(cls, model_type, model_version):
        return {"model_type": model_type, "model_version": model_version}

    def __init__(self, estimator=None, model_type=None, model_version=None):
        if estimator is None:
            estimator = clone(self.default_estimator)
        self.estimator = estimator
        self.model_type = model_type
        self.model_version = model_version

    @classmethod
    def log_features(cls, X):
        means = X._get_numeric_data().mean().to_dict()
        log.info(f"{cls.__name__} feature means: {kv_format(means, digits=3)}")

    def fit(self, X, y, **fit_params):
        """Fit the model.

        Parameters
        ----------
        X : DataFrame
            feature matrix

        y : DataFrame or Series
            targets
        """
        self.log_features(X)
        self.estimator.fit(X, y, **fit_params)
        return self

    @misc.timeit()
    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : DataFrame
            feature matrix
        """
        return self.estimator.predict(X)

    @property
    def storage_key(self):
        """Returns key to be used with storage models."""
        return self.get_storage_key(
            model_type=self.model_type, model_version=self.model_version
        )

    def save(self, storage):
        """Save model via storage module."""
        storage.save(self.storage_key, self)

    def score(
        self,
        X,
        y,
        n_jobs=1,
        n_folds=3,
        cv=None,
        fit_params=None,
        timestamps=None,
        scoring="neg_mean_absolute_error",
        random_state=0,
        return_train_score=True,
    ):
        """CV score entire dataset."""
        if cv is None:
            cv = model_selection.KFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )

        score = model_selection.cross_validate(
            self.estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            groups=timestamps,
            fit_params=fit_params,
            return_train_score=return_train_score,
        )

        return score

    def __repr__(self, *args):
        data = ", ".join(f"{k}={repr(v)}" for k, v in self.storage_key.items())
        return f"{type(self).__name__}({data})"
