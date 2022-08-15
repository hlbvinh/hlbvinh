import abc
import copy

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection


def scores_rolling_mean(
    scores, validation_split_interval, average_interval, return_error=False
):
    try:
        period = int(average_interval / validation_split_interval)
    except TypeError:
        period = int(
            pd.Timedelta(average_interval) / pd.Timedelta(validation_split_interval)
        )

    padding = scores[-(period - 1) :]
    padded_scores = list(padding) + list(scores)
    scores_series = pd.Series(padded_scores)
    rolling_mean = scores_series.rolling(window=period).mean().dropna()
    if return_error:
        rolling_error = scores_series.rolling(window=period).std().dropna()
        return rolling_mean.values, rolling_error.values
    return rolling_mean.values


def scores_resample(scores, validation_split_interval, average_interval):
    try:
        period = average_interval / validation_split_interval
    except TypeError:
        period = pd.Timedelta(average_interval) / pd.Timedelta(
            validation_split_interval
        )
    return pd.Series(scores).groupby(lambda x: x // period).mean().values


def scores_avg(scores, validation_split_interval, average_interval, rolling_window):
    resampled_scores = scores_resample(
        scores, validation_split_interval, average_interval
    )
    avg_scores = scores_rolling_mean(resampled_scores, average_interval, rolling_window)
    return avg_scores


def reseample_grid_scores(
    grid_scores_dict, validation_split_interval, average_interval, rolling_window
):
    new_dict = copy.copy(grid_scores_dict)
    cv_validation_scores = scores_avg(
        grid_scores_dict["cv_validation_scores"],
        validation_split_interval,
        average_interval,
        rolling_window,
    )
    mean_validation_score = np.mean(cv_validation_scores)
    new_dict["cv_validation_scores"] = cv_validation_scores
    new_dict["mean_validation_score"] = mean_validation_score
    return new_dict


def convert_cv_results_to_grid_scores_as_dict(cv_results):
    # cv_results_ does not have an easy way of getting the cross validation
    # test scores as it was with the now deprecated grid_scores_
    n_fold = len(
        [
            key
            for key in cv_results
            if key.startswith("split") and key.endswith("_test_score")
        ]
    )

    return [
        {"parameters": params, "cv_validation_scores": np.array(cv_validation_scores)}
        for params, *cv_validation_scores in zip(
            cv_results["params"],
            *(cv_results[f"split{i}_test_score"] for i in range(n_fold)),
        )
    ]


def grid_search_score_dataframe(
    gs, validation_split_interval=None, average_interval=None, rolling_interval=None
):
    as_dict = convert_cv_results_to_grid_scores_as_dict(gs.cv_results_)
    if average_interval is None or rolling_interval is None:
        dicts = as_dict
    else:
        dicts = [
            reseample_grid_scores(
                d, validation_split_interval, average_interval, rolling_interval
            )
            for d in as_dict
        ]
    df = pd.DataFrame([d["parameters"] for d in dicts])
    scores = np.array([d["cv_validation_scores"] for d in dicts])
    df.loc[:, "mean"] = scores.mean(axis=1)
    df.loc[:, "std"] = scores.std(axis=1)
    return df.iloc[np.argsort(df["mean"])]


def permutation_feature_importance(
    estimator, X, y, n_splits, feature_groups=None, normalize=True
):
    """Generate permutation_feature_importance based on input data and
    estimator.

    Parameters
    ----------
    estimator: str
        Estimator used in training data.

    X: DataFrame
        DataFrame of features.

    y: Series
        Pandas series of target variable.

    n_splits: int
        Number of folds in running cross validation and fitting

    feature_groups: dict
        Dictionary containing list of features that are to be grouped together
        Default = None

    Returns
    -------
    DataFrame
        Permutation_feature_importance with PFI_score and normalized_PFI_score.
    """
    if feature_groups is None:
        feature_groups = {f: f for f in X.columns}
    PFI_score = []
    kf = model_selection.KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        base_metric_score = metrics.mean_absolute_error(y_test, y_pred)
        shuffle_metric_score = []
        for shuffle_feature in feature_groups:
            shuffle_X = X_test.copy()
            avg_feature_value = np.mean(
                shuffle_X[feature_groups[shuffle_feature]].values
            )
            shuffle_X[feature_groups[shuffle_feature]] = avg_feature_value
            y_pred_shuffle = estimator.predict(shuffle_X)
            shuffle_metric_score.append(
                metrics.mean_absolute_error(y_test, y_pred_shuffle)
            )
        pfi_score = [base_metric_score - score for score in shuffle_metric_score]
        PFI_score.append(pfi_score)
    pfi_score = np.mean(PFI_score, axis=0)
    if normalize:
        pfi_score = np.abs(pfi_score) / np.sum(np.abs(pfi_score))
    return pd.Series(pfi_score, index=feature_groups.keys())


class BaseSeriesFold(metaclass=abc.ABCMeta):
    def __init__(
        self, n_train=None, validation_split_interval=None, validation_interval=None
    ):
        self._n_train = n_train
        self._validation_split_interval = validation_split_interval
        self._validation_interval = validation_interval
        self._split_labels = []

    def split(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        groups = getattr(groups, "values", groups)

        sorted_group_idx = np.argsort(groups)
        groups = groups[sorted_group_idx]

        for start, end in self.get_split_items(groups):
            start_idx = groups.searchsorted(start, side="right")
            end_idx = groups.searchsorted(end, side="right")
            test_idx = sorted_group_idx[start_idx:end_idx]
            if self._n_train > 0:
                train_idx = sorted_group_idx[start_idx - self._n_train : start_idx]
            else:
                train_idx = sorted_group_idx[0:start_idx]
            if np.any(test_idx) and np.any(train_idx):
                yield train_idx, test_idx

                self._split_labels.append(end)

    def get_n_splits(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        return len(list(self.split(X, groups=groups)))

    @abc.abstractmethod
    def get_split_values(self, groups):
        """Split locations.

        Returns
        -------
        array-like: points where to split groups
        """

    def get_split_items(self, groups):
        split_values = self.get_split_values(groups)
        for st, ed in zip(split_values[:-1], split_values[1:]):
            yield st, ed


class TimeSeriesFold(BaseSeriesFold):
    def get_split_values(self, groups):
        end = safe_indexing(groups, -1)
        split_timestamp = pd.to_datetime(end) - self._validation_interval
        split_timestamps = pd.date_range(
            split_timestamp, end, freq=self._validation_split_interval
        ).values
        return split_timestamps


def safe_indexing(X, indices):
    """taken from sklearn.utils._safe_indexing"""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    if hasattr(X, "shape"):
        if isinstance(indices, tuple):
            indices = list(indices)
        return X[indices]
    if np.isscalar(indices) or isinstance(indices, slice):
        return X[indices]
    return [X[i] for i in indices]
