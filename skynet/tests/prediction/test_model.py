import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, clone

from ...prediction import model


@pytest.fixture
def test_model(get_test_estimator, model_type="a", model_version=1):
    return model.Model(get_test_estimator(), model_type, model_version)


@pytest.fixture
def test_model_type_b(get_test_estimator, model_type="b", model_version=1):
    return model.Model(get_test_estimator(), model_type, model_version)


@pytest.fixture
def X(sample_size):  # noqa: N802
    return pd.DataFrame(np.random.rand(sample_size, 2), columns=["a", "b"])


@pytest.fixture
def y(sample_size):
    y = np.random.rand(sample_size)
    return y


def test_init(test_model):
    assert test_model.model_type == "a"
    assert test_model.model_version == 1
    assert isinstance(test_model.estimator, BaseEstimator)


def test_fit_predict(test_model, X, y):
    # test a simple model and the default
    test_model.fit(X, y)
    y_pred = test_model.predict(X)
    assert len(y) == len(y_pred)


def test_key(test_model, test_model_type_b):

    model1 = test_model
    model2 = clone(test_model)
    assert model1.storage_key == model2.storage_key

    key1 = model.Model.get_storage_key(model_type="a", model_version=1)
    assert model1.storage_key == key1

    model3 = test_model_type_b
    assert model1.storage_key != model3.storage_key


def test_score(X, y, test_model):
    # use all zero targets to get about the same results back each time
    zeros = np.zeros(len(y))
    scores1 = test_model.score(X, zeros, n_jobs=1)
    scores2 = test_model.score(X, zeros, n_jobs=1)
    columns = ["test_score", "train_score"]
    assert_frame_equal(pd.DataFrame(scores1)[columns], pd.DataFrame(scores2)[columns])
