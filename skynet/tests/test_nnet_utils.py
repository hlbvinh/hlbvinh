import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from ..utils import nnet_utils
from ..utils.sklearn_wrappers import DFVectorizer


def test_pytorch():
    a = torch.ones(2)
    b = 2 * a + 1
    assert np.allclose([3, 3], b.numpy())


@pytest.fixture(params=["categorical", "no categorical"])
def dataset(request):
    if request.param == "categorical":
        return pd.DataFrame(
            dict(
                device_id=["a", "b", "c"],
                temperature=[15, 16, 18],
                target_1=[0, 1, 2],
                target_2=[0, 1, 2],
            )
        )
    return pd.DataFrame(
        dict(temperature=[15, 16, 18], target_1=[0, 1, 2], target_2=[0, 1, 2])
    )


@pytest.fixture
def model():
    return nnet_utils.EmbeddedRegression()


@pytest.fixture
def trained_model(dataset, model):
    X = dataset.drop(columns="target_1")
    y = dataset["target_1"]
    model.fit(X, y)
    return model


def test_train_and_predict(dataset, trained_model):
    X = dataset.drop(columns="target_1")
    y = dataset["target_1"]
    m1 = trained_model

    m2 = make_pipeline(DFVectorizer(), LinearRegression())
    m2.fit(X, y)

    data = [X, X.to_dict("records"), X.to_dict("records")[0]]
    for d in data:
        p1, p2 = m1.predict(d), m2.predict(d)
        assert p1.shape == p2.shape
        assert isinstance(p1[0], type(p2[0]))


def test_multioutput_regression(dataset, model):
    X = dataset.drop(columns=["target_1", "target_2"])
    y = dataset[["target_1", "target_2"]]
    model.fit(X, y)

    r = model.predict(X)
    assert r.shape == (len(X), y.shape[1])


def test_pickled_prediction(dataset, trained_model):
    p = pickle.dumps(trained_model, -1)
    m = pickle.loads(p)
    X = dataset.drop(columns="target_1")
    m.predict(X)


def test_sequencernn():
    batch, seq, input_dim = 20, 10, 3
    input_ = torch.randn(batch, seq, input_dim)
    tensor_size = torch.tensor([seq] * batch)  # pylint: disable=not-callable
    init_dim = 5
    init = torch.randn(batch, init_dim)

    output_dim = 1
    stacked_layers = 2
    m = nnet_utils.SequenceRNN(
        init_dim,
        input_dim,
        stacked_layers=stacked_layers,
        timeseries_output_dim=output_dim,
    )
    assert tuple(m(input_, tensor_size, init).shape) == (batch, output_dim)


def test_sequencereshaper():
    x = [[[10, 10], [12, 12]], [[1, 1, 1], [3, 3, 3]]]
    r = nnet_utils.SequenceReshaper()

    lower_dim_x = r.to_1d(x)
    expected = np.array([[10, 12], [10, 12], [1, 3], [1, 3], [1, 3]])
    assert (lower_dim_x == expected).all()

    lower_dim_x_tensor = r.to_tensor(lower_dim_x)
    expected_tensor = [
        torch.tensor([[10, 12], [10, 12]]),  # pylint: disable=not-callable
        torch.tensor([[1, 3], [1, 3], [1, 3]]),  # pylint: disable=not-callable
    ]
    assert all(torch.equal(x, y) for x, y in zip(lower_dim_x_tensor, expected_tensor))


def test_is_string():
    df = pd.DataFrame([dict(device_id="1234", prev_temperatures=[1, 2, 3])])
    assert nnet_utils.string_columns(df) == ["device_id"]


@pytest.fixture
def sequence_dataset():
    return pd.DataFrame(
        dict(
            prev_temperatures=[[1, 2, 3], [2, 3], [1]],
            prev_humidities=[[1, 2, 3], [2, 3], [1]],
            weather=[15, 16, 18],
            mode_hist=["off", "on", "on"],
            target_1=[0, 1, 2],
            target_2=[0, 1, 2],
        )
    )


@pytest.fixture
def sequence_model():
    return nnet_utils.SequenceEmbeddedRegression(
        timeseries_features=["prev_temperatures", "prev_humidities"],
        initialization_features=["weather", "mode_hist"],
    )


@pytest.fixture
def trained_sequence_model(sequence_dataset, sequence_model):
    X = sequence_dataset.drop(columns="target_1")
    y = sequence_dataset["target_1"]
    sequence_model.fit(X, y)
    return sequence_model


def test_train_and_predict_sequence(sequence_dataset, trained_sequence_model):
    X = sequence_dataset.drop(columns="target_1")
    m1 = trained_sequence_model

    data = [X, X.to_dict("records"), X.to_dict("records")[0]]
    for d in data:
        p1 = m1.predict(d)
        if isinstance(d, dict):
            assert len(p1) == 1
        else:
            assert len(p1) == len(d)


def test_delta_array():
    input_ = [[[1, 2, 3], [1, 2, 3]], [[4, 5], [4, 5]]]
    expected = [[[2, 1, 0], [2, 1, 0]], [[1, 0], [1, 0]]]
    result = nnet_utils.delta_array(input_)
    assert result == expected
