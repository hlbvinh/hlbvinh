import numpy as np
import pytest

from ...prediction import predict

np.random.seed(1)


@pytest.fixture
def predictor(trained_climate_model):
    return predict.Predictor(trained_climate_model)


def test_predict(predictor, climate_features):
    states = predict.generate_on_signals()
    for _, s in climate_features.iterrows():
        history_features = s.to_dict()
        pred = predictor.predict(history_features, states)
        assert len(pred) == len(states)
