import numpy as np
import pandas as pd

import pytest
from sklearn.linear_model import LinearRegression

from ...prediction import climate_model, mode_model, mode_model_util
from ...prediction.estimators import (
    climate_model as climate_model_estimator,
    comfort_model_estimator,
)
from ...user import comfort_model, sample
from ...utils import testing
from ...utils.compensation import COMPENSATE_COLUMN
from ...utils.enums import Power

np.random.seed(5)


@pytest.fixture
def sample_size():
    return 10


@pytest.fixture
def split_score_sample_size():
    """
    some sample are meant to be use for test scoring method of a prediction
    model, the we will use a a bit bigger sample size such that the
    corresponding test has sample to split
    """
    return 30


@pytest.fixture
def get_test_estimator():
    return LinearRegression


@pytest.fixture
def feature_sigma():
    """
    For unique each mode model target, we want to specify some indoor and
    outdoor's expeced standard deviation
    """
    return {"humidex": 0.1, "temperature_out": 0.1, "target": 0.4}


@pytest.fixture
def mode_feature_spec():
    """
    For unique each mode model target, we want to specify some indoor and
    outdoor's expected average values in generating sample for tests
    """
    return [
        {"modes": ["auto"], "target": 1, "humidex": 23.0, "temperature_out": 10.0},
        {"modes": ["cool"], "target": -10, "humidex": 35.0, "temperature_out": 40.0},
        {"modes": ["dry"], "target": -6, "humidex": 30.0, "temperature_out": 30.0},
        {"modes": ["fan"], "target": -4, "humidex": 25.0, "temperature_out": 20.0},
        {"modes": ["heat"], "target": 2, "humidex": 20.0, "temperature_out": 5.0},
    ]


@pytest.fixture
def mode_model_targets():
    """
    We are testing all a/c modes that specified in mode_model_util
    """
    return mode_model_util.MULTIMODES


def create_configs(spec, label, sigma):
    return {"mean": spec[label], "sigma": sigma[label], "name": label, "type": "normal"}


@pytest.fixture
def feature_configs(feature_sigma, mode_feature_spec):
    cols = ["target", "temperature_out"]
    return [
        [create_configs(spec, q, feature_sigma) for q in cols]
        for spec in mode_feature_spec
    ]


@pytest.fixture
def mode_dataset(sample_size, feature_configs, mode_feature_spec, mode_model_targets):
    features_col = mode_model.FEATURE_COLUMNS
    xs = []
    for mode_feature_configs, target in zip(feature_configs, mode_model_targets):
        x = testing.gen_feature_matrix(
            features_col, sample_size, defined_feature_configs=mode_feature_configs
        )
        x["power_hist"] = np.random.choice([Power.ON, Power.OFF], sample_size)
        x["mode_hist"] = np.random.choice(
            ["cool", "heat", "auto", "fan", "dry"], sample_size
        )
        x["mode"] = target
        x["origin"] = np.random.choice(
            ["irdeployment", "reverse", "skynet"], sample_size
        )
        xs.append(x)
    X = pd.concat(xs)
    n_samples = len(X)
    X["appliance_id"] = testing.gen_feature_string("appliance", n_samples, 5, 5)
    X[COMPENSATE_COLUMN] = testing.gen_feature_bool(COMPENSATE_COLUMN, n_samples)
    y = testing.gen_feature_matrix(
        climate_model.QUANTITIES, sample_size * len(mode_feature_spec)
    )
    return mode_model.make_mode_model_dataset(X, y)


@pytest.fixture()
def climate_features(split_score_sample_size):
    cols = climate_model.FEATURE_COLUMNS
    X = pd.DataFrame(np.random.rand(split_score_sample_size, len(cols)), columns=cols)
    X[COMPENSATE_COLUMN] = testing.gen_feature_bool(COMPENSATE_COLUMN, len(X))
    return X


@pytest.fixture
def climate_targets(split_score_sample_size):
    return np.random.rand(split_score_sample_size, len(climate_model.QUANTITIES))


@pytest.fixture
def climate_model_pipeline(get_test_estimator):
    return climate_model_estimator.get_pipeline(estimator=get_test_estimator())


@pytest.fixture
def untrained_climate_model(climate_model_pipeline):
    return climate_model.ClimateModel(
        estimator=climate_model_pipeline, model_type="a", model_version=1
    )


@pytest.fixture
def untrained_climate_model_a(climate_model_pipeline):
    return climate_model.ClimateModel(
        estimator=climate_model_pipeline, model_type="a", model_version=1
    )


@pytest.fixture
def untrained_climate_model_b(climate_model_pipeline):
    return climate_model.ClimateModel(
        estimator=climate_model_pipeline, model_type="b", model_version=1
    )


@pytest.fixture
def trained_climate_model(climate_model_pipeline, climate_features, climate_targets):
    return climate_model.ClimateModel(estimator=climate_model_pipeline).fit(
        climate_features, climate_targets
    )


@pytest.fixture
def mode_features(mode_dataset):
    return mode_dataset[0]


@pytest.fixture
def mode_targets(mode_dataset):
    return mode_dataset[1]


@pytest.fixture
def trained_mode_model(mode_features, mode_targets):
    return mode_model.ModeModel([tuple(mode_model_util.MULTIMODES)]).fit(
        mode_features, mode_targets
    )


@pytest.fixture
def comfort_model_pipeline(get_test_estimator):
    return comfort_model_estimator.get_pipeline(estimator=get_test_estimator())


@pytest.fixture
def raw_comfort_dataset(sample_size):
    datapoint = {
        "compensated": 1.0,
        "device_id": "78b65c28-b840-4a82-b097-d6fd94b9200d",
        "feedback": 3.0,
        "humidity": 67.6774997711,
        "humidity_out": 74.1249998411,
        "humidity_raw": 51.6774997711,
        "humidity_refined": 67.6774997711,
        "luminosity": 196.5,
        "pircount": 0.0,
        "pirload": 0.0,
        "previous_temperatures": [24.595000267],
        "previous_humidities": [67.6774997711],
        "temperature": 24.595000267,
        "temperature_out": 29.1875,
        "temperature_raw": 28.595000267,
        "temperature_refined": 24.595000267,
        "timestamp": "2018-06-12 02:47:00",
        "tod_cos": 0.3131638065,
        "tod_sin": -0.9496991262,
        "tow_cos": 0.9637968442,
        "tow_sin": 0.2666376626,
        "toy_cos": 0.3288540583,
        "toy_sin": -0.9443807539,
        "type": "user_feedback",
        "user_id": "a0e755aa-6368-4a29-80d2-dba932b6d2d2",
    }
    return pd.DataFrame([datapoint] * sample_size)


@pytest.fixture
def comfort_dataset(raw_comfort_dataset):
    return sample.prepare_dataset(raw_comfort_dataset)


@pytest.fixture
def comfort_features(comfort_dataset):
    return comfort_model.split(comfort_dataset)[0]


@pytest.fixture
def comfort_targets(comfort_dataset):
    return comfort_model.split(comfort_dataset)[1]


@pytest.fixture
def untrained_comfort_model(comfort_model_pipeline):
    return comfort_model.ComfortModel(estimator=comfort_model_pipeline)


@pytest.fixture
def trained_comfort_model(comfort_features, comfort_targets, untrained_comfort_model):
    return untrained_comfort_model.fit(comfort_features, comfort_targets)
