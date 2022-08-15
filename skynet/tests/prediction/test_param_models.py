import pytest
from sklearn.model_selection import ParameterGrid

from ...prediction import mode_model, climate_model, mode_model_util
from ...user import comfort_model
from ...prediction.estimators import comfort_model_estimator
from ...prediction import estimators


@pytest.fixture(params=["mode_model", "climate_model", "comfort_model"])
def estimator(request):

    model = None
    grid_params = None
    parameter_grid = ParameterGrid
    if request.param == "mode_model":
        model = mode_model.ModeModel([tuple(mode_model_util.MULTIMODES)])
        grid_params = estimators.mode_model.get_params()
        parameter_grid = mode_model_util.ModeModelParameterGrid

    elif request.param == "climate_model":
        model = climate_model.ClimateModel()
        grid_params = estimators.climate_model.get_params()

    elif request.param == "comfort_model":
        model = comfort_model.ComfortModel()
        grid_params = comfort_model_estimator.get_params()

    return model, grid_params, parameter_grid


def test_grid_search_params(estimator):
    model, grid_params, parameter_grid = estimator

    for params in parameter_grid(grid_params):
        model.set_params(**params)
