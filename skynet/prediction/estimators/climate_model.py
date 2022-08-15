from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from ...utils import nnet_utils, sklearn_wrappers

# Chosen based on experiments.
# Outside temperature and difference between current and previous
# set temperature are determining the starting state of RNN.
INITIALIZATION_FEATURES = [
    "temperature_out",
    "temperature_out_mean_day",
    "temperature_set_difference",
]


def get_estimator():

    estimator = nnet_utils.SequenceEmbeddedRegression(
        timeseries_features=["previous_temperatures"],
        initialization_features=INITIALIZATION_FEATURES,
        hidden_layer_sizes=[128, 128, 128],
        batch_size=1024,
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=10,
        dropout=0.5,
        batch_normalization=True,
        weight_decay=0.00,
        loss="huber",
        no_embedding=["mode"],
    )

    estimator = sklearn_wrappers.AverageEstimator(
        estimator=estimator, n_estimators=4, n_jobs=1
    )

    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def get_pipeline(estimator=None):

    pipeline_steps = [
        ("set_temperature_conversion", sklearn_wrappers.SetTemperatureConverter()),
        ("set_temperature_difference", sklearn_wrappers.TemperatureSetDifference()),
    ]

    if estimator is None:
        estimator = get_estimator()
    else:
        pipeline_steps.append(
            (
                "feats",
                make_union(
                    make_pipeline(
                        sklearn_wrappers.Selector(["mode"]),
                        sklearn_wrappers.DFVectorizer(sparse=False),
                        SimpleImputer(),
                    ),
                    make_pipeline(
                        sklearn_wrappers.Selector(["appliance_id"]),
                        sklearn_wrappers.DFVectorizer(sparse=False),
                        SimpleImputer(),
                        StandardScaler(),
                    ),
                    make_pipeline(
                        sklearn_wrappers.Dropper(["appliance_id", "mode"]),
                        sklearn_wrappers.DFVectorizer(sparse=False),
                        SimpleImputer(),
                        StandardScaler(),
                    ),
                ),
            )
        )

    pipeline_steps.append(("fit", estimator))

    return Pipeline(pipeline_steps)


def get_params():
    prefix = "estimator__fit__regressor__estimator"
    grid = {}
    return {f"{prefix}__{name}": params for name, params in grid.items()}
