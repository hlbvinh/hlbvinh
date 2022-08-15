from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from ...user.sample import USE_COMFORT_LSTM
from ...utils import nnet_utils
from ...utils.sklearn_wrappers import (
    AverageEstimator,
    DFVectorizer,
    Dropper,
    PipelineWithSampleFiltering,
    Selector,
    StdBasedFeatureAndTargetFilter,
)

STD_CONFIDENT_LEVEL = 2.0


def get_estimator():

    if USE_COMFORT_LSTM:
        estimator = nnet_utils.SequenceEmbeddedRegression(
            hidden_layer_sizes=[128, 128, 128],
            batch_size=1024,
            learning_rate_init=0.05,
            max_iter=10,
            dropout=0.0,
            solver="adam",
            batch_normalization=True,
            weight_decay=0,
            loss="huber",
            timeseries_features=["previous_temperatures", "previous_humidities"],
            initialization_features=["temperature_out", "humidity_out"],
        )
    else:
        estimator = nnet_utils.EmbeddedRegression(
            hidden_layer_sizes=[128, 128, 128],
            batch_size=1024,
            learning_rate_init=0.05,
            max_iter=10,
            dropout=0.0,
            solver="adam",
            batch_normalization=True,
            weight_decay=0,
            loss="huber",
        )
    estimator = AverageEstimator(estimator=estimator, n_estimators=8, n_jobs=1)

    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def get_pipeline(bypass=True, estimator=None):

    pipeline_steps = [
        (
            "filter",
            StdBasedFeatureAndTargetFilter(
                estimator=filter_regressor(),
                columns=["device_id", "user_id", "humidex"],
                bypass=bypass,
                std_confident_level=STD_CONFIDENT_LEVEL,
            ),
        )
    ]
    if estimator is None:
        estimator = get_estimator()
    else:
        # FIXME: is it deprecated?
        pipeline_steps.append(
            (
                "feats",
                make_union(
                    make_pipeline(
                        Dropper(["device_id", "user_id"]),
                        DFVectorizer(sparse=False),
                        SimpleImputer(),
                        StandardScaler(),
                    ),
                    make_pipeline(
                        Selector(["device_id", "user_id"]),
                        DFVectorizer(sparse=True),
                        SimpleImputer(),
                    ),
                ),
            )
        )
    pipeline_steps.append(("fit", estimator))

    return PipelineWithSampleFiltering(pipeline_steps)


def filter_regressor():
    return Pipeline(
        [
            (
                "feats",
                make_union(
                    make_pipeline(
                        Selector(["humidex"]),
                        DFVectorizer(sparse=False),
                        SimpleImputer(),
                        StandardScaler(),
                    ),
                    make_pipeline(
                        Selector(["device_id", "user_id"]),
                        DFVectorizer(sparse=True),
                        SimpleImputer(),
                    ),
                ),
            ),
            ("fit", LinearRegression()),
        ]
    )


def get_params():
    prefix = "estimator__fit__regressor__estimator"
    params = {"learning_rate_init": [0.1, 0.05, 0.01]}
    return {f"{prefix}__{k}": v for k, v in params.items()}
