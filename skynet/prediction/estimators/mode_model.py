import itertools

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ...utils.data import drop_from_sequence
from ...utils.sklearn_wrappers import (
    ClassifierWrapper,
    DFVectorizer,
    Individual,
    Selector,
)


def mini_mode_est(features):
    return Pipeline(
        [
            (
                "features",
                make_union(
                    make_pipeline(
                        Selector(drop_from_sequence(features, ["appliance_id"])),
                        DFVectorizer(sparse=False),
                        SimpleImputer(),
                        PolynomialFeatures(degree=2),
                        StandardScaler(),
                    ),
                    make_pipeline(
                        Selector(["appliance_id"]),
                        DFVectorizer(sparse=True),
                        SimpleImputer(),
                    ),
                ),
            ),
            (
                "fit",
                ClassifierWrapper(
                    LogisticRegression(
                        C=0.2,
                        random_state=1,
                        solver="liblinear",
                        multi_class="ovr",
                        penalty="l2",
                        max_iter=300,
                    )
                ),
            ),
        ]
    )


def get_mini_pipeline(features):
    group_column = "quantity"
    return Individual(
        mini_mode_est(drop_from_sequence(features, [group_column])),
        column=[group_column],
        max_samples=None,
        min_samples=0,
        fallback_estimator=None,
    )


def get_mini_weights(model):
    rets = {}
    for modes, mini_estimator in model.mini_estimators.items():
        ret = {}
        for q, m in mini_estimator.models.items():
            if not hasattr(m.steps[-1][1].base_estimator, "coef_"):
                continue
            weights = m.steps[-1][1].base_estimator.coef_
            weights = np.sum(weights, axis=0)
            union = m.steps[0][1]
            names = itertools.chain(
                *(tf[1].steps[1][1].dv_.feature_names_ for tf in union.transformer_list)
            )

            appliance_id_weights = []
            weights_summary = {}

            for name, weight in zip(names, weights):
                if name.startswith("appliance_id="):
                    appliance_id_weights.append(weight)
                else:
                    weights_summary[name] = weight

            weights_summary["appliance_id_mean"] = np.mean(appliance_id_weights)
            weights_summary["appliance_id_sparsity"] = np.count_nonzero(
                appliance_id_weights
            ) / len(appliance_id_weights)

            ret[q] = weights_summary

        rets[modes] = ret

    return rets


def get_params():
    return {("cool", "heat"): {"estimator__fit__base_estimator__C": [0.01, 0.05]}}
