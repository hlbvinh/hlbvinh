from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn import linear_model

from ..utils.sklearn_wrappers import DFVectorizer


def get_pipeline():
    return Pipeline(
        [
            ("dv", DFVectorizer(sparse=False)),
            ("imp", SimpleImputer()),
            ("scale", StandardScaler(with_mean=True)),
            ("fit", linear_model.Ridge()),
        ]
    )
