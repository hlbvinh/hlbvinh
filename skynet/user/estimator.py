from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..utils.sklearn_wrappers import DFVectorizer


def get_pipeline():
    return Pipeline(
        [
            ("dv", DFVectorizer()),
            ("imp", SimpleImputer()),
            ("scale", StandardScaler(with_mean=False)),
            ("fit", SVR(C=100, epsilon=0.8, verbose=2)),
        ]
    )
