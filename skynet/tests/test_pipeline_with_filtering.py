"""
Test the pipeline module.
"""
import numpy as np
from scipy import sparse

import pytest

from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from ..utils.sklearn_wrappers import PipelineWithSampleFiltering

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

# Mathis 2016 09 27 sklearn's Bunch objects cause no-member linting errors
# pylint: disable=no-member


class IncorrectT:
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class T(IncorrectT):
    def fit(self, X, y):  # pylint: disable=unused-argument
        return self

    def get_params(self, deep=False):
        # vulture reports unused parameters
        # next line to avoid it
        if deep or not deep:
            pass
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class Filtr(IncorrectT):
    def fit(self, X, y):  # pylint: disable=unused-argument
        return self

    def fit_predict(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        return self.fit(X, y)

    @staticmethod
    def transform(X, y=None, **fit_params):
        if y is None:
            return X
        fit_params = {key: val[1::] for key, val in fit_params.items()}
        return X[1::], y[1::], fit_params

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y).transform(X, y, **fit_params)

    def get_params(self, deep=False):
        # vulture reports unused parameters
        # next line to avoid it
        if deep or not deep:
            pass
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class TransfT(T):
    @staticmethod
    def transform(X, y=None):  # pylint: disable=unused-argument
        return X

    @staticmethod
    def inverse_transform(X):
        return X


class FitParamT:
    """Mock classifier"""

    def __init__(self):
        self.successful = False

    def fit(self, X, y, should_succeed=False):  # pylint: disable=unused-argument
        self.successful = should_succeed

    def predict(self, X):  # pylint: disable=unused-argument
        return self.successful


def test_pipeline_init():
    # Check that we can't instantiate pipelines with objects without fit
    # method
    with pytest.raises(TypeError):
        PipelineWithSampleFiltering(["svc", IncorrectT])
    # Smoke test with only an estimator
    clf = T()
    pipe = PipelineWithSampleFiltering([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC(gamma="scale")
    filter1 = SelectKBest(f_classif)
    pipe = PipelineWithSampleFiltering([("anova", filter1), ("svc", clf)])

    # Check that we can't use the same stage name twice
    with pytest.raises(ValueError):
        PipelineWithSampleFiltering(
            [("svc", SVC(gamma="scale")), ("svc", SVC(gamma="scale"))]
        )

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    with pytest.raises(ValueError):
        pipe.set_params(anova__C=0.1)

    # Test clone
    pipe2 = clone(pipe)
    assert pipe["svc"] is not pipe2["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


def test_pipeline_methods_anova():
    # Test the various methods of the pipeline (anova).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression(solver="liblinear", multi_class="auto")
    filter1 = SelectKBest(f_classif, k=2)
    pipe = PipelineWithSampleFiltering([("anova", filter1), ("logistic", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_fit_params():
    # Test that the pipeline can take fit parameters
    pipe = PipelineWithSampleFiltering([("transf", TransfT()), ("clf", FitParamT())])
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert pipe.predict(None)
    # and transformer params should not be changed
    assert pipe["transf"].a is None
    assert pipe["transf"].b is None


def test_pipeline_raise_set_params_error():
    # Test pipeline raises set params error message for nested models.
    pipe = PipelineWithSampleFiltering([("cls", LinearRegression())])

    with pytest.raises(ValueError):
        pipe.set_params(fake="nope")

    # nested model check
    with pytest.raises(ValueError):
        pipe.set_params(fake__estimator="nope")


def test_pipeline_methods_pca_svm():
    # Test the various methods of the pipeline (pca + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Test with PCA + SVC
    clf = SVC(gamma="scale", probability=True, random_state=0)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    pipe = PipelineWithSampleFiltering([("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


def test_pipeline_methods_preprocessing_svm():
    # Test the various methods of the pipeline (preprocessing + svm).
    iris = load_iris()
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    clf = SVC(
        gamma="scale", probability=True, random_state=0, decision_function_shape="ovr"
    )

    for preprocessing in [scaler, pca]:
        pipe = PipelineWithSampleFiltering(
            [("preprocess", preprocessing), ("svc", clf)]
        )
        pipe.fit(X, y)

        # check shapes of various prediction functions
        predict = pipe.predict(X)
        assert predict.shape == (n_samples,)

        proba = pipe.predict_proba(X)
        assert proba.shape == (n_samples, n_classes)

        log_proba = pipe.predict_log_proba(X)
        assert log_proba.shape == (n_samples, n_classes)

        decision_function = pipe.decision_function(X)
        assert decision_function.shape == (n_samples, n_classes)

        pipe.score(X, y)


def test_fit_predict_on_pipeline():
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # transform and clustering steps separately
    iris = load_iris()
    scaler = StandardScaler()
    km = KMeans(random_state=0)

    # first compute the transform and clustering step separately
    scaled = scaler.fit_transform(iris.data)
    separate_pred = km.fit_predict(scaled)

    # use a pipeline to do the transform and clustering in one step
    pipe = PipelineWithSampleFiltering([("scaler", scaler), ("Kmeans", km)])
    pipeline_pred = pipe.fit_predict(iris.data)

    np.testing.assert_array_almost_equal(pipeline_pred, separate_pred)


def test_feature_union():
    # basic sanity check for feature union
    iris = load_iris()
    X = iris.data
    X -= X.mean(axis=0)
    y = iris.target
    svd = TruncatedSVD(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    fs = FeatureUnion([("svd", svd), ("select", select)])
    fs.fit(X, y)
    X_transformed = fs.transform(X)
    assert X_transformed.shape == (X.shape[0], 3)

    # check if it does the expected thing
    np.testing.assert_array_almost_equal(X_transformed[:, :-1], svd.fit_transform(X))
    np.testing.assert_array_equal(
        X_transformed[:, -1], select.fit_transform(X, y).ravel()
    )

    # test if it also works for sparse input
    # We use a different svd object to control the random_state stream
    fs = FeatureUnion([("svd", svd), ("select", select)])
    X_sp = sparse.csr_matrix(X)
    X_sp_transformed = fs.fit_transform(X_sp, y)
    np.testing.assert_array_almost_equal(X_transformed, X_sp_transformed.toarray())

    # test setting parameters
    fs.set_params(select__k=2)
    assert fs.fit_transform(X, y).shape == (X.shape[0], 4)

    # test it works with transformers missing fit_transform
    fs = FeatureUnion([("mock", TransfT()), ("svd", svd), ("select", select)])
    X_transformed = fs.fit_transform(X, y)
    assert X_transformed.shape == (X.shape[0], 8)


def test_make_union():
    pca = PCA(svd_solver="full")
    mock = TransfT()
    fu = make_union(pca, mock)
    names, transformers = zip(*fu.transformer_list)
    assert names == ("pca", "transft")
    assert transformers == (pca, mock)


def test_pipeline_transform():
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = PipelineWithSampleFiltering([("pca", pca)])

    # test transform and fit_transform:
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    np.testing.assert_array_almost_equal(X_trans, X_trans2)
    np.testing.assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    np.testing.assert_array_almost_equal(X_back, X_back2)


def test_pipeline_fit_transform():
    # Test whether pipeline works with a transformer missing fit_transform
    iris = load_iris()
    X = iris.data
    y = iris.target
    transft = TransfT()
    pipeline = PipelineWithSampleFiltering([("mock", transft)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transft.fit(X, y).transform(X)
    np.testing.assert_array_almost_equal(X_trans, X_trans2)


def test_make_pipeline():
    t1 = ("transft-1", TransfT())
    t2 = ("transft-2", TransfT())

    pipe = PipelineWithSampleFiltering([t1, t2])
    assert isinstance(pipe, PipelineWithSampleFiltering)
    assert pipe.steps[0][0] == "transft-1"
    assert pipe.steps[1][0] == "transft-2"

    pipe = PipelineWithSampleFiltering([t1, t2, ("fitparamt", FitParamT())])
    assert isinstance(pipe, PipelineWithSampleFiltering)
    assert pipe.steps[0][0] == "transft-1"
    assert pipe.steps[1][0] == "transft-2"
    assert pipe.steps[2][0] == "fitparamt"


def test_feature_union_weights():
    # test feature union with transformer weights
    iris = load_iris()
    X = iris.data
    y = iris.target
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    select = SelectKBest(k=1)
    # test using fit followed by transform
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    fs.fit(X, y)
    X_transformed = fs.transform(X)
    # test using fit_transform
    fs = FeatureUnion(
        [("pca", pca), ("select", select)], transformer_weights={"pca": 10}
    )
    X_fit_transformed = fs.fit_transform(X, y)
    # test it works with transformers missing fit_transform
    fs = FeatureUnion(
        [("mock", TransfT()), ("pca", pca), ("select", select)],
        transformer_weights={"mock": 10},
    )
    X_fit_transformed_wo_method = fs.fit_transform(X, y)
    # check against expected result

    # We use a different pca object to control the random_state stream
    np.testing.assert_array_almost_equal(
        X_transformed[:, :-1], 10 * pca.fit_transform(X)
    )
    np.testing.assert_array_equal(
        X_transformed[:, -1], select.fit_transform(X, y).ravel()
    )
    np.testing.assert_array_almost_equal(
        X_fit_transformed[:, :-1], 10 * pca.fit_transform(X)
    )
    np.testing.assert_array_equal(
        X_fit_transformed[:, -1], select.fit_transform(X, y).ravel()
    )
    assert X_fit_transformed_wo_method.shape == (X.shape[0], 7)


@pytest.mark.skip("somehow failing on CI when updating from 0.19.2 to 0.20.2")
def test_feature_union_parallel():
    # test that n_jobs work for FeatureUnion
    X = JUNK_FOOD_DOCS

    fs = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ]
    )

    fs_parallel = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,
    )

    fs_parallel2 = FeatureUnion(
        [
            ("words", CountVectorizer(analyzer="word")),
            ("chars", CountVectorizer(analyzer="char")),
        ],
        n_jobs=2,
    )

    fs.fit(X)
    X_transformed = fs.transform(X)
    assert X_transformed.shape[0] == len(X)

    fs_parallel.fit(X)
    X_transformed_parallel = fs_parallel.transform(X)
    assert X_transformed.shape == X_transformed_parallel.shape
    np.testing.assert_array_equal(
        X_transformed.toarray(), X_transformed_parallel.toarray()
    )

    # fit_transform should behave the same
    X_transformed_parallel2 = fs_parallel2.fit_transform(X)
    np.testing.assert_array_equal(
        X_transformed.toarray(), X_transformed_parallel2.toarray()
    )

    # transformers should stay fit after fit_transform
    X_transformed_parallel2 = fs_parallel2.transform(X)
    np.testing.assert_array_equal(
        X_transformed.toarray(), X_transformed_parallel2.toarray()
    )


def test_feature_union_feature_names():
    word_vect = CountVectorizer(analyzer="word")
    char_vect = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    ft = FeatureUnion([("chars", char_vect), ("words", word_vect)])
    ft.fit(JUNK_FOOD_DOCS)
    feature_names = ft.get_feature_names()
    for feat in feature_names:
        assert "chars__" in feat or "words__" in feat
    assert len(feature_names) == 35


def test_classes_property():
    iris = load_iris()
    X = iris.data
    y = iris.target

    reg = PipelineWithSampleFiltering(
        [("select", SelectKBest(k=1)), ("fit", LinearRegression())]
    )
    reg.fit(X, y)
    with pytest.raises(AttributeError):
        reg.classes_  # pylint:disable=pointless-statement

    clf = PipelineWithSampleFiltering(
        [
            ("select", SelectKBest(k=1)),
            (
                "fit",
                LogisticRegression(
                    solver="liblinear", multi_class="auto", random_state=0
                ),
            ),
        ]
    )
    with pytest.raises(AttributeError):
        reg.classes_  # pylint:disable=pointless-statement
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))


def test_with_step_that_tranform_target():
    iris = load_iris()
    X = iris.data
    y = iris.target
    filtr = PipelineWithSampleFiltering([("filtr", Filtr())])
    res = filtr.fit_transform(X, y)

    assert isinstance(res, tuple)
    assert len(res[0]) == len(X) - 1
    assert len(res[1]) == len(y) - 1
    assert filtr.fit_predict(X, y) == filtr.fit_predict(X)

    clf = PipelineWithSampleFiltering(
        [
            ("filtr", Filtr()),
            (
                "fit",
                LogisticRegression(
                    solver="liblinear", multi_class="auto", random_state=0
                ),
            ),
        ]
    )

    clf2 = clone(clf)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert len(y_pred) == len(y)

    clf2.fit(X, y, fit__sample_weight=np.arange(len(X)))
    y_pred2 = clf2.predict(X)
    assert not all([i == j for i, j in zip(y_pred, y_pred2)])


# pylint: enable=no-member
