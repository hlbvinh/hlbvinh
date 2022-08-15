import importlib
import pickle
import pprint
from datetime import timedelta
from random import randint

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, train_test_split

from skynet.prediction import climate_model, mode_model, mode_model_util
from skynet.sample import climate_sample_store
from skynet.utils import sklearn_utils
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.storage import get_storage
from skynet.utils.script_utils import get_connections

log = get_logger("skynet")
CACHE_FILE = "mode_model_data.pkl"
SOURCE_SIZE = 200
TRAIN_SIZE = 30
TEST_SIZE = 10
CHANGE_MAP = [
    ("temperature", (5, 10)),
    ("humidex", (7, 15)),
    ("target_temperature", (5, 10)),
    ("target_humidex", (7, 15)),
]
FAKE_NAME = "test"
RANGES = {"temperature": np.arange(-5, 6), "humidex": np.arange(-6, 7)}
VALIDATION_SPLIT_INTERVAL = "24H"
AVG_PARAMS = {
    "7": {"average_interval": "24H", "rolling_window": "84H"},
    "14": {"average_interval": "24H", "rolling_window": "168H"},
    "90": {"average_interval": "24H", "rolling_window": "336H"},
    "365": {"average_interval": "24H", "rolling_window": "336H"},
}

SCORE_ONE_MODES = tuple(sorted(["cool", "heat"]))


def pretty_print_score(scores):
    for mode_select, results in scores.items():
        print("===================================================")
        print("Mode Selections:", mode_select)

        do_confusion_matrix = False
        for scorer, result in results.items():

            if scorer == "confusion_matrix":
                do_confusion_matrix = True
            else:
                print("\n")
                print("Scorer:", scorer)
                for q, val in result.items():
                    print(q + ":", val)

        if do_confusion_matrix:
            print("\n")
            print("Confusion Matrix:")
            for quantity, cm in results["confusion_matrix"].items():
                print(quantity)
                print_cm(cm, labels=mode_select)
        print("\n")


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def score_avg(scores, score_variant):
    scores_dict = {}
    for modes, scores_modes in scores.items():
        for q in climate_model.QUANTITIES:
            score = scores_modes[score_variant][q]
            score_mean = float(score.split(" +/- ", 1)[0])
            score_std = float(score.split(" +/- ", 1)[1])
            score_str = str(modes) + "_" + q
            scores_dict.update(
                {score_str + "_mean": score_mean, score_str + "_std": score_std}
            )
    all_mean = [v for k, v in scores_dict.items() if "_mean" in k]
    mean_of_all = np.mean(all_mean)
    std_of_all = np.std(all_mean)
    scores_dict.update({"average_of_scores": mean_of_all, "std_of_scores": std_of_all})
    return scores_dict


def get_appliance_ids_greater_than(X, samples_size):
    appliance_ids_counts = X["appliance_id"].value_counts()
    is_train = (appliance_ids_counts > samples_size).values
    appliance_id_train = appliance_ids_counts.index.values[is_train]
    return appliance_id_train


def add_fake_values(X, change_map, fake_appliance_name):
    for _col, _range in change_map:
        offset = randint(_range[0], _range[1])
        X[_col] = X[_col].values + offset

    X["appliance_id"] = fake_appliance_name
    return X


def get_fake_dataset(
    X,
    y,
    source_samples_size,
    train_samples_size,
    test_samples_size,
    change_map,
    fake_name="test",
):
    """
    generate fake sample from samples by adding offset
    to features and value limited by change_map.

    fake sample are assigned with fake appliance_id
    with the format "fake_name + '_' + [real appliance_id]

    Parameters:
    ------------
    X: pandas DataFrame
        original features set

    y: pandas Series
        original target set

    source_samples_size: int
        minimum no of samples that applaince_id owns to be
        qualify to be a source for fake data

    train_samples_size: int
        number of fake samples to be generated from
        qualified appliance as part of training set

    test_samples_size: int
        number of fake samples to be generated from
        qualified appliance as part of test set

    change_map: list of tuples of string and tuple
        [(column_label, (low_limit, high_limit))]
        for each column label, a randomly selected offset
        between low_limit and high_limit will be added
        to each element of the the column under
        column label.

    fake_name: string
        a string to be added to exisiting appliance_id
        to generate fake appliance_id

    Returns:
    ------------
    X_train: pandas DataFrame
       train features set

    y_train: pandas Series
        train target set

    X_test: pandas DataFrame
        test features set

    y_test: pandas Series
        test target set
    """
    appliance_ids = get_appliance_ids_greater_than(X, source_samples_size)
    X_train = X.copy()
    y_train = y.copy()
    X_test = pd.DataFrame([])
    y_test = pd.Series([])
    for app in appliance_ids:
        sample_ids = X[X["appliance_id"] == app].index.values
        train, test = train_test_split(
            sample_ids, train_size=train_samples_size, test_size=test_samples_size
        )
        X_app_train = X.loc[train]
        y_app_train = y.loc[train]
        X_app_test = X.loc[test]
        y_app_test = y.loc[test]

        fake_appliance_name = fake_name + "_" + app
        X_app_train = add_fake_values(X_app_train, change_map, fake_appliance_name)

        X_app_test = add_fake_values(X_app_test, change_map, fake_appliance_name)

        X_train = X_train.append(X_app_train)
        y_train = y_train.append(y_app_train)
        X_test = X_test.append(X_app_test)
        y_test = y_test.append(y_app_test)

    return X_train, y_train, X_test, y_test


def get_cv_n_groups(
    X, n_train=None, validation_interval=None, use_time_series_cv=False
):

    if use_time_series_cv:
        return (
            sklearn_utils.TimeSeriesFold(
                n_train=n_train,
                validation_interval=timedelta(days=validation_interval),
                validation_split_interval=VALIDATION_SPLIT_INTERVAL,
            ),
            X.index.get_level_values(0),
        )
    return GroupKFold(n_splits=4), X["appliance_id"]


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--grid_module",
    default="skynet.prediction.estimators.mode_model",
    help="grid search pipeline and parameters to be used.",
)
@click.option("--mongo", default="production")
@click.option(
    "--task",
    default="train",
    type=click.Choice(
        ["grid_search", "score", "score_one", "score_fake_data", "train"]
    ),
)
@click.option("--n_jobs", default=1, type=int, help="Number of parallel jobs.")
@click.option("--cache", is_flag=True, default=False)
@click.option("--score", type=click.Choice(mode_model.SCORING_VARIANTS), default="all")
@click.option("--needs_confusion_matrix", default=False, is_flag=True)
@click.option("--use_time_series_cv", default=False, is_flag=True)
@click.option(
    "--validation_interval", default="14", type=click.Choice(AVG_PARAMS.keys())
)
@click.option("--n_train", default=80000, type=int)
@click.option("--log_directory", default="log")
@click.option("--storage", type=click.Choice(["s3", "file"]), default="s3")
def main(
    config,
    grid_module,
    mongo,
    task,
    n_jobs,
    cache,
    score,
    needs_confusion_matrix,
    use_time_series_cv,
    validation_interval,
    n_train,
    log_directory,
    storage,
):
    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config("mode_model", cnf=cnf, log_directory=log_directory)

    sample_store = climate_sample_store.ClimateSampleStore(
        get_connections(cnf, mongo=mongo).mongo
    )
    mod = importlib.import_module(grid_module)
    mode_selections = mode_model_util.get_all_possible_mode_selections()
    model = mode_model.ModeModel(mode_selections, n_jobs=n_jobs)

    def save(model):
        model_store = get_storage(
            storage, **cnf["model_store"], directory="data/models"
        )
        model_store.save(mode_model.ModeModel.get_storage_key(), model)

    def fetch(modes=mode_model.MODES):
        if cache:
            try:
                Xs, ys = pickle.load(open(CACHE_FILE, "rb"))
                print(f"loaded data from {CACHE_FILE}")
                return Xs, ys

            except Exception:
                pass

        key = mode_model.create_lookup_key()
        X, y = climate_sample_store.get_mode_model_samples(
            sample_store, modes=modes, key=key
        )
        # TODO: twice filtering
        Xs, ys = climate_model.make_static_mode_dataset(X, y)
        Xs, ys = mode_model.make_mode_model_dataset(Xs, ys)
        if cache:
            pickle.dump((Xs, ys), open(CACHE_FILE, "wb"), -1)
            print(f"saved data to {CACHE_FILE}")
        return Xs, ys

    if task in [
        "grid_search",
        "score",
        "score_fake_data",
        "score_one",
        "train",
        "confusion_matrix",
    ]:
        X, y = fetch()

        if task == "train":
            model.fit(X, y)
            weights = mod.get_mini_weights(model)
            # TODO: summarized version of weights
            log.debug(pprint.pformat(weights))
            save(model)

        elif task == "grid_search":
            #
            # Doing a manual grid search here so that we can use the custom
            # scoring function of the mode model.
            #
            results = []
            param_grid = mode_model_util.ModeModelParameterGrid(mod.get_params())
            cv, groups = get_cv_n_groups(
                X, n_train, int(validation_interval), use_time_series_cv
            )
            for params in param_grid:
                model.set_params(**params)
                scores = model.score(
                    X,
                    y,
                    score="accuracy",
                    n_jobs=n_jobs,
                    cv=cv,
                    groups=groups,
                    mode_selections=mode_selections,
                )
                scores = score_avg(scores, "accuracy")
                scores["params"] = params["params"]
                results.append(scores)
            results = pd.DataFrame(results)
            results = results.sort_values(by=["average_of_scores", "std_of_scores"])
            results.to_pickle("scores.pkl")
            print(results[["params", "average_of_scores", "std_of_scores"]])

        elif task == "score":
            cv, groups = get_cv_n_groups(
                X, n_train, int(validation_interval), use_time_series_cv
            )
            scores = model.score(
                X,
                y,
                score=score,
                n_jobs=n_jobs,
                cv=cv,
                groups=groups,
                needs_confusion_matrix=needs_confusion_matrix,
                mode_selections=mode_selections,
            )
            pretty_print_score(scores)

        elif task == "score_one":
            mask = np.in1d(y, SCORE_ONE_MODES)
            score_one = model._score(
                X[mask],
                y[mask],
                SCORE_ONE_MODES,
                sample_weights=mode_model.get_sample_weights(y[mask]),
            )
            pretty_print_score({SCORE_ONE_MODES: score_one})

        elif task == "score_fake_data":
            # Sooner or later, when we have more user,
            # or when weather changes for some user,

            # we may have contain users who targets or preferences
            # appeared to be significantly different.

            # This task help us see how well a model that trained
            # by samples that contain minority different samples to
            # make prediction on these samples.
            custom_scorer = make_scorer(mode_model.mode_accuracy_score)
            X_train, y_train, X_test, y_test = get_fake_dataset(
                X,
                y,
                source_samples_size=SOURCE_SIZE,
                train_samples_size=TRAIN_SIZE,
                test_samples_size=TEST_SIZE,
                change_map=CHANGE_MAP,
                fake_name=FAKE_NAME,
            )
            model.fit(X_train, y_train)
            y_test = np.array(y_test)
            X_test = model.get_features(X_test)
            columns = mode_model.TARGET_FEATURE_COLUMNS
            scores = {c: [] for c in columns}
            for column in columns:
                drop = [c for c in columns if c != column]
                X_t = X_test.drop(drop, axis=1)
                scores[column] = custom_scorer(model, X_t, y_test)
            print(scores)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
