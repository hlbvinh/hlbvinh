from datetime import datetime

import pytest

from ..prediction import mode_model
from .. import sample


@pytest.fixture
def num_unique_modes():
    return 3


@pytest.fixture
def num_samples_per_mode():
    return 4


@pytest.fixture
def num_samples(num_unique_modes, num_samples_per_mode):
    return num_unique_modes * num_samples_per_mode


@pytest.fixture
def modes_labels(num_unique_modes):
    return [str(i) for i in range(num_unique_modes)]


@pytest.fixture
def modes_targets(modes_labels, num_samples_per_mode):
    return modes_labels * num_samples_per_mode


@pytest.fixture
def mode_hists(num_unique_modes, modes_targets):
    mode_copy = modes_targets.copy()
    for i in range(num_unique_modes):
        mode_copy[i] = "other"
    return mode_copy


@pytest.fixture
def sample_ids(num_samples):
    return [str(i) for i in range(num_samples)]


def upsert_samples(store, ids, targets, hists):
    for _id, mode, hist in zip(ids, targets, hists):
        record = {k: _id for k in store.sample_id}
        record["mode"] = mode
        record["mode_hist"] = hist
        record["origin"] = "irdeployment"
        record["target"] = [{"a": 1}]
        record["timestamp"] = datetime.utcnow()
        store.upsert(record)


def assert_fetched_sample(store, key, sample_size, labels):
    X, y = sample.climate_sample_store.get_mode_model_samples(
        store, modes=labels, key=key
    )
    assert len(X) == sample_size
    assert len(y) == sample_size
    assert sorted(list(X.index)) == sorted(list(y.index))
    for i in labels:
        assert i in list(X["mode"])


def test_fetch_mode_samples(
    climate_sample_store,
    modes_labels,
    modes_targets,
    mode_hists,
    sample_ids,
    num_samples,
):

    upsert_samples(climate_sample_store, sample_ids, modes_targets, mode_hists)

    runs = [
        {"key": None, "sample_size": num_samples, "labels": modes_labels},
        {
            "key": mode_model.create_lookup_key(),
            "sample_size": num_samples,
            "labels": modes_labels,
        },
    ]

    for run in runs:
        assert_fetched_sample(climate_sample_store, **run)
