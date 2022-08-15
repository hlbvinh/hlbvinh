from datetime import datetime, timedelta
from numbers import Number

import numpy as np
import pandas as pd
import pytest

from ..sample import climate_sample_store as store
from ..sample import sample, selection
from ..utils.database import queries

HISTORY_FEATURE_TYPES = (Number, str, type(None), bool, list)
SAMPLE_VALUE_TYPES = HISTORY_FEATURE_TYPES + (datetime,)


def _check_target_timestamps(sample):
    # change if modifying skynet.sample.sample.TARGET_IVAL
    ival = timedelta(minutes=5)
    timestamp = sample["timestamp"]
    target = sample["target"]
    assert ival == timedelta(minutes=5)
    assert timestamp + ival == target[0]["timestamp"]
    for t0, t1 in zip(target[:-1], target[1:]):
        assert t0["timestamp"] + ival == t1["timestamp"]


@pytest.fixture
def device_ids(db):
    devs = queries.execute(db, "SELECT device_id FROM Device")
    return [x["device_id"] for x in devs]


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_generate_all_samples_cassandra(
    pool, cassandra_session, climate_sample_store
):
    await sample.generate_samples(
        pool, cassandra_session, climate_sample_store, max_candidates=160
    )
    X, y = store.get_climate_samples(climate_sample_store)
    assert isinstance(X, pd.DataFrame)
    assert not X.empty
    assert isinstance(y, pd.DataFrame)
    assert not y.empty


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_generate_samples_by_device(
    pool, cassandra_session, climate_sample_store, device_appliance_rows
):
    device_ids = [d["device_id"] for d in device_appliance_rows]
    for device_id in device_ids:
        await sample.generate_samples(
            pool, cassandra_session, climate_sample_store, device_id, max_candidates=160
        )
        X, y = store.get_climate_samples(climate_sample_store)
        assert isinstance(X, pd.DataFrame)
        assert not X.empty
        assert isinstance(y, pd.DataFrame)
        assert not y.empty


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_upload_fetch(
    pool, cassandra_session, climate_sample_store, device_appliance_rows
):
    for row in device_appliance_rows:
        device_id = row["device_id"]
        appliance_id = row["appliance_id"]
        end = datetime.utcnow()

        samples = await sample.make_device_samples(
            pool,
            cassandra_session,
            device_id=device_id,
            appliance_id=appliance_id,
            start=None,
            end=end,
            max_candidates=160,
        )

        for s in samples:
            _check_target_timestamps(s)
            for key, value in s.items():
                if key != "target":
                    assert isinstance(value, SAMPLE_VALUE_TYPES)

        sample.upload_device_samples(
            climate_sample_store, device_id, samples, watermark=end
        )
        if not samples:
            pytest.fail("no samples created")

        for s in samples:
            fetched = climate_sample_store.get(
                {"device_id": device_id, "timestamp": s["timestamp"]}
            )[0]
            assert s == fetched


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
@pytest.mark.asyncio
async def test_prediction_sample(db, pool, cassandra_session, device_ids):
    states = [
        queries.get_appliance_states_from_device(
            db, device_id, start=datetime.fromtimestamp(0), end=datetime.utcnow()
        )
        for device_id in device_ids
    ]

    for device_id, state in zip(device_ids, states):
        for s in state:
            s["device_id"] = device_id

    states = [s for device_states in states for s in device_states][:10]

    for s in states:
        samp = sample.PredictionSample(s["device_id"], s["created_on"])
        await samp.fetch(pool, cassandra_session)
        history_feature = samp.get_history_feature()
        for k in sample.selection.SELECTORS:
            assert k in history_feature
            assert isinstance(history_feature[k], HISTORY_FEATURE_TYPES)

    # test for device that doesn't exist in db
    samp = sample.PredictionSample("fake_device", datetime.utcnow())
    await samp.fetch(pool, cassandra_session)
    history_feature = samp.get_history_feature()
    for name, selector in sample.selection.SELECTORS.items():
        assert is_equal(history_feature[name], selector.default_value)


@pytest.mark.asyncio
async def test_bad_samples(pool, cassandra_session):
    with pytest.raises(ValueError):
        samp = sample.Sample(
            0, "fake_appliance", "fake_device", {}, datetime.utcnow(), datetime.utcnow()
        )
        await samp.fetch(pool, cassandra_session)


@pytest.mark.parametrize(
    "y, result",
    [
        (0, 0),
        (selection.TARGET_INTERVAL_MINUTES, selection.STATIC_INTERPOLATION_MINUTES),
    ],
)
def test_linear_extrapolation(y, result):
    """Extrapolate the y = x slope."""
    ys = np.array([0, y])
    extrapolated = selection._compute_static_target(ys)
    assert np.allclose(extrapolated, result)


def is_equal(x, y):
    return x == y or (np.isnan(x) and np.isnan(y))
