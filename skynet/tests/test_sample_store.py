from datetime import datetime
import pytest


def get_sample(store, **kwargs):
    sample = {
        k: datetime(2015, 1, 1) if k == "timestamp" else "a" for k in store.sample_id
    }
    sample.update(kwargs)
    return sample


def clear(store):
    store.clear()
    store.reset_watermarks()


def test_watermarks(sample_store):
    clear(sample_store)

    # test empty
    assert sample_store.get_watermark("a") is None

    sample_store.set_watermark("a", 1)
    sample_store.set_watermark("b", 2)
    assert sample_store.get_watermark("a") == 1
    assert sample_store.get_watermark("b") == 2

    assert len(sample_store.get_watermarks()) == 2

    # test update
    sample_store.set_watermark("a", 3)
    assert sample_store.get_watermark("a") == 3
    sample_store.set_watermark("a", 0)
    assert sample_store.get_watermark("a") == 0
    clear(sample_store)
    assert not sample_store.get_watermarks()


def test_sample_insert(sample_store):
    clear(sample_store)
    s = get_sample(sample_store)
    sample_store.upsert(s)
    s2 = sample_store.get(s)[0]
    assert s == s2
    clear(sample_store)


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
def test_sample_upsert(sample_store):
    clear(sample_store)
    for s in [get_sample(sample_store) for _ in range(10)]:
        sample_store.upsert(s)
    s = get_sample(sample_store)
    samples = sample_store.get()
    assert len(samples) == 1
    assert samples[0] == s
    clear(sample_store)


def test_sample_upsert_many(sample_store):
    clear(sample_store)
    samples = [
        get_sample(sample_store, timestamp=datetime(2015, 1, i)) for i in range(1, 10)
    ]
    sample_store.upsert_many(samples)
    fetched_samples = sample_store.get()
    assert samples == sorted(fetched_samples, key=lambda x: x["timestamp"])
    clear(sample_store)


@pytest.mark.flaky(reruns=2, reruns_delay=0.1)
def test_multiple_keys(sample_store):

    clear(sample_store)
    ids = ["x", "y"]
    timestamps = [datetime(2015, 1, 1), datetime(2015, 1, 2)]

    for key, values in zip(sample_store.sample_id, [ids, timestamps]):
        samples = [get_sample(sample_store, **{key: v}) for v in values]

        for s in samples:
            sample_store.upsert(s)

        retrieved = [sample_store.get(s) for s in samples]
        for s, r in zip(samples, retrieved):
            assert len(r) == 1
            assert s == r[0]

    assert len(sample_store.get()) == 4
    clear(sample_store)


def test_bad_key(sample_store):
    clear(sample_store)
    bad_samples = [{k: "test"} for k in sample_store.sample_id]
    for s in bad_samples:
        with pytest.raises(ValueError):
            sample_store.upsert(s)
    clear(sample_store)
