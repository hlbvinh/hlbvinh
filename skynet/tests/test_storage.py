from asyncio import sleep

import pytest

from ..utils import storage


def test_loader(model_store):
    model_keys = {"model_a": {"model_type": "test"}}
    loader = storage.Loader(model_store, model_keys, reload_seconds=1.0)
    with pytest.raises(KeyError):
        loader._load()

    model_store.save(key=model_keys["model_a"], obj="stuff1")
    assert loader._load()["model_a"] == "stuff1"


@pytest.fixture
def reload_seconds():
    return 0.05


@pytest.mark.skip("too flakly now that we use gc.collect()")
@pytest.mark.flaky(reruns=3, reruns_delay=1)
@pytest.mark.asyncio
async def test_model_reload_actor(model_store, reload_seconds):
    model_keys = {"model_a": {"model_type": "test"}}
    model_store.save(key=model_keys["model_a"], obj="stuff2")
    loader = storage.Loader(model_store, model_keys, reload_seconds=reload_seconds)
    mra = storage.ModelReloadActor(loader)
    # we use ensure_future in the reload, need to make sure we switch context on the event loop
    await sleep(0)
    assert mra.models["model_a"] == "stuff2"

    # insert a new model
    model_store.save(key=model_keys["model_a"], obj="stuff3")

    # we should still have the old model
    await sleep(0)
    assert mra.models["model_a"] == "stuff2"

    # the new model should have been loaded
    await sleep(reload_seconds * 1.5)
    assert mra.models["model_a"] == "stuff3"


def test_mongo_storage(model_store):
    # try saving with key that contains data field
    with pytest.raises(ValueError):
        model_store.save({"data": "hello"}, 1)

    # try saving unpicklable object
    with pytest.raises(ValueError):
        model_store.save({"key": "value"}, lambda: None)

    # save something normal
    model_store.save({"key": "value"}, 1)
    assert model_store.load({"key": "value"}) == 1
    model_store.remove({"key": "value"})
    with pytest.raises(KeyError):
        model_store.load({"key": "value"})
