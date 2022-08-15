from datetime import datetime

import numpy as np
import pandas as pd

from ..user import store


def test_get_samples(user_sample_store):
    record = {k: "a" for k in store.SAMPLE_ID}
    record["data_field"] = 1
    user_sample_store.upsert(record)
    fetched = user_sample_store.get_samples_sorted_by_timestamp_desc().to_dict(
        "records"
    )[0]
    fetched.pop("_id")
    assert record == fetched

    user_sample_store.clear_all()
    records = [
        {"device_id": "a", "timestamp": datetime(2015, 1, i)} for i in range(1, 11)
    ]
    df = pd.DataFrame(records)
    timestamps = df["timestamp"]
    np.random.shuffle(records)
    for r in records:
        user_sample_store.upsert(r)

    fetched = user_sample_store.get_samples_sorted_by_timestamp_desc(sample_limit=10)
    assert list(fetched["timestamp"]) == sorted(timestamps, reverse=True)

    fetched2 = user_sample_store.get_samples_sorted_by_timestamp_desc(sample_limit=5)
    assert list(fetched2["timestamp"]) == sorted(timestamps, reverse=True)[:5]

    fetched3 = user_sample_store.get_samples_sorted_by_timestamp_desc(sample_limit=0)
    assert list(fetched3["timestamp"]) == sorted(timestamps, reverse=True)

    user_sample_store.clear_all()
