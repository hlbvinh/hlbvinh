from typing import List

import pandas as pd

from ..utils.log_util import get_logger
from ..utils.mongo import Client
from ..utils.sample_store import SampleStore

log = get_logger(__name__)

SAMPLE_COLLECTION = "user_model_samples"
WATERMARK_COLLECTION = "user_model_watermarks"
SAMPLE_ID = ["device_id", "timestamp"]
WATERMARK_KEY = "device_id"
WATERMARK_VALUE = "timestamp"
INDEX_ORDER = [1, -1]
EXTRA_INDICES = [(["timestamp"], [1])]


class UserSampleStore(SampleStore):
    def __init__(
        self,
        client: Client,
        sample_collection: str = SAMPLE_COLLECTION,
        sample_id: List[str] = SAMPLE_ID,
        index_order: List[int] = INDEX_ORDER,
        watermark_collection: str = WATERMARK_COLLECTION,
        watermark_key: str = WATERMARK_KEY,
        watermark_value: str = WATERMARK_VALUE,
        extra_indices=EXTRA_INDICES,
    ) -> None:
        super().__init__(
            client,
            sample_collection=sample_collection,
            sample_id=sample_id,
            index_order=index_order,
            watermark_collection=watermark_collection,
            watermark_key=watermark_key,
            watermark_value=watermark_value,
            extra_indices=extra_indices,
        )

    def get_samples_sorted_by_timestamp_desc(
        self, key: dict = {}, sample_limit: int = 0
    ) -> pd.DataFrame:
        cursor = self.client._col(self.sample_collection).find(
            key, limit=sample_limit, sort=[("timestamp", -1)]
        )
        return pd.DataFrame(cursor)
