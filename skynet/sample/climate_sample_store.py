from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from pymongo.cursor import CursorType

from skynet.prediction.climate_model import DROP_COLUMNS, FEATURE_COLUMNS
from skynet.sample.selection import COMPENSATE_COLUMN

from ..prediction.mode_model_util import MULTIMODES
from ..prediction.climate_model import downcast_float
from ..sample.selection import RECOMPUTATIONS
from ..utils import thermo
from ..utils.compensation import compensate_features
from ..utils.log_util import get_logger
from ..utils.misc import timeit
from ..utils.sample_store import SampleStore

log = get_logger(__name__)

SAMPLE_COLLECTION = "climate_model_samples"
WATERMARK_COLLECTION = "climate_model_watermarks"
SAMPLE_ID = ["timestamp", "appliance_id", "device_id"]
WATERMARK_KEY = "device_id"
WATERMARK_VALUE = "timestamp"
INDEX_ORDER = [-1, 1, 1]

RELATIVE_DELTA = relativedelta(
    datetime.date(datetime.utcnow()), datetime.date(parse("2014-01-01"))
)

MONTH_LIMIT = RELATIVE_DELTA.months + RELATIVE_DELTA.years * 12  # type: ignore
MODES = MULTIMODES + ["off"]
EXTRA_INDICES = [
    (["origin"], [1]),
    (["mode", "timestamp"], [1, -1]),
    (["mode_hist", "mode", "origin", "timestamp"], [1, 1, 1, -1]),
]
TARGET = ["target"]


def create_projections_for_climate_samples() -> Dict[str, int]:
    list_of_keys_to_be_fetched = (
        FEATURE_COLUMNS + DROP_COLUMNS + TARGET + [COMPENSATE_COLUMN]
    )
    for values in RECOMPUTATIONS.values():
        for single_column_name in values:
            if single_column_name not in list_of_keys_to_be_fetched:
                list_of_keys_to_be_fetched.append(single_column_name)
    keys_to_be_fetched = {k: 1 for k in list_of_keys_to_be_fetched}
    keys_to_be_fetched["_id"] = 0
    return keys_to_be_fetched


def _recompute_features(df: pd.DataFrame) -> None:
    for name, columns in RECOMPUTATIONS.items():
        has_columns = all(c in df for c in columns)
        if has_columns:
            log.info(f"recomputing {name} from {', '.join(columns)}")
            df[name] = thermo.humidex(
                **{arg: df[col] for arg, col in columns._asdict().items()}
            )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compensate old (raw) samples and recompute necessary quantities."""
    log.info(f"preparing features with shape {df.shape}")
    df = compensate_features(df)
    _recompute_features(df)
    return df


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"preparing targets with shape {df.shape}")
    df = compensate_features(df)
    if "temperature" in df and "humidity" in df:
        log.info("recomputing humidex target")
        df["humidex"] = thermo.humidex(df["temperature"], df["humidity"])
    return df


def get_climate_samples(
    sample_store,
    limit=0,
    key={"timestamp": {"$gt": datetime.utcnow() - relativedelta(months=MONTH_LIMIT)}},
):
    features, targets = sample_store.get_reverse_sorted_climate_samples(
        limit=limit, key=key
    )
    return prepare_features(features), prepare_targets(targets)


def get_mode_model_samples(sample_store, modes, key, limit=0):
    features, targets = sample_store.get_mode_model_samples(
        modes=modes, key=key, limit=limit
    )
    return prepare_features(features), prepare_targets(targets)


class ClimateSampleStore(SampleStore):
    def __init__(
        self,
        client,
        sample_collection=SAMPLE_COLLECTION,
        sample_id=SAMPLE_ID,
        index_order=INDEX_ORDER,
        watermark_collection=WATERMARK_COLLECTION,
        watermark_key=WATERMARK_KEY,
        watermark_value=WATERMARK_VALUE,
        extra_indices=EXTRA_INDICES,
    ):
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

    @timeit()
    def _aggregate_samples(
        self, key, limit, sort, batch_size, sample_id=0, projection=None
    ):
        # mongo EXHAUST cursor reduces latency by not requiring to request each batch
        cursor = self.client._col(self.sample_collection).find(
            key,
            limit=limit,
            sort=sort,
            projection=projection,
            batch_size=1000,
            cursor_type=CursorType.NON_TAILABLE if limit else CursorType.EXHAUST,
        )
        features_df = []
        targets_df = []

        while True:
            batch_targets = []
            batch_features = []
            try:
                for _ in range(batch_size):
                    # We can store the data from mongo inside a single dataframe but not
                    # as a list of dictionaries, so we need to batch over the list of
                    # dictionaries and create intermediate dataframes to make everything
                    # fit in memory.
                    record = next(cursor)
                    target = record.pop("target")
                    record["sample_id"] = sample_id
                    for t in target:
                        t["sample_id"] = sample_id
                    sample_id += 1
                    record["previous_temperatures"] = np.array(
                        record.get("previous_temperatures", []), dtype=np.float32
                    )
                    batch_features.append(record)  # record is a dict
                    batch_targets.extend(target)  # targets is a list
            except StopIteration:
                break
            finally:
                features_df.append(downcast_float(pd.DataFrame(batch_features)))
                targets_df.append(downcast_float(pd.DataFrame(batch_targets)))
        features = pd.concat(features_df)
        targets = pd.concat(targets_df)
        return features, targets

    @timeit()
    def get_reverse_sorted_climate_samples(self, key=None, limit=0, batch_size=10000):
        if key is None:
            key = {}
        sort = [("timestamp", -1)]
        features, targets = self._aggregate_samples(
            key,
            limit,
            sort,
            batch_size,
            projection=create_projections_for_climate_samples(),
        )
        return features.set_index("sample_id"), targets.set_index("sample_id")

    @timeit()
    def get_mode_model_samples(self, key=None, modes=None, limit=0, batch_size=10000):
        if key is None:
            key = {}
        sort = [("timestamp", -1)] if limit else None
        features = []
        targets = []
        sample_id = 0

        # using the following approach to construct the mongo query rather than
        # the $or logical operator because this is much faster
        for condition, timestamp in [
            ("$ne", key.get("timestamp", {})),
            ("$eq", {"$gte": datetime.utcnow() - timedelta(days=120)}),
        ]:
            if timestamp != {}:
                key["timestamp"] = timestamp
            for mode in modes:
                mode_key = {
                    "$and": [
                        key,
                        {"mode": {"$eq": mode}},
                        {"mode_hist": {condition: mode}},
                    ]
                }
                feat, tar = self._aggregate_samples(
                    mode_key, limit, sort, batch_size, sample_id
                )
                features.append(feat)
                targets.append(tar)
                sample_id += len(feat)

        features = pd.concat(features)
        targets = pd.concat(targets)
        features = features.set_index("sample_id")
        targets = targets.set_index("sample_id")
        features, targets = balance_mode_proportion(features, targets)
        log.info(
            "sample_size={}, modes_change_sample_size={}, "
            "non_modes_change_sample_size={}, sample_size_per_mode={}, "
            "origin={}, distinct_appliance_ids={}".format(
                len(features),
                len(features[features["mode"] != features["mode_hist"]]),
                len(features[features["mode"] == features["mode_hist"]]),
                features["mode"].value_counts().to_dict(),
                features["origin"].value_counts().to_dict(),
                len(np.unique(features["appliance_id"])),
            )
        )
        return features, targets


def balance_mode_proportion(
    features: pd.DataFrame, targets: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # in addition to balancing mode samples proportion using sample weights, also balancing
    # the proportion manually because sample weights are not working as well as expected.
    # specifically, ensure that the two modes with the most number of samples have the same size
    mode: str
    (mode, count_a), (_, count_b) = Counter(features["mode"]).most_common(2)
    excess_count = count_a - count_b
    discard_indices = features[features["mode"] == mode].sample(excess_count).index
    return features.drop(discard_indices), targets.drop(discard_indices)
