import asyncio
import logging
from datetime import datetime, timedelta
from itertools import chain
from typing import Any, Dict, List, Optional

import pandas as pd

from . import aggregate, compose, query, selection
from ..utils import analyze, data, parse, preprocess, thermo
from ..utils.database import queries
from ..utils.database.dbconnection import Pool
from ..utils.enums import Power
from ..utils.types import ApplianceState
from ..utils.log_util import get_logger
from ..utils.async_util import multi

log = get_logger(__name__)

SAMPLE_CONCURRENCY = 10
SAMPLE_IVAL = "3H"
SAMPLE_IVAL_TD = timedelta(hours=3)
TARGET_INTERVAL = timedelta(minutes=5)

# Analysis: For continuous OFF state, 30 days duration covers 99% of such states
MAX_OFF_STATE_DURATION = 30  # days
# Analysis: For continuous ON state, 10h duration covers 99.9% of such states
MAX_ON_STATE_DURATION = 10

START = datetime(2014, 3, 1)


class MissingDataError(ValueError):
    pass


class Sample:
    def __init__(
        self,
        appliance_state_id,
        appliance_id,
        device_id,
        appliance_state,
        timestamp,
        end,
    ):
        """Generate Features and Target for Predictions.

        The get_all method constructs all samples from this appliance state.

        Parameters
        ----------
        appliance_state_id : int
            ID of appliance state in Database

        appliance_id : string
            ID of appliance in Database

        device_id : string
            ID of device associated with above appliance

        appliance_state : dict
            state of the appliance

        timestamp : datetime
            time when the appliance state was captured / sent

        end : datetime
            point time until which the targets should be generated

        """
        self.appliance_state_id = appliance_state_id
        self.appliance_id = appliance_id
        self.device_id = device_id
        self.appliance_state = appliance_state
        self.timestamp = timestamp
        self.end = end

    def log(self, msg, level=logging.INFO, **kwargs):
        extra = {"data": kwargs}
        extra["data"]["device_id"] = self.device_id
        log.log(
            level,
            "{}, {}, {}, {}".format(
                self.device_id, self.appliance_id, self.appliance_state_id, msg
            ),
            extra=extra,
        )

    async def fetch(self, pool, session):
        """Fetch all required data for this feature."""
        try:
            self.appliance_state = preprocess.prediction_signal(self.appliance_state)
        except KeyError as exc:
            raise ValueError("bad appliance state, missing {}".format(exc))

        try:
            result = await multi(
                {
                    "query_result": query.query_feature(
                        pool, session, self.device_id, self.timestamp, self.end
                    ),
                    "target_query_result": query.query_target(
                        session, self.device_id, self.timestamp, self.end
                    ),
                }
            )
            query_result = result["query_result"]
            target_query_result = result["target_query_result"]

            query.check_query(query_result)
            aggregated = aggregate.aggregate(query_result)
            self.feature_data = compose.compose(aggregated)

            target_aggregated = aggregate.aggregate(target_query_result)
            self.target_data = compose.compose(target_aggregated)
        except ValueError as exc:
            raise MissingDataError(exc) from exc
        return self

    def get_feature(self, timestamp):
        """Make a feature located at timestamp."""
        # discard data after timestamp
        feature_data = {k: v[v.index < timestamp] for k, v in self.feature_data.items()}

        feat = selection.check_select_features(
            selection.SELECTORS, feature_data, self.log
        )
        feat["appliance_id"] = self.appliance_id
        feat["appliance_state_id"] = self.appliance_state_id
        feat["device_id"] = self.device_id
        # feat['location_id'] = self.location_id
        feat["timestamp"] = timestamp

        feat = add_state(feat, self.appliance_state)
        return feat

    def get_target(self, timestamp):
        """Make targets originating at timestamp."""
        # discard data before timestamp + 1 target interval
        sensors = self.target_data["sensors"]
        sensors = sensors.loc[
            (sensors.index > timestamp) & (sensors.index <= timestamp + SAMPLE_IVAL_TD)
        ]
        target = selection.select_targets(
            sensors, target_start=timestamp + TARGET_INTERVAL
        )

        return target

    def get_all(self):
        """Make all features and targets from start to end.

        Returns
        -------
        features : list of DataFrame
            feature data for each sample

        targets : dict, datetime:pd.DataFrame
            target data, keys are points in time of the sample
        """
        if self.target_data["sensors"].empty:
            self.log("no target data")
            return []

        last_time = self.target_data["sensors"].index.max()
        # pylint: disable=no-member
        timestamps = pd.date_range(
            self.timestamp, last_time - TARGET_INTERVAL, freq=SAMPLE_IVAL, closed="left"
        ).to_pydatetime()
        # pylint: enable=no-member

        samples = []
        mode_hist = None
        previous_set_temperature = None
        for timestamp in timestamps:
            try:
                feature = self.get_feature(timestamp)
                target = self.get_target(timestamp)
            except ValueError as e:
                self.log("at {}, {}".format(timestamp, e), level=logging.ERROR)
                continue

            for key in ["temperature_set", "temperature_set_last"]:
                feature[key] = thermo.fix_temperature(feature[key])

            mode_hist = self.reevaluate_mode_hist(feature, mode_hist)
            previous_set_temperature = self.reevaluate_previous_set_temperature(
                feature, previous_set_temperature
            )
            weather = align_series(
                to_be_aligned=self.feature_data["weather"], align_on=target
            )
            target = analyze.filter_bad_targets(
                feature, target, weather, mode_hist, previous_set_temperature
            )
            if target.empty:
                self.log("target empty after filtering")
                break

            sample = feature.copy()
            sample["target"] = target.reset_index().to_dict("records")
            samples.append(sample)

        self.log("created {} samples".format(len(samples)))
        return samples

    @staticmethod
    def reevaluate_mode_hist(feature: Dict[str, Any], mode_hist: Optional[str]) -> str:
        if mode_hist:
            return feature["mode"]
        return feature["mode_hist"] if feature["power_hist"] == Power.ON else Power.OFF

    @staticmethod
    def reevaluate_previous_set_temperature(
        feature: Dict[str, Any], previous_set_temperature: Optional[float]
    ) -> float:
        if previous_set_temperature is None:
            return feature["temperature_set_last"]
        return feature["temperature_set"]


def align_series(
    to_be_aligned: pd.DataFrame = pd.DataFrame(),
    align_on: pd.DataFrame = pd.DataFrame(),
) -> pd.DataFrame:

    if to_be_aligned.empty:
        return pd.DataFrame()

    combined_index = sorted(to_be_aligned.index.union(align_on.index))

    aligned = to_be_aligned.reindex(combined_index)

    return aligned.interpolate(method="time").loc[align_on.index]


class PredictionSample:
    def __init__(self, device_id, timestamp):
        self.device_id = device_id
        self.timestamp = timestamp

    async def fetch(self, pool, session):

        data = await multi(
            {
                "query_result": query.query_feature(
                    pool, session, self.device_id, self.timestamp, end=self.timestamp
                ),
                "appliance_id": queries.get_appliance(pool, self.device_id),
            }
        )
        aggregated = aggregate.aggregate(data["query_result"])
        self.feature_data = compose.compose(aggregated, raise_on_missing=False)
        self.appliance_id = data["appliance_id"]
        return self

    def get_history_feature(self):
        feat = selection.select_features(
            selection.SELECTORS, self.feature_data, prediction=True
        )
        feat["appliance_id"] = self.appliance_id
        feat["device_id"] = self.device_id
        return feat

    def get_history_feature_or_none(self):
        feat = self.get_history_feature()
        valid_quantities = None not in (
            feat["humidex"],
            feat["humidity"],
            feat["temperature"],
        )

        return feat if valid_quantities else None


def add_state(history_features, current_state):
    """Combine history_features and AC state."""
    state = current_state.copy()
    combined = history_features.copy()
    combined.update(state)
    return combined


async def make_device_samples(
    pool, session, device_id, appliance_id, start, end, max_candidates=None
):
    appliance_states = await get_appliance_states_from_just_before_start_till_end(
        pool, appliance_id, device_id, start, end
    )
    candidates = create_sample_candidates(appliance_states, device_id, max_candidates)

    semaphore = asyncio.Semaphore(SAMPLE_CONCURRENCY)
    samples = await multi(
        fetch_samples_wrap(pool, session, sample, semaphore) for sample in candidates
    )
    flatten_samples = list(chain.from_iterable(samples))

    return flatten_samples


async def get_appliance_states_from_just_before_start_till_end(
    pool: Pool, appliance_id: str, device_id: str, start: datetime, end: datetime
) -> List[ApplianceState]:
    state_records = await pool.execute(
        *queries.query_appliance_states(appliance_id, start, end)
    )
    states = parse.lower_dicts(state_records)

    if not states:
        return []

    states = sorted(states, key=lambda x: (x["created_on"], x["appliance_state_id"]))

    log.debug("{} fetched {} unprocessed states".format(device_id, len(states)))

    return states


def create_sample_candidates(
    states: List[ApplianceState], device_id: str, max_candidates: Optional[int]
) -> List[Sample]:
    samples = []
    for s, s_next in zip(states[:-1], states[1:]):
        if are_states_invalid(s, s_next):
            continue
        samples.append(
            Sample(
                s["appliance_state_id"],
                s["appliance_id"],
                device_id,
                selection.get_appliance_state_features(s),
                s["created_on"],
                state_end_timestamp(s, s_next),
            )
        )
    return samples[:max_candidates]


def are_states_invalid(
    current_state: ApplianceState, next_state: ApplianceState
) -> bool:
    if next_state["created_on"] - current_state["created_on"] < timedelta(minutes=5):
        log.debug(
            "skipping {} next state too close"
            "".format(current_state["appliance_state_id"])
        )
        return True
    if not preprocess.has_prediction_fields(current_state):
        log.debug("skipping {} missing properties".format(current_state))
        return True
    return False


def state_end_timestamp(s: ApplianceState, s_next: ApplianceState) -> pd.Timestamp:
    duration = timedelta(hours=MAX_ON_STATE_DURATION)
    if s["power"] == Power.OFF:
        duration = timedelta(days=MAX_OFF_STATE_DURATION)
    return min([s_next["created_on"], s["created_on"] + duration])


async def fetch_samples_wrap(pool, session, sample, semaphore):
    try:
        async with semaphore:
            await sample.fetch(pool, session)
        samples = sample.get_all()
    except Exception as exc:
        log.error(exc, extra={"data": {"device_id": sample.device_id}})
        samples = []
    return samples


def upload_device_samples(sample_store, device_id, samples, watermark):
    ret = [sample_store.upsert(sample) for sample in samples]
    sample_store.set_watermark(device_id, watermark)
    return ret


async def generate_samples(
    pool, session, sample_store, device_id=None, max_candidates=None
):
    """Loop over devices and generate Sample data."""

    device_appliance_rows = await pool.execute(*queries.query_device_appliance_list())
    device_appliance_map = data.group_by("device_id", device_appliance_rows)

    if device_id is not None:
        device_appliance_map = {device_id: device_appliance_map[device_id]}

    watermarks = {d: sample_store.get_watermark(d) for d in device_appliance_map}

    for device_id_, start in watermarks.items():
        for appliance in device_appliance_map[device_id_]:
            appliance_id = appliance["appliance_id"]
            await make_upload_device_samples(
                pool,
                session,
                sample_store,
                device_id_,
                appliance_id,
                start,
                max_candidates=max_candidates,
            )


async def make_upload_device_samples(
    pool, session, sample_store, device_id, appliance_id, start, max_candidates=None
):

    new_watermark = datetime.utcnow()
    try:
        samples = await make_device_samples(
            pool,
            session,
            device_id,
            appliance_id,
            start=start,
            end=new_watermark,
            max_candidates=max_candidates,
        )
    except Exception as exc:
        log.error(exc, extra={"data": {"device_id": device_id}})
        return

    upload_device_samples(sample_store, device_id, samples, watermark=new_watermark)
