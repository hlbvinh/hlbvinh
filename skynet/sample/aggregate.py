from typing import Any, Dict, List

import pandas as pd

from skynet.user.sample import OPTIONAL_FEATURE
from ..utils import parse, preprocess
from ..control.util import AMBI_SENSOR_INTERVAL_SECONDS


def weather_agg(weather_records: List[dict]) -> pd.DataFrame:
    if weather_records:
        weather_records = parse.lower_dicts(weather_records)
        return pd.DataFrame.from_records(weather_records, index="timestamp")
    return pd.DataFrame()


def appliance_state_agg(state_records: List[dict]) -> pd.DataFrame:
    if state_records:
        state_records = parse.lower_dicts(state_records)
        return preprocess.appliance_states_to_df(state_records).ffill().bfill()
    return pd.DataFrame()


def sensor_agg(sensor_records: List) -> pd.DataFrame:
    rows = list(sensor_records)
    if rows:
        df = (
            pd.DataFrame(rows)
            .set_index("created_on")
            .resample(str(AMBI_SENSOR_INTERVAL_SECONDS) + "s")
            .mean()
            .interpolate()
        )

        filled = df.ffill().bfill()
        if is_invalid_data(filled):
            raise ValueError(
                "NaNs present for required features after sensor aggregation"
            )
        return filled
    return pd.DataFrame()


def is_invalid_data(df):
    required_features = [x for x in df.columns if x not in [OPTIONAL_FEATURE]]
    return df[required_features].isna().values.any()


def aggregate(query_data: Dict[str, List[Any]]) -> Dict[str, pd.DataFrame]:

    r = {}
    r["sensors"] = sensor_agg(query_data.get("sensors", []))
    r["weather"] = weather_agg(query_data.get("weather", []))
    r["states"] = appliance_state_agg(query_data.get("states", []))

    return r
