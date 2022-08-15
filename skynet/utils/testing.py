import random
import string
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, DateOffset

from ..utils.config import MAX_COMFORT, MIN_COMFORT
from .compensation import COMPENSATE_COLUMN

DEFAULT_FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "apparent_temperature": {"max_value": 34.0, "min_value": 28.0, "type": "uniform"},
    "appliance_id": {"n_values": 15, "string_length": 36, "type": "string"},
    "cloud_cover": {"min_value": 80.0, "type": "uniform"},
    "created_on": {"type": "timestamp"},
    COMPENSATE_COLUMN: {"type": "bool"},
    "device_id": {"n_values": 15, "string_length": 8, "type": "string"},
    "fan_hist": {"n_values": 3, "string_length": 5, "type": "string"},
    "feedback": {"max_value": MAX_COMFORT, "min_value": MIN_COMFORT, "type": "uniform"},
    "feelslike": {"mean": 28.0, "sigma": 3.0, "type": "normal"},
    "feelslike_delta": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "humidex": {"mean": 36.0, "sigma": 4.0, "type": "normal"},
    "humidex_delta": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "humidex_mean": {"mean": 37.0, "sigma": 4.0, "type": "normal"},
    "humidex_out": {"mean": 40.0, "sigma": 5.0, "type": "normal"},
    "humidity": {"mean": 55.0, "sigma": 5.0, "type": "normal"},
    "humidity_refined": {"mean": 55.0, "sigma": 5.0, "type": "normal"},
    "humidity_raw": {"mean": 55.0, "sigma": 5.0, "type": "normal"},
    "humidity_delta": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "humidity_mean": {"mean": 55.0, "sigma": 5.0, "type": "normal"},
    "humidity_out": {"mean": 60.0, "sigma": 5.0, "type": "normal"},
    "humidity_out_max_day": {"max_value": 75.0, "min_value": 70.0, "type": "uniform"},
    "humidity_out_mean_day": {"mean": 59.0, "sigma": 7.0, "type": "normal"},
    "humidity_out_mean_week": {"mean": 57.0, "sigma": 8.0, "type": "normal"},
    "humidity_out_min_day": {"max_value": 55.0, "min_value": 50.0, "type": "uniform"},
    "luminosity": {"max_value": 500, "min_value": 0, "type": "int"},
    "luminosity_mean": {"max_value": 500, "min_value": 300, "type": "int"},
    "luminosity_slope": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "mode": {"n_values": 2, "string_length": 4, "type": "string"},
    "mode_hist": {"n_values": 2, "string_length": 4, "type": "string"},
    "pircount": {"max_value": 5.0, "min_value": 0.0, "type": "uniform"},
    "pircount_mean": {"max_value": 5.0, "min_value": 0.0, "type": "uniform"},
    "pirload": {"max_value": 5.0, "min_value": 0.0, "type": "uniform"},
    "power_hist": {"n_values": 2, "string_length": 2, "type": "string"},
    "quantity": {"n_values": 3, "string_length": 2, "type": "string"},
    "origin": {"n_values": 4, "string_length": 5, "type": "string"},
    "target": {"mean": 0.0, "sigma": 8.0, "type": "normal"},
    "temperature": {"mean": 25.0, "sigma": 3.0, "type": "normal"},
    "temperature_refined": {"mean": 25.0, "sigma": 3.0, "type": "normal"},
    "temperature_raw": {"mean": 25.0, "sigma": 3.0, "type": "normal"},
    "temperature_delta": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "temperature_mean": {"mean": 25.0, "sigma": 1.0, "type": "normal"},
    "temperature_out": {"mean": 31.0, "sigma": 5.0, "type": "normal"},
    "temperature_out_delta": {"mean": 0.0, "sigma": 0.1, "type": "normal"},
    "temperature_out_max_day": {"mean": 34.0, "sigma": 2.0, "type": "normal"},
    "temperature_out_mean_day": {"mean": 32.0, "sigma": 3.0, "type": "normal"},
    "temperature_out_mean_week": {"mean": 30.0, "sigma": 6.0, "type": "normal"},
    "temperature_out_min_day": {"mean": 26.0, "sigma": 2.0, "type": "normal"},
    "temperature_set": {"max_value": 25, "min_value": 16, "type": "int"},
    "temperature_set_last": {"max_value": 25, "min_value": 16, "type": "int"},
    "timestamp": {"type": "timestamp"},
    "tod_cos": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "tod_sin": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "tow_cos": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "tow_sin": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "toy_cos": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "toy_sin": {"max_value": 1.0, "min_value": -1.0, "type": "uniform"},
    "user_id": {"n_values": 15, "string_length": 36, "type": "string"},
}


def gen_random_feature_normal(
    name: str, n_samples: int, mean: float = 0.0, sigma: float = 1.0
) -> DataFrame:
    """Generate feature from normal distribution.

    Parameters
    ----------
    name: str
        Name of the DataFrame column.

    n_samples: int
        Size of the DataFrame

    mean: float
        Mean of normal distribution. (default = 0.0 if not specified)

    sigma: float
        Standard deviation of normal distribution.
        (default = 1.0 if not specified)

    Returns
    -------
    DataFrame
        Randomly generated feature from normal distribution.
    """
    return pd.DataFrame(
        data=np.random.normal(mean, sigma, size=n_samples), columns=[name]
    )


def gen_feature_uniform(
    name: str, n_samples: int, min_value: float = 0.0, max_value: float = 0.0
) -> DataFrame:
    """Generate feature from uniform distribution.

    Parameters
    ----------
    name: str
        Name of the DataFrame column.

    n_samples: int
        Size of the DataFrame

    min_value: float
        Minimum value of uniform distribution. (default = 0.0 if not specified)

    max_value: float
        Maximum value of uniform distribution. (default = 0.0 if not specified)

    Returns
    -------
    DataFrame
        Randomly generated feature from uniform distribution.
    """
    return pd.DataFrame(
        data=np.random.uniform(min_value, max_value, size=n_samples), columns=[name]
    )


def gen_feature_string(
    name: str, n_samples: int, n_values: int, string_length: int
) -> DataFrame:
    """Generate feature from specified number of strings.

    Parameters
    ----------
    name: str
        Name of the DataFrame column.

    n_samples: int
        Size of the DataFrame

    n_values: int
        Number of distinct strings

    string_length: int
        Length of each string

    Returns
    -------
    DataFrame
        Randomly generated feature from specified number of strings.
    """
    random_strings = []
    for _ in range(n_values):
        random_strings.append(
            "".join(random.choice(string.ascii_lowercase) for _ in range(string_length))
        )
    random_strings = np.random.choice(random_strings, n_samples)
    return pd.DataFrame(data=random_strings, columns=[name])


def gen_feature_int(
    name: str, n_samples: int, min_value: int = -100, max_value: int = 100
) -> DataFrame:
    """Generate feature from integers.

    Parameters
    ----------
    name: str
        Name of the DataFrame column.

    n_samples: int
        Size of the DataFrame

    min_value: int
        Lower bound of the range of random integers, default = -100

    max_value: int
        Upper bound of the range of random integers, default = 100

    Returns
    -------
    DataFrame
        Randomly generated feature from integers.
    """

    return pd.DataFrame(
        data=np.random.randint(min_value, max_value, size=n_samples), columns=[name]
    )


def gen_feature_bool(name: str, n_samples: int) -> DataFrame:
    return pd.DataFrame(
        data=np.random.randint(2, size=n_samples).astype(bool), columns=[name]
    )


def gen_feature_datetime(
    name: str,
    n_samples: int,
    start: Union[str, datetime] = datetime(2014, 1, 1),
    freq: Union[str, DateOffset] = "D",
) -> DataFrame:
    """Generate dataframe with timestamp according to specification

    Parameters
    ----------
    name: str
        Name of the DataFrame column.

    n_samples: int
        Size of the DataFrame , period of timestamp

    start : string or datetime-like, default = datetime(2014, 1, 1)
        Left bound for generating dates

    freq : string or DateOffset, default 'D' (calendar daily)
        Frequency strings can have multiples, e.g. '5H'

    Returns
    -------
    DataFrame
        A feature matrix with randomly generated values according to type of
        features.
    """

    return pd.DataFrame(
        pd.date_range(start=start, periods=n_samples, freq=freq), columns=[name]
    )


# define a dictionary with a map from 'type' and call the generating function
FEATURE_MAP: Dict[str, Callable] = {
    "bool": gen_feature_bool,
    "int": gen_feature_int,
    "normal": gen_random_feature_normal,
    "string": gen_feature_string,
    "timestamp": gen_feature_datetime,
    "uniform": gen_feature_uniform,
}


def gen_matrix(feature_config: List[Dict[str, Any]], n_samples: int) -> DataFrame:
    """Create a feature matrix with n_samples rows based on feature config.

    Parameters
    ----------
    feature_config: list of dictionaries
        a list of dictionaries containing different features.

    n_samples: int
        Size of the DataFrame

    Returns
    -------
    DataFrame
        A feature matrix with randomly generated values according to
        feature_config.
    """
    dataframes = []
    for config in feature_config:
        temp_dict = config.copy()
        name = temp_dict.pop("name")
        feature_type = temp_dict.pop("type")
        df = FEATURE_MAP[feature_type](name=name, n_samples=n_samples, **temp_dict)
        dataframes.append(df)
    return pd.concat(dataframes, axis=1)


def gen_feature_matrix(
    features: List[str],
    n_samples: int,
    defined_feature_configs: Optional[List[Dict[str, Union[float, int, str]]]] = None,
) -> DataFrame:
    defined_feature_configs = defined_feature_configs or []

    feature_configs = defined_feature_configs.copy()
    defined_feat = [config["name"] for config in defined_feature_configs]
    for feat_name, config in deepcopy(DEFAULT_FEATURE_CONFIG).items():
        if feat_name in features and feat_name not in defined_feat:
            config["name"] = feat_name
            feature_configs.append(config)

    return gen_matrix(feature_configs, n_samples)
