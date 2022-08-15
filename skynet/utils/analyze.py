from itertools import groupby
from typing import List, Dict, Any, Iterator
import logging
import numpy as np
import pandas as pd
from scipy.signal.windows import gaussian

from ..utils.log_util import get_logger

log = get_logger(__name__)

MAX_TARGET_LENGTH = 36
SIGNIFICANT_TEMPERATURE_DELTA_MOVEMENT = 0.4
SIGNIFICANT_CUMULATIVE_TEMPERATURE_DELTA_MOVEMENT = 0.13
MAXIMUM_ALLOWED_SMOOTH_TEMPERATURE_VARIATION_IN_FIVE_MINUTES = 0.35
ALLOWED_TEMPERATURE_DELTA_FOR_TREND_CHANGING_BIT = 0.25


def convolve_with_gaussian(t, std):
    """Convolve series with gaussian of given standard deviation.

    PARAMETERS
    ----------
    t: array-like or pandas DataFrame
        values of timeseries

    std: float
        standard deviation of gaussian window

    """
    if isinstance(t, pd.DataFrame):
        ret = t.copy()
        for col in ret:
            ret[col] = convolve_with_gaussian(np.asarray(ret[col]), std)
        return ret
    if isinstance(t, pd.Series):
        ret = t.copy()
        ret[:] = convolve_with_gaussian(np.asarray(t), std)
        return ret
    # window with width of 6 standard deviations should look smooth enough
    nwin = int(np.ceil(std) * 6)
    pad_width = nwin // 2
    return np.convolve(
        np.pad(t, (pad_width,), mode="edge"),
        (lambda x: x / sum(x))(gaussian(nwin, std)),
        mode="valid",
    )[: len(t)]


def filter_bad_targets(
    feature: Dict[str, Any],
    target: pd.DataFrame,
    weather: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
) -> pd.DataFrame:

    try:
        modified_target = target.copy(deep=True)

        modified_target["weather_temperature_out"] = weather["temperature_out"].values

        return filter_target_based_on_current_mode(
            target, modified_target, mode_hist, previous_set_temperature, feature
        )

    except KeyError as exc:
        log(exc, level=logging.ERROR)
        return target


def filter_target_based_on_current_mode(
    target: pd.DataFrame,
    modified_target: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
    feature: Dict[str, Any],
) -> pd.DataFrame:

    target_to_return = pd.DataFrame()

    if feature["power"] == "off":
        if room_temperature_behaving_like_off_mode(modified_target, mode_hist):
            target_to_return = target

    elif feature["mode"] == "cool":
        if room_temperature_behaving_like_cool_mode(
            modified_target,
            mode_hist,
            previous_set_temperature,
            feature["temperature_set"],
        ):
            target_to_return = target

    elif feature["mode"] == "dry":
        if room_temperature_behaving_like_dry_mode(
            modified_target,
            mode_hist,
            previous_set_temperature,
            feature["temperature_set"],
        ):
            target_to_return = target

    elif feature["mode"] == "heat":
        if room_temperature_behaving_like_heat_mode(
            modified_target,
            mode_hist,
            previous_set_temperature,
            feature["temperature_set"],
        ):
            target_to_return = target

    elif feature["mode"] == "fan":
        if room_temperature_behaving_like_fan_mode(modified_target, mode_hist):
            target_to_return = target

    elif feature["mode"] == "auto":
        if room_temperature_behaving_like_auto_mode(
            modified_target,
            mode_hist,
            previous_set_temperature,
            feature["temperature_set"],
        ):
            target_to_return = target

    return target_to_return


def room_temperature_behaving_like_off_mode(
    target: pd.DataFrame, mode_hist: str
) -> bool:
    if is_target_a_short_sample(target):
        return True
    if is_previous_mode_stale_mode(mode_hist):
        return is_temperature_matching_outside_temperature(target)
    if is_previous_mode_non_heating_mode(mode_hist):
        return is_temperature_maintaining_or_heating(target)
    if is_previous_mode_non_cooling_mode(mode_hist):
        return is_temperature_maintaining_or_cooling(target)
    return True


def is_temperature_matching_outside_temperature(target: pd.DataFrame) -> bool:

    if is_temperature_changing_direction_in_between(target):
        return False
    if is_temperature_variation_minimal(target, allowed_variation_in_three_hours=1):
        return True
    if is_temperature_variation_too_much(
        target, maximum_allowed_variation_in_three_hours=4
    ) or is_temperature_changing_too_fast(target):
        return False
    if is_outdoor_hotter_than_indoor(target, 0):
        return is_temperature_maintaining_or_heating(target)
    if is_outdoor_colder_than_indoor(target, 0):
        return is_temperature_maintaining_or_cooling(target)
    return True


def room_temperature_behaving_like_cool_mode(
    target: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
    set_temperature: float,
) -> bool:
    if is_target_a_short_sample(target):
        return True
    if is_previous_mode_non_cooling_mode(
        mode_hist
    ) or is_set_temperature_lower_than_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return is_temperature_maintaining_or_cooling(target)
    if is_set_temperature_same_as_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return is_temperature_variation_minimal(
            target, allowed_variation_in_three_hours=3
        )
    if is_set_temperature_higher_than_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return not is_temperature_showing_any_weird_behavior(
            target, previous_set_temperature, set_temperature
        )
    return True


def room_temperature_behaving_like_dry_mode(
    target: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
    set_temperature: float,
) -> bool:
    return room_temperature_behaving_like_cool_mode(
        target, mode_hist, previous_set_temperature, set_temperature
    )


def room_temperature_behaving_like_heat_mode(
    target: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
    set_temperature: float,
) -> bool:

    if is_target_a_short_sample(target):
        return True
    if is_previous_mode_non_heating_mode(
        mode_hist
    ) or is_set_temperature_higher_than_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return is_temperature_maintaining_or_heating(target)
    if is_set_temperature_same_as_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return is_temperature_variation_minimal(
            target, allowed_variation_in_three_hours=2
        )
    if is_set_temperature_lower_than_previous_set_temperature(
        previous_set_temperature, set_temperature
    ):
        return not is_temperature_showing_any_weird_behavior(
            target, previous_set_temperature, set_temperature
        )
    return True


def room_temperature_behaving_like_fan_mode(
    target: pd.DataFrame, mode_hist: str
) -> bool:

    if is_target_a_short_sample(target):
        return True
    if is_previous_mode_stale_mode(mode_hist):
        return is_temperature_variation_minimal(
            target, allowed_variation_in_three_hours=2
        )
    if mode_hist in ["heat"] and is_outdoor_colder_than_indoor(target, 0.0):
        return is_temperature_maintaining_or_cooling(target)
    return is_temperature_maintaining_or_heating(target)


def room_temperature_behaving_like_auto_mode(
    target: pd.DataFrame,
    mode_hist: str,
    previous_set_temperature: float,
    set_temperature: float,
) -> bool:

    if is_target_a_short_sample(target):
        return True

    if mode_hist in ["auto"] and set_temperature == previous_set_temperature:
        return is_temperature_variation_minimal(
            target, allowed_variation_in_three_hours=2
        )

    if is_temperature_variation_minimal(target, allowed_variation_in_three_hours=2):
        return True

    if is_temperature_changing_direction_in_between(target):
        return False

    return True


def is_temperature_maintaining_or_heating(target: pd.DataFrame) -> bool:

    if is_temperature_variation_minimal(target, allowed_variation_in_three_hours=1):
        return True

    temperature_deltas = calculate_temperature_deltas(target)

    return not is_temperature_getting_colder(temperature_deltas)


def is_temperature_getting_colder(temperature_deltas: pd.Series) -> bool:
    if min(temperature_deltas) < -SIGNIFICANT_TEMPERATURE_DELTA_MOVEMENT:
        return True

    for is_negative_streak, streak in groupby(temperature_deltas, key=lambda x: x < 0):
        if is_negative_streak and is_any_of_the_cumulative_mean_significant(streak):
            return True
    return False


def is_temperature_maintaining_or_cooling(target: pd.DataFrame) -> bool:

    if is_temperature_variation_minimal(target, allowed_variation_in_three_hours=1):
        return True

    temperature_deltas = calculate_temperature_deltas(target)

    return not is_temperature_getting_hotter(temperature_deltas)


def is_temperature_getting_hotter(temperature_deltas: pd.Series) -> bool:
    # if the temperature becomes hot more than 0.4 degrees in 5 minutes
    # then its enough to know that temperature is getting hotter in the
    # room. 0.4 degree is based on sample observations.
    if max(temperature_deltas) > SIGNIFICANT_TEMPERATURE_DELTA_MOVEMENT:
        return True

    for is_positive_streak, streak in groupby(temperature_deltas, key=lambda x: x > 0):
        if is_positive_streak and is_any_of_the_cumulative_mean_significant(streak):
            return True

    return False


def calculate_temperature_deltas(target: pd.DataFrame) -> pd.Series:
    temperature_deltas = (
        target["temperature"] - target["temperature"].shift(1)
    ).dropna()
    return temperature_deltas


def is_temperature_showing_any_weird_behavior(
    target: pd.DataFrame, previous_set_temperature: float, set_temperature: float
) -> bool:

    if is_high_temperature_variation_per_degree_set_temperature(
        target, previous_set_temperature, set_temperature
    ):
        return True

    if is_temperature_variation_minimal(target, allowed_variation_in_three_hours=2):
        return False

    if is_temperature_changing_direction_in_between(target):
        return True

    return False


def is_high_temperature_variation_per_degree_set_temperature(
    target: pd.DataFrame, previous_set_temperature: float, set_temperature: float
) -> bool:
    # For the samples where a single degree change in set temperature i.e. 19->20
    # is causing a high change in room temperature. The allowed variation is
    # dependent upon two factors - delta set temperature (20 - 19 = 1 degree)
    # and sample length.
    return target["temperature"].max() - target["temperature"].min() > (
        allowed_variation_for_set_temperature_delta_and_sample_length(
            previous_set_temperature, set_temperature, len(target)
        )
    )


def allowed_variation_for_set_temperature_delta_and_sample_length(
    previous_set_temperature: float, set_temperature: float, target_length: int
) -> float:
    # Equation of a plane
    # 55x + 150y - 330z - 810 = 0
    # Deduced from points - (length, delta, allowed variation)
    # (36, 1, 4); (36, 12, 9); (6, 12, 4); (6, 1, 1)
    target_length = max(target_length, 6)
    return (
        55 * target_length
        + 150 * (abs(previous_set_temperature - set_temperature))
        - 810
    ) / 330


def is_temperature_changing_direction_in_between(target: pd.DataFrame) -> bool:

    temperature_deltas = calculate_temperature_deltas(target)

    if is_temperature_heating_then_suddenly_cooling(temperature_deltas):
        return True

    if is_temperature_cooling_then_suddenly_heating(temperature_deltas):
        return True

    return False


def is_temperature_heating_then_suddenly_cooling(
    temperature_deltas: List[bool],
) -> bool:
    is_series_already_heating = False
    does_series_start_cooling = False

    for is_positive_streak, streak in groupby(temperature_deltas, key=lambda x: x > 0):

        if not is_any_of_the_cumulative_mean_significant(streak):
            continue

        if is_positive_streak:
            is_series_already_heating = True
        elif not is_positive_streak and is_series_already_heating:
            does_series_start_cooling = True

    return is_series_already_heating and does_series_start_cooling


def is_temperature_cooling_then_suddenly_heating(
    temperature_deltas: List[bool],
) -> bool:

    is_series_already_cooling = False
    does_series_start_heating = False

    for is_positive_streak, streak in groupby(temperature_deltas, key=lambda x: x > 0):

        if not is_any_of_the_cumulative_mean_significant(streak):
            continue

        if is_positive_streak and is_series_already_cooling:
            does_series_start_heating = True
        elif not is_positive_streak:
            is_series_already_cooling = True

    return is_series_already_cooling and does_series_start_heating


def is_any_of_the_cumulative_mean_significant(group: Iterator[float]) -> bool:
    streak = list(group)

    # First value is allowed to show slightly more variation
    # since it is generally a trend changing value.
    if abs(streak[0]) > ALLOWED_TEMPERATURE_DELTA_FOR_TREND_CHANGING_BIT:
        return True

    if max([abs(x) for x in streak]) > SIGNIFICANT_TEMPERATURE_DELTA_MOVEMENT:
        return True

    cumulative_means = calculate_cumulative_mean(streak)

    # Mean of the first value doesn't make sense since its
    # only a single value. However, if the mean of any two
    # or more cumulative values is more than allowed
    # delta movement then it signifies the variation
    # is significant
    return any(
        [
            abs(cumulative_mean) > SIGNIFICANT_CUMULATIVE_TEMPERATURE_DELTA_MOVEMENT
            for cumulative_mean in cumulative_means[1:]
        ]
    )


def calculate_cumulative_mean(streak: List[float]) -> List[float]:
    cumulative_sums = np.cumsum(streak)
    return [
        cumulative_sum / (index + 1)
        for index, cumulative_sum in enumerate(cumulative_sums)
    ]


def is_temperature_variation_minimal(
    target: pd.DataFrame, allowed_variation_in_three_hours: int = 3
) -> bool:
    return (
        target["temperature"].max() - target["temperature"].min()
        < allowed_variation_in_three_hours * len(target) / MAX_TARGET_LENGTH
    )


def is_temperature_variation_too_much(
    target: pd.DataFrame, maximum_allowed_variation_in_three_hours: int = 4
) -> bool:
    return (
        target["temperature"].max() - target["temperature"].min()
        > maximum_allowed_variation_in_three_hours * len(target) / MAX_TARGET_LENGTH
    )


def is_temperature_changing_too_fast(target: pd.DataFrame) -> bool:

    temperature_deltas = calculate_temperature_deltas(target)

    return (
        np.max([abs(x) for x in temperature_deltas])
        > MAXIMUM_ALLOWED_SMOOTH_TEMPERATURE_VARIATION_IN_FIVE_MINUTES
    )


def is_target_a_short_sample(target: pd.DataFrame) -> bool:
    return len(target) == 1


def is_previous_mode_stale_mode(mode_hist: str) -> bool:
    return mode_hist in ["off", "fan"]


def is_previous_mode_non_cooling_mode(mode_hist: str) -> bool:
    return mode_hist not in ["cool", "dry", "auto"]


def is_previous_mode_non_heating_mode(mode_hist: str) -> bool:
    return mode_hist in ["cool", "dry", "off", "fan"]


def is_set_temperature_lower_than_previous_set_temperature(
    previous_set_temperature: float, set_temperature: float
) -> bool:  # XXX - make sure that its similar - heating or cooling mode
    return set_temperature < previous_set_temperature


def is_set_temperature_same_as_previous_set_temperature(
    previous_set_temperature: float, set_temperature: float
) -> bool:
    return set_temperature == previous_set_temperature


def is_set_temperature_higher_than_previous_set_temperature(
    previous_set_temperature: float, set_temperature: float
) -> bool:
    return set_temperature > previous_set_temperature


def is_outdoor_hotter_than_indoor(
    target: pd.DataFrame, outside_climate_threshold: float
) -> bool:
    mean_outdoor_indoor_temperature_difference = np.mean(
        target["weather_temperature_out"] - target["temperature"]
    )
    return mean_outdoor_indoor_temperature_difference > outside_climate_threshold


def is_outdoor_colder_than_indoor(
    target: pd.DataFrame, outside_climate_threshold: float
) -> bool:
    return not is_outdoor_hotter_than_indoor(target, outside_climate_threshold)
