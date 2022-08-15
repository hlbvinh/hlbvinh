from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import pytest
import pandas as pd
import numpy as np
from skynet.utils.script_utils import get_sample_limit_according_to_number_of_months


@pytest.fixture
def data_less_than_month_and_validation_interval():
    months = 3
    validation_interval = 7
    less_data_offset = 5

    date_end = parse("2018-05-01")
    date_start = (
        date_end
        - relativedelta(months=months)
        - relativedelta(days=validation_interval - less_data_offset)
    )

    days = pd.date_range(date_start, date_end, freq="D")
    data = np.random.randint(1, high=100, size=len(days))

    df = pd.DataFrame({"timestamp": days, "data": data})
    return df


@pytest.fixture
def data_more_than_month_and_validation_interval():
    months = 3
    validation_interval = 7
    more_data_offset = 5

    date_end = parse("2018-05-01")
    date_start = (
        date_end
        - relativedelta(months=months)
        - relativedelta(days=validation_interval + more_data_offset)
    )

    days = pd.date_range(date_start, date_end, freq="D")
    data = np.random.randint(1, high=100, size=len(days))

    df = pd.DataFrame({"timestamp": days, "data": data})
    return df


def test_get_sample_limit_according_to_number_of_months(
    data_less_than_month_and_validation_interval,
    data_more_than_month_and_validation_interval,
):
    months = 3
    validation_interval = 7
    correct_sample_limit = 93

    accurate_sample_limit = get_sample_limit_according_to_number_of_months(
        data_more_than_month_and_validation_interval, months, validation_interval
    )
    inaccurate_sample_limit = get_sample_limit_according_to_number_of_months(
        data_less_than_month_and_validation_interval, months, validation_interval
    )

    assert correct_sample_limit == accurate_sample_limit
    assert correct_sample_limit > inaccurate_sample_limit
