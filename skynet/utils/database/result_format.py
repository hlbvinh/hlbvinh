from typing import Dict, List

import pandas as pd
import pytz
from pytz.exceptions import UnknownTimeZoneError

from ..types import Record


def format_sensors_cassandra(rows: List) -> Record:
    return pd.DataFrame(rows).mean().to_dict()


def parse_timezone(tz: Dict[str, str]):
    try:
        return pytz.timezone(tz["timezone"])

    except (UnknownTimeZoneError, AttributeError):
        return pytz.timezone("Asia/Hong_Kong")
