import datetime

import pytest

from ..utils.database import queries
from ..utils.enums import Power


def test_last_appliance_state(db, device_id):
    last_state = queries.execute(db, *queries.query_last_appliance_state(device_id))
    assert last_state
    assert last_state[0]["power"] in [Power.ON, Power.OFF]


def test_last_on_appliance_state(db, device_id):
    last_state = queries.execute(db, *queries.query_last_on_appliance_state(device_id))
    assert last_state
    assert last_state[0]["power"] == Power.ON


@pytest.mark.asyncio
async def test_latest_feedbacks(
    feedback_db, pool, device_id
):  # pylint: disable=unused-argument
    latest_feedbacks = await pool.execute(*queries.query_latest_feedbacks(device_id))
    single_latest_feedback = latest_feedbacks[0]
    assert isinstance(single_latest_feedback["created_on"], datetime.datetime)
    assert isinstance(single_latest_feedback["feedback"], float)
    assert isinstance(single_latest_feedback["user_id"], str)
