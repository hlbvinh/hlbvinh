from datetime import datetime

import pytest
from asynctest import mock

from ...prediction.climate_model import QUANTITIES
from ...utils import cache_util
from ...utils.types import ComfortPrediction

TIMESTAMP = datetime(2019, 9, 1)


@pytest.mark.asyncio
async def test_get_comfort(default_comfort):
    default_comfort.prediction_request = mock.CoroutineMock(
        return_value={"prediction": [1.1]}
    )
    assert await default_comfort.predict() == 1.1


@pytest.fixture
async def comfort_with_adjustment(default_comfort, connections, feedback, device_id):
    feedback_prediction = ComfortPrediction(
        feedback["feedback"], 2.5, 40, feedback["created_on"]
    )
    await cache_util.set_comfort_prediction(
        redis=connections.redis,
        key_arg=(device_id, feedback["user_id"]),
        value=feedback_prediction,
    )
    return default_comfort


@pytest.fixture
def nearby_users(user_id):
    return set([user_id, "user1"])


@pytest.fixture
def latest_feedbacks(feedback, nearby_users):
    feedbacks = []

    for user_id in nearby_users:
        single_feedback = feedback.copy()
        single_feedback["user_id"] = user_id
        feedbacks.append(single_feedback)

    return feedbacks


@pytest.fixture
def adjusted_predictions(nearby_users):
    return [1.1] * len(nearby_users)


@pytest.mark.asyncio
async def test_get_comfort_with_adjustment(
    comfort_with_adjustment, nearby_users, latest_feedbacks, adjusted_predictions
):
    comfort_with_adjustment.nearby_users = nearby_users
    comfort_with_adjustment.latest_feedbacks = latest_feedbacks
    comfort_with_adjustment._adjust_predictions = mock.CoroutineMock(
        return_value=adjusted_predictions
    )

    adjusted = await comfort_with_adjustment.predict_adjusted()
    assert adjusted == comfort_with_adjustment.multi_user_predictions_average(
        adjusted_predictions
    )


@pytest.fixture
def climate_predictions(sensors):
    return [[sensors[quantity] for quantity in QUANTITIES]] * 3


@pytest.mark.asyncio
async def test_from_climate_predictions(default_comfort, climate_predictions):
    assert len(
        await default_comfort.from_climate_predictions(climate_predictions)
    ) == len(climate_predictions)


@pytest.mark.parametrize(
    "nearby_users, latest_feedbacks, result",
    [
        pytest.param(
            [],
            [],
            [{}],
            id="no nearby users feedbacks if there are no nearby users and latest feedbacks",
        ),
        pytest.param(
            ["user_id"],
            [],
            [{}],
            id="no nearby users feedbacks if there are no latest feedbacks",
        ),
        pytest.param(
            ["user_id1", "user_id2"],
            [dict(user_id="user_id2")],
            [dict(user_id="user_id2")],
            id="consider only the nearby user who has given a feedback",
        ),
        pytest.param(
            ["user_id1"],
            [dict(user_id="user_id2", created_on=TIMESTAMP)],
            [dict(user_id="user_id2", created_on=TIMESTAMP)],
            id="when no nearby users have given feedback, use the latest available feedback",
        ),
        pytest.param(
            [],
            [dict(user_id="user_id1", created_on=TIMESTAMP)],
            [dict(user_id="user_id1", created_on=TIMESTAMP)],
            id="when there are no nearby users, use the latest available feedback",
        ),
    ],
)
def test_nearby_users_feedbacks(
    default_comfort, nearby_users, latest_feedbacks, result
):
    default_comfort.nearby_users = nearby_users
    default_comfort.latest_feedbacks = latest_feedbacks
    assert default_comfort.nearby_users_feedbacks == result
