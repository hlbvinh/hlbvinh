import pytest

from ...control import target
from ...utils.types import AutomatedDemandResponse


@pytest.fixture
def control_mode():
    return "comfort"


@pytest.fixture
def quantity():
    return "comfort"


@pytest.fixture(params=[True, False])
def inactive_adrevent(request):
    if request.param:
        return AutomatedDemandResponse("stop", 0)
    return None


def test_no_adrpenalty_for_inactive_adrevent(inactive_adrevent, control_mode, quantity):
    assert target.ADRPenalty(inactive_adrevent, control_mode, quantity).penalty == 0


@pytest.fixture
def start_adrevent():
    return AutomatedDemandResponse("start", 2)


def test_positive_adrpenalty_when_in_cooling_modes(
    start_adrevent, control_mode, quantity
):
    adr = target.ADRPenalty(start_adrevent, control_mode, quantity)
    adr.set_predicted_mode("cool")
    assert adr.penalty > 0


def test_negative_adrpenalty_when_in_heating_modes(
    start_adrevent, control_mode, quantity
):
    adr = target.ADRPenalty(start_adrevent, control_mode, quantity)
    adr.set_predicted_mode("heat")
    assert adr.penalty < 0


@pytest.fixture
def higher_adrevent(start_adrevent):
    return AutomatedDemandResponse("start", start_adrevent.signal_level + 1)


def test_lower_adrpenalty_with_higher_signal_level(
    start_adrevent, higher_adrevent, control_mode, quantity
):

    higher_adr = target.ADRPenalty(higher_adrevent, control_mode, quantity)
    higher_adr.set_predicted_mode("cool")
    lower_adr = target.ADRPenalty(start_adrevent, control_mode, quantity)
    lower_adr.set_predicted_mode("cool")
    assert higher_adr.penalty < lower_adr.penalty


@pytest.fixture
def turn_off_adr():
    return AutomatedDemandResponse("start", 1)


def test_adrpenalty_turn_off(turn_off_adr, control_mode, quantity):
    assert target.ADRPenalty(turn_off_adr, control_mode, quantity).turn_off_appliance


@pytest.fixture(params=[True, False])
def away_upper_quantity(request):
    if request.param:
        return "temperature"
    return "humidity"


def test_adr_penalty_away_upper(away_upper_quantity, start_adrevent, inactive_adrevent):

    active_adr = target.ADRPenalty(start_adrevent, "away", away_upper_quantity, "upper")
    inactive_adr = target.ADRPenalty(
        inactive_adrevent, "away", away_upper_quantity, "upper"
    )
    assert active_adr.penalty > inactive_adr.penalty


def test_adr_penalty_away_lower(start_adrevent, inactive_adrevent):
    active_adr = target.ADRPenalty(start_adrevent, "away", "temperature", "lower")
    inactive_adr = target.ADRPenalty(inactive_adrevent, "away", "temperature", "lower")
    assert active_adr.penalty < inactive_adr.penalty


@pytest.fixture
def control_target(control_target):
    control_target.update({"quantity": "temperature", "value": 24})
    return control_target


@pytest.fixture
def default_target(device_id, sensors, control_target, start_adrevent):
    return target.Target(device_id, sensors, None, None, control_target, start_adrevent)


@pytest.mark.asyncio
async def test_target_delta_with_adr(default_target):
    default_target.mode_model_target_delta  # pylint:disable=pointless-statement

    with pytest.raises(AttributeError):
        default_target.target_delta  # pylint:disable=pointless-statement

    default_target.set_predicted_mode("cool")
    default_target.target_delta  # pylint:disable=pointless-statement
