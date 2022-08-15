from datetime import datetime

import pytest
import pytz
from voluptuous import Invalid

from ..utils import compensation, event_parsing
from ..utils.config import MAX_COMFORT, MIN_COMFORT
from ..utils.enums import NearbyUser
from ..utils.types import ModePref, ModePrefKey

TIMESTAMP = datetime(2015, 1, 1)


def test_parse_host_port():
    host, port = "127.0.0.1", "6677"
    address = f"tcp://{host}:{port}"
    assert (host, port) == event_parsing.parse_host_port(address)


def test_to_datetime():
    fun = event_parsing.to_naive_datetime
    d_str = "2015-01-01 01:02:03"
    d = datetime(2015, 1, 1, 1, 2, 3)
    assert fun(d_str) == d
    assert fun(d) == d

    utc_datetime = d.replace(tzinfo=pytz.UTC)
    assert fun(utc_datetime) == d


@pytest.mark.parametrize(
    "msg,target",
    [
        (
            {
                "QUANTITY": "Climate",
                "value": 5,
                "origin": "User",
                "created_on": TIMESTAMP,
            },
            {
                "quantity": "climate",
                "value": 5.0,
                "origin": "user",
                "created_on": TIMESTAMP,
            },
        ),
        (
            {
                "QUANTITY": "Climate",
                "value": None,
                "origin": "user",
                "created_on": TIMESTAMP,
            },
            {
                "quantity": "climate",
                "value": None,
                "origin": "user",
                "created_on": TIMESTAMP,
            },
        ),
    ],
)
def test_parse_control_target(msg, target):
    assert event_parsing.parse_control_target(msg) == target


@pytest.mark.parametrize("bad_target", [dict(quantity="climate")])
def test_parse_control_target_bad(bad_target):
    with pytest.raises(Invalid):
        event_parsing.parse_control_target(bad_target)


@pytest.mark.parametrize(
    "msg,sensor",
    [
        pytest.param(
            dict(created_on=TIMESTAMP, HM=2.0, TP=1.0, LU=dict(infrared_spectrum=3.0)),
            dict(
                temperature=1.0,
                humidity=2.0,
                luminosity=3.0,
                created_on=TIMESTAMP,
                compensated=False,
            ),
        ),
        pytest.param(
            dict(
                created_on=TIMESTAMP,
                HM_refined=2.0,
                TP_refined=1.0,
                LU=dict(infrared_spectrum=3.0),
            ),
            dict(
                temperature=1.0,
                humidity=2.0,
                luminosity=3.0,
                created_on=TIMESTAMP,
                compensated=True,
            ),
        ),
        pytest.param(
            dict(created_on=TIMESTAMP, HM_refined=2.0, TP_refined=1.0),
            dict(temperature=1.0, humidity=2.0, created_on=TIMESTAMP, compensated=True),
            id="Check for no luminosity case",
        ),
    ],
)
def test_parse_sensor_event_parsing(msg, sensor):
    assert event_parsing._parse_sensors(msg) == sensor


@pytest.fixture
def sensor_msg_legacy():
    return dict(created_on=TIMESTAMP, HM=2.0, TP=1.0, LU=dict(infrared_spectrum=3.0))


@pytest.fixture
def sensor_msg_refined():
    return dict(
        created_on=TIMESTAMP,
        HM_refined=2.0,
        TP_refined=1.0,
        LU=dict(infrared_spectrum=3.0),
    )


def test_parse_sensor_event_compensation_legacy(sensor_msg_legacy):
    # test data type
    sensor = event_parsing.parse_sensor_event(sensor_msg_legacy)
    assert sensor["temperature"] == compensation.compensate_temperature(
        sensor_msg_legacy["TP"]
    )
    assert sensor["humidity"] == compensation.compensate_humidity(
        sensor_msg_legacy["HM"]
    )
    assert sensor[compensation.COMPENSATE_COLUMN] is True


def test_parse_sensor_event_compensation_refined(sensor_msg_refined):
    sensor = event_parsing.parse_sensor_event(sensor_msg_refined)
    assert sensor["temperature"] == sensor_msg_refined["TP_refined"]
    assert sensor["humidity"] == sensor_msg_refined["HM_refined"]
    assert sensor[compensation.COMPENSATE_COLUMN] is True


def test_parse_sensor_event_validation():
    with pytest.raises(Invalid):
        msg = {
            "created_on": TIMESTAMP,
            "HM": None,
            "TP": 1.0,
            "LU": {"infrared_spectrum": 3.0},
        }
        event_parsing._parse_sensors(msg)

    # test missing data
    with pytest.raises(Invalid):
        msg = {"created_on": TIMESTAMP, "HM": 0.0, "LU": {"infrared_spectrum": 3.0}}
        event_parsing._parse_sensors(msg)


def test_parse_sensor_event_int_valid():
    msg = {
        "created_on": TIMESTAMP,
        "HM": 10,
        "TP": 1.0,
        "LU": {"infrared_spectrum": 3.0},
    }
    event_parsing._parse_sensors(msg)


def test_parse_sensor_co2():
    msg = {
        "created_on": TIMESTAMP,
        "HM": 10,
        "TP": 1.0,
        "LU": {"infrared_spectrum": 3.0},
        "CO": 808.7,
    }
    sensors = event_parsing._parse_sensors(msg)
    assert sensors["co2"] == msg["CO"]


def test_parse_daikin_missing_humidity():
    msg = {"created_on": TIMESTAMP, "HM": float("nan"), "TP": 1.0}
    sensors = event_parsing.parse_sensor_event(msg)
    assert sensors["humidity"] == event_parsing.AVERAGE_HUMIDITY


def test_parse_irprofile_event(ir_feature):
    msg = {"device_id": "12789", "irprofile_id": "748912", "irfeature": ir_feature}
    assert event_parsing.parse_irprofile_event(msg) == ir_feature


def test_parse_quantity_field():
    fields = [
        ("comfort", "comfort", None, "climate"),
        ("temperature", "temperature", None, "temperature"),
        ("away", "temperature", "upper", "away_temperature_upper"),
        ("away", "temperature", "lower", "away_temperature_lower"),
        ("away", "humidity", "upper", "away_humidity_upper"),
        ("off", None, None, "off"),
        ("manual", None, None, "manual"),
        ("manual", None, None, "baby_mode"),
        ("managed_manual", "set_temperature", None, "managed_manual"),
    ]
    for control_mode, quantity, threshold, field in fields:
        assert (
            control_mode,
            quantity,
            threshold,
        ) == event_parsing.parse_quantity_field(field)


def test_parse_mode_prefs():
    """
    parse_mode_prefs(msg, multimodes=MULTIMODES)
    """
    msc_msg = {"device_id": "blah", "created_on": "blahblah"}
    control_modes = [
        ("comfort", "comfort", None, "climate"),
        ("temperature", "temperature", None, "temperature"),
        ("away", "temperature", "upper", "away_temperature_upper"),
        ("away", "temperature", "lower", "away_temperature_lower"),
        ("away", "humidity", "upper", "away_humidity_upper"),
    ]
    modes_prefs = [
        (["heat", "cool"], {"heat": 1, "cool": 1, "fan": 0, "auto": 0, "dry": 0}),
        (
            ["heat", "cool", "dry"],
            {"heat": 1, "cool": 1, "fan": 0, "auto": 0, "dry": 1},
        ),
        (
            ["heat", "cool", "dry", "auto"],
            {"heat": 1, "cool": 1, "fan": 0, "auto": 1, "dry": 1},
        ),
        (["heat"], {"heat": 1, "cool": 0, "fan": 0, "auto": 0, "dry": 0}),
        (
            ["heat", "dry", "auto"],
            {"heat": 1, "cool": 0, "fan": 0, "auto": 1, "dry": 1},
        ),
    ]
    for control_mode, mode_pref in zip(control_modes, modes_prefs):
        msg = msc_msg.copy()
        control_mode, quantity, threshold, field = control_mode
        modes, mode_msg = mode_pref
        msg["quantity"] = field
        msg.update(mode_msg)
        key = ModePrefKey(control_mode, quantity, threshold)
        assert ModePref(key, sorted(modes)) == event_parsing.parse_mode_prefs(msg)


@pytest.fixture
def bad_feedbacks():
    return [
        {"feedback": 1},
        {},
        {"user_id": "oien"},
        {"feedback": 1, "user_id": None, "created_on": datetime.utcnow()},
    ] + [
        dict(feedback=feedback, user_id="user_id", created_on=datetime.utcnow())
        for feedback in [+100, -50.0]
    ]


@pytest.fixture
def good_feedback():
    return [
        dict(feedback=feedback, user_id="user_id", created_on=datetime.utcnow())
        for feedback in [1, -1.0]
    ]


def test_parse_feedback(bad_feedbacks, good_feedback):
    for msg in bad_feedbacks:
        with pytest.raises(Invalid):
            event_parsing.parse_feedback_event(msg)
    for msg in good_feedback:
        feedback = event_parsing.parse_feedback_event(msg)
        assert MIN_COMFORT <= feedback["feedback"] <= MAX_COMFORT


@pytest.fixture
def mode_feedback(device_id):
    return dict(mode_feedback="fan", device_id=device_id, created_on=datetime.utcnow())


@pytest.fixture
def bad_mode_feedback(device_id):
    return dict(
        mode_feedback="unknow", device_id=device_id, created_on=datetime.utcnow()
    )


def test_parse_mode_feedback(mode_feedback):
    assert event_parsing.parse_mode_feedback_event(mode_feedback)


def test_parse_bad_mode_feedback(bad_mode_feedback):
    with pytest.raises(Invalid):
        event_parsing.parse_mode_feedback_event(bad_mode_feedback)


@pytest.fixture
def nearby_user_action_message(device_id, user_id):
    return dict(
        device_id=device_id,
        user_id=user_id,
        action=NearbyUser.USER_IN.value,
        created_on=datetime.utcnow(),
    )


def test_parse_nearby_user_event(nearby_user_action_message):
    nearby_user_action = event_parsing.parse_nearby_user_event(
        nearby_user_action_message
    )
    assert nearby_user_action.user_id == nearby_user_action_message["user_id"]
    assert nearby_user_action.action == NearbyUser(nearby_user_action_message["action"])


@pytest.fixture
def adr_message(device_id):
    return dict(
        device_id=device_id,
        action="start",
        signal_level=4,
        create_time=datetime.utcnow(),
        group_name="Comfort DR Trial",
    )


def test_parse_adr(adr_message):
    adr = event_parsing.parse_automated_demand_response(adr_message)
    assert adr.created_on == adr_message["create_time"]
    assert adr.group_name == adr_message["group_name"].lower()
