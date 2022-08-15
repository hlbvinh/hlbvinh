from datetime import datetime, timedelta
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor
from aredis import StrictRedis as Redis
from scipy.sparse import csr_matrix
from voluptuous import Any

from skynet.utils.enums import NearbyUser
from .database.cassandra import CassandraSession
from .database.dbconnection import Pool
from .mongo import Client

ModePrefKey = NamedTuple(
    "ModePrefKey",
    [
        ("control_mode", str),
        ("quantity", Optional[str]),
        ("threshold_type", Optional[str]),
    ],
)
# setting defaults for last two values
# see https://stackoverflow.com/a/18348004/469992
ModePrefKey.__new__.__defaults__ = (None, None)  # type: ignore

ModeSelection = List[str]  # e.g. ['auto', 'cool', 'heat'] but not 'dry' and 'fan'
ModePref = NamedTuple("ModePref", [("key", ModePrefKey), ("modes", ModeSelection)])
ModeProbas = Dict[str, float]

SparseFeature = csr_matrix
Feature = pd.DataFrame
Target = Union[np.ndarray, pd.Series, pd.DataFrame]
PredictionFeature = Union[pd.DataFrame, Dict, List[Dict]]
Record = Dict[str, Any]
RecordSequence = Sequence[Record]
SensorReadings = TypeVar("SensorReadings", float, pd.Series, np.ndarray)

Connections = NamedTuple(
    "Connections",
    [
        ("mongo", Client),
        ("pool", Pool),
        ("redis", Redis),
        ("session", CassandraSession),
        ("db_service_msger", DealerActor),
    ],
)
defaults = (None,) * len(Connections._fields)
Connections.__new__.__defaults__ = defaults  # type: ignore

ComfortPrediction = NamedTuple(
    "ComfortPrediction",
    [
        ("feedback", float),
        ("feedback_prediction", float),
        ("feedback_humidex", float),
        ("feedback_timestamp", datetime),
    ],
)

# FIXME (big changes): should make them a class for better understanding of schema and content
# also use schema to validate the payload
# trade off: maintenance work on schema
# AVAILABLE_FAN_STATUS = (
#     "auto", "high", "med", "low", "med-low", "med-high", "very-high", "quiet",
#     "night", "off", "oscillate"
# )
# AVAILABLE_LOUVER_STATUS = (
#     "up", "mid", "down", "on", "off", "mid-up", "mid-down", "auto", "oscillate", "swing",
#     "up-high", "up-low", "mid-high", "mid-low", "down-high", "down-low"
# )
# AVAILABLE_MODE = ("auto", "cool", "dry", "fan", "heat")
# AVAILABLE_SWING_STATUS = (
#     "on", "off", "auto", "left", "mid-left", "mid", "mid-right", "right", "oscillate",
#     "left-right", "up-down", "both"
# )
# AVAILABLE_CONTROL_QUANTITIES = (
#     "ambi", "humidex", "temperature", "manual", "climate", "managed_manual", "off",
#     "away_humidity_upper", "away_temperature_upper", "away_humidex_upper",
#     "away_humidity_lower", "away_temperature_lower", "away_humidex_lower"
# )

DeviceID = str
# UserID = str
ApplianceID = str

Sensors = Dict[str, Any]
# {"temperature": 25.42,
#  "humidity": 52.6,
#  "compensated": True,
#  "created_on": datetime.datetime(2020, 12, 1, 5, 52, 24),
#  "luminosity": 1252.0,
#  "humidex": 29.352580486156356}
# Sensor_ = Schema({
#     "compensated": bool,
#     "created_on": datetime,
#     "humidex": float,
#     "humidity": float,
#     "luminosity": float,
#     "temperature": float
# })

Feedback = Dict[str, Any]
# [{'created_on': datetime.datetime(2020, 10, 29, 4, 58, 57),
#   'feedback': 3.0,
#   'user_id': '08f62869-b3e0-456b-9503-6726673e4f0a'},
#  {'created_on': datetime.datetime(2020, 11, 11, 7, 20, 42),
#   'feedback': 3.0,
#   'user_id': '25a4af9e-abd8-49c9-9280-1d147edc203a'}]
# UserFeedback_ = Schema({
#     "created_on": datetime,
#     "feedback": float,
#     "user_id": str
# })

ModeFeedback = Dict[str, Any]
# {'created_on': datetime.datetime(2020, 11, 4, 10, 13, 13),
#           'device_id': '96f0ebef-28ab-4ed2-9b25-0b5e45885cf0',
#           'mode_feedback': 'cool',
#           'user_id': '25a4af9e-abd8-49c9-9280-1d147edc203a'}
# ModeFeedback_ = Schema({
#     "created_on": datetime,
#     "device_id": str,
#     "mode_feedback": In(AVAILABLE_MODE),
#     "user_id": str
# }, extra=REMOVE_EXTRA)

ControlTarget = Dict[str, Any]
# {'created_on': datetime.datetime(2020, 12, 1, 1, 30, 1),
#   'device_id': '96f0ebef-28ab-4ed2-9b25-0b5e45885cf0',
#   'origin': 'timer',
#   'quantity': 'climate',
#   'value': -1.0}
# ControlTarget_ = Schema({
#     "created_on": datetime,
#     "device_id": str,
#     "origin": str,
#     "quantity": In(AVAILABLE_CONTROL_QUANTITIES),
#     "value": Any(int, float, None)
# }, extra=REMOVE_EXTRA)

ApplianceState = Dict[str, Any]
# {'appliance_id': 'cd39fed1-8c21-486b-9a5b-399d7a5bab04',
#   'created_on': datetime.datetime(2020, 11, 30, 7, 18, 20),
#   'fan': 'low',
#   'louver': 'mid-up',
#   'mode': 'heat',
#   'origin': 'skynet_timer',
#   'power': 'on',
#   'swing': None,
#   'temperature': 16,
#   'ventilation': None}
# ApplianceState_ = Schema({
#     "appliance_id": str,
#     "created_on": datetime,
#     "fan": Any(In(AVAILABLE_FAN_STATUS), None),
#     "louver": Any(In(AVAILABLE_LOUVER_STATUS), None),
#     "mode": In(AVAILABLE_MODE),
#     "origin": str,
#     "power": In(("on", "off")),
#     "swing": Any(In(AVAILABLE_SWING_STATUS), None),
#     "temperature": Coerce(int),  # ?
#     "ventilation": Any(str, None)
# }, extra=REMOVE_EXTRA)

Timezone = Dict[str, str]
# Timezone_ = Schema({
#     "timezone": str
# }, extra=REMOVE_EXTRA)
# {'timezone': 'Asia/Hong_Kong'}

ModePreferences = Dict[ModePrefKey, ModeSelection]  # redundant?

IRFeature = Dict[str, Any]
# {'dry': {
#     'temperature': {
#         'ftype': 'select_option',
#         'value': ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
#                   '29', '30', '31']
#     },
#     'fan': {'ftype': 'select_option', 'value': ['auto', 'high', 'med', 'low']},
#     'louver': {'ftype': 'select_option', 'value': ['auto', 'up', 'mid-up', 'mid', 'mid-down']}},
#     'auto': {
#         'temperature': {
#             'ftype': 'select_option',
#             'value': ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#                       '28', '29', '30', '31']},
#         'fan': {'ftype': 'select_option', 'value': ['auto', 'high', 'med', 'low']},
#         'louver': {'ftype': 'select_option', 'value': ['auto', 'up', 'mid-up', 'mid', 'mid-down']}
#     },
#     'heat': {
#         'temperature': {
#             'ftype': 'select_option',
#             'value': ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#                       '28', '29', '30', '31']},
#         'fan': {'ftype': 'select_option', 'value': ['auto', 'high', 'med', 'low']},
#         'louver': {'ftype': 'select_option', 'value': ['auto', 'up', 'mid-up', 'mid', 'mid-down']}
#     },
#     'fan': {
#         'temperature': {'ftype': 'select_option', 'value': ['27']},
#         'fan': {'ftype': 'select_option', 'value': ['auto', 'high', 'med', 'low']},
#         'louver': {'ftype': 'select_option', 'value': ['auto', 'up', 'mid-up', 'mid', 'mid-down']}
#     },
#     'cool': {
#         'temperature': {
#             'ftype': 'select_option',
#             'value': ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
#                       '28', '29', '30', '31']
#         },
#         'fan': {'ftype': 'select_option', 'value': ['auto', 'high', 'med', 'low']},
#         'louver': {'ftype': 'select_option', 'value': ['auto', 'up', 'mid-up', 'mid', 'mid-down']}
#     }
# }
# IRFeature_ = Schema({})  # TODO: create schema for IRFeature

Config = Dict[str, Any]
Weather = Dict[str, Any]
UserIdSet = Set[str]
UserFeatures = Dict[str, Any]

NearbyUserAction = NamedTuple(
    "NearbyUserAction", [("action", NearbyUser), ("user_id", Optional[str])]
)

Prediction = NamedTuple(
    "Prediction",
    [
        ("temperature", float),
        ("humidity", float),
        ("humidex", float),
        ("created_on", datetime),
        ("horizon", timedelta),
    ],
)

BasicState = NamedTuple(
    "BasicState", [("mode", str), ("temperature", str), ("created_on", datetime)]
)

AircomResponse = Tuple[int, Dict[str, Any]]

DeploymentSettings = NamedTuple(
    "DeploymentSettings",
    [
        ("temperature", Union[int, str, float]),
        ("mode", str),
        ("power", str),
        ("fan", Union[int, str, float]),
        ("louver", Union[int, str, float, None]),
        ("swing", Union[int, str, float, None]),
        ("ventilation", Optional[str]),
        ("button", str),
        ("device_id", str),
    ],
)
defaults = (None,) * len(DeploymentSettings._fields)
DeploymentSettings.__new__.__defaults__ = defaults  # type: ignore

BasicDeployment = NamedTuple(
    "BasicDeployment",
    [
        ("power", Optional[str]),
        ("mode", Optional[str]),
        ("temperature", Optional[Union[int, str, float]]),
        ("ventilation", Optional[str]),
    ],
)
defaults = (None,) * len(BasicDeployment._fields)
BasicDeployment.__new__.__defaults__ = defaults  # type: ignore

StateTemperature = Optional[Union[str, int]]

AutomatedDemandResponse = NamedTuple(
    "AutomatedDemandResponse",
    [
        ("action", str),
        ("signal_level", int),
        ("created_on", datetime),
        ("group_name", str),
    ],
)
AutomatedDemandResponse.__new__.__defaults__ = (datetime.utcnow(), "")  # type: ignore
