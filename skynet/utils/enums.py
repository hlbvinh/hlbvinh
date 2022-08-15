from enum import Enum


class NearbyUser(Enum):
    USER_IN = "user_in"
    USER_OUT = "user_out"
    DEVICE_CLEAR = "device_clear"


class Power:
    ON = "on"
    OFF = "off"


class VentilationState:
    ON = "on"
    OFF = "off"


class EventPriority(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2
