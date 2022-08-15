import copy
from typing import Any, Dict, Optional

import pendulum
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from ..async_util import PREDICTION_TIMEOUT, request_with_timeout
from ..ir_feature import is_default_irfeature
from ..log_util import get_logger
from ..types import DeploymentSettings, UserIdSet

log = get_logger(__name__)


async def fetch_ir_feature_from_db_service(
    db_service_msger: DealerActor, appliance_id: str
) -> Dict[str, Any]:
    aircom_request = get_aircom_request("IrFeatureRead", {"appliance_id": appliance_id})
    code, response = await request_with_timeout(
        PREDICTION_TIMEOUT, db_service_msger, aircom_request
    )
    if code > 300 or code < 200:
        raise LookupError(
            f"code: {code} " + response.get("reason", "unknown error in db service")
        )
    ir_feature = parse_ir_feature_response(response)
    return ir_feature


async def fetch_nearby_users_from_db_service(
    db_service_msger: DealerActor, device_id: str
) -> UserIdSet:
    """

    Args:
        db_service_msger:
        device_id:

    Returns:
        Set of user IDs: Set[str],
            e.g. {'6faa1416-fd6b-45e8-8481-a5c812bde987', '25a4af9e-abd8-49c9-9280-1d147edc203a'}

    """
    nearby_query = """
        query ($device_id: String!, $is_checked_in: Boolean!) {
            device(device_id: $device_id){
                    device_nearby_user_status(is_checked_in:$is_checked_in){
                user_id,
                }
            }
        }
    """
    aircom_request = get_aircom_request(
        "GraphQL",
        dict(
            query=nearby_query, variables=dict(device_id=device_id, is_checked_in=True)
        ),
    )
    _, response = await request_with_timeout(
        PREDICTION_TIMEOUT, db_service_msger, aircom_request
    )
    if response["errors"] is not None:
        raise LookupError(f"Response error: {response['errors']}")
    user_ids = parse_nearby_user_response(response)
    return user_ids


def parse_nearby_user_response(response: Dict[str, Any]) -> UserIdSet:
    try:
        nearby_users = [
            user["user_id"]
            for user in response["data"]["device"]["device_nearby_user_status"]
        ]
    except (ValueError, TypeError, KeyError) as e:
        raise TypeError(f"could not parse nearby users {response}") from e

    return set(nearby_users)


def get_aircom_request(req_name: str, params: Dict[str, Any] = None) -> AircomRequest:
    params = copy.deepcopy(params)
    req = AircomRequest.new(req_name, params)
    return req


def parse_ir_feature_response(response: Dict[str, Any]):
    try:
        ir_feature = response["data"]
    except (ValueError, TypeError, KeyError) as e:
        raise TypeError(f"could not parse ir feature {response}") from e

    if is_default_irfeature(ir_feature):
        return None

    return ir_feature


def get_irorigin(control_target: Dict[str, Any]) -> str:
    origin = control_target["origin"].lower()
    if origin in ["geo", "timer"]:
        return "skynet_{}".format(origin)
    return "skynet"


def get_ir_deployment_params(
    deployment_settings: DeploymentSettings,
    control_target,
    user_id: Optional[str],
    logger,
) -> Dict[str, Any]:
    state = {
        "fan": deployment_settings.fan,
        "power": deployment_settings.power,
        "mode": deployment_settings.mode,
        "louver": deployment_settings.louver,
        "swing": deployment_settings.swing,
        "ventilation": deployment_settings.ventilation,
        "temperature": ensure_temperature_is_serialisable(
            deployment_settings.temperature, logger
        ),
        "button": deployment_settings.button,
    }

    return {
        "irorigin": get_irorigin(control_target),
        "device_id": deployment_settings.device_id,
        "state": state,
        "timestamp": int(pendulum.now("UTC").timestamp()),
        "user_id": user_id,
    }


def ensure_temperature_is_serialisable(temperature, logger) -> Optional[int]:
    if temperature is not None:
        if not isinstance(temperature, int) and not isinstance(temperature, str):
            logger("Non-int and Non-str temperature found", type=type(temperature))
            temperature = int(temperature)

    return temperature
