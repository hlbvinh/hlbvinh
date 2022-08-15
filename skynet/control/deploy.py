from datetime import timedelta

from . import util
from ..utils import monitor
from ..utils.async_util import request_with_timeout
from ..utils.db_service import db_service_util
from ..utils.enums import Power
from ..utils.log_util import get_logger
from ..utils.monitor import Monitor
from ..utils.types import (
    AircomResponse,
    ApplianceState,
    BasicDeployment,
    Connections,
    ControlTarget,
    Feedback,
    IRFeature,
)

log = get_logger(__name__)

DEPLOYMENT_TIMEOUT = timedelta(seconds=5)


class Deploy(util.LogMixin):
    def __init__(
        self,
        connections: Connections,
        device_id: str,
        appliance_id: str,
        state: ApplianceState,
        ir_feature: IRFeature,
        control_target: ControlTarget,
        latest_feedback: Feedback,
        testing: bool,
        monitor: Monitor,
    ):
        self.connections = connections
        self.device_id = device_id
        self.appliance_id = appliance_id
        self.state = state
        self.ir_feature = ir_feature
        self.control_target = control_target
        self.latest_feedback = latest_feedback
        self.testing = testing
        self.monitor = monitor

    async def deploy_settings(self, settings: BasicDeployment) -> AircomResponse:
        aircom_response = await self.post_signal(settings)
        self.signal_post_callback(aircom_response)
        return aircom_response

    async def post_signal(self, settings: BasicDeployment) -> AircomResponse:
        # adjust (polish and make sure it is deployable) deployment setting
        deployment_settings = await util.adjust_deployment_settings(
            self.connections,
            settings,
            self.device_id,
            self.appliance_id,
            self.state,
            self.ir_feature,
        )

        # convert to suitable format for db_service
        aircom_params = db_service_util.get_ir_deployment_params(
            deployment_settings=deployment_settings,
            control_target=self.control_target,
            user_id=self.latest_feedback.get("user_id"),
            logger=self.log,
        )
        self.log(
            "deploying",
            deployment=aircom_params,
            testing=self.testing,
            control_target=self.control_target["quantity"],
        )

        # send deployment request
        if not self.testing:
            aircom_request = db_service_util.get_aircom_request(
                "IrDeploymentCreate", aircom_params
            )
            return await request_with_timeout(
                DEPLOYMENT_TIMEOUT, self.connections.db_service_msger, aircom_request
            )
        return 200, {}

    @monitor.monitored("deployments")
    def signal_post_callback(self, aircom_response: AircomResponse) -> None:
        """For deployment processes monitoring.

        Args:
            aircom_response:

        Returns:

        """
        prefix = "WOULD HAVE " if self.testing else ""
        code, response = aircom_response
        if code in [200, 201]:
            self.log(msg=f"{prefix}signal posted {code}")
        else:
            self.log(f"{prefix}no signal posted {code}, {response.get('reason')}")

    @property
    def off_deployment(self) -> BasicDeployment:
        return BasicDeployment(
            power=Power.OFF,
            mode=self.current_mode,
            temperature=self.state["temperature"],
            ventilation=None,
        )

    @property
    def current_mode(self) -> str:
        return self.state["mode"]
