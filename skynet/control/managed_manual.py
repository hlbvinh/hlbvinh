import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt

from ..utils.database.queries import query_appliance_states_from_device
from ..utils.enums import Power
from ..utils.parse import lower_dict
from ..utils.types import ApplianceState, BasicDeployment, Connections, IRFeature
from .util import LogMixin

AVERAGE_PERIOD = timedelta(weeks=2)
MAX_STATE_DURATION = timedelta(days=1)
LOW_AC_USAGE = timedelta(hours=4)

# Modes are restricted to this list because their set points
# are assumed to always be numbers and can be averaged
SUPPORTED_MODES = ["cool", "heat"]
DEFAULT_SET_TEMPERATURE = dict(cool="24", heat="21")


class ManagedManual(LogMixin):
    def __init__(self, device_id: str, state: ApplianceState, connections: Connections):
        self.device_id = device_id
        self.state = state
        self.connections = connections

    async def get_deployment(
        self, target: float, ir_feature: IRFeature
    ) -> Optional[BasicDeployment]:
        if self.state["power"] == Power.OFF:
            return None
        try:
            return await self._get_deployment(target, ir_feature)
        except ValueError:
            return None

    async def _get_deployment(
        self, target: float, ir_feature: IRFeature
    ) -> BasicDeployment:
        precomputation = await self._fetch_precomputation()
        setting = dict(
            power=Power.ON,
            mode=precomputation["mode"],
            temperature=self._determine_set_temperature(
                target, ir_feature, precomputation
            ),
        )
        self.log_(precomputation, setting)
        return BasicDeployment(**setting)

    async def get_mode(self) -> str:
        return (await self._fetch_precomputation())["mode"]

    async def get_target_value(self) -> float:
        return (await self._fetch_precomputation())["target"]

    async def _fetch_precomputation(self):
        # TODO if needed cache _precomputation result in redis for a day to
        # prevent fetching states and computing it every time
        return self._precomputation(await self._fetch_states())

    @retry(retry=retry_if_exception(asyncio.TimeoutError), stop=stop_after_attempt(3))
    async def _fetch_states(self) -> List[ApplianceState]:
        end = datetime.utcnow()
        start = end - AVERAGE_PERIOD
        states = await self.connections.pool.execute(
            *query_appliance_states_from_device(self.device_id, start=start, end=end)
        )
        return [self._parse_state(state) for state in states]

    def _precomputation(self, states: List[ApplianceState]) -> dict:
        """Selects mode, target value based on pass two weeks data.

        Most used mode is selected, weighted set temperature of that mode is selected.

        Args:
            states:

        Returns:

        """
        if len(states) < 2:
            self.log("managed_manual Not enough appliance states", level="error")
            raise ValueError("Not enough appliance states")
        df = pd.DataFrame(states).sort_values(by="created_on")
        df["duration"] = df.created_on.shift(-1) - df.created_on
        df.dropna(inplace=True, subset=["duration"])
        df = df[
            (df.power == "on")
            & (df.duration < MAX_STATE_DURATION)
            & (df["mode"].isin(SUPPORTED_MODES))
        ]
        if df.empty:
            self.log("managed_manual empty dataframe", level="error")
            raise ValueError("Not enough valid appliance states")

        most_used_mode = df.groupby("mode").duration.sum().idxmax()
        most_used_mode_duration = df[df["mode"] == most_used_mode].duration.sum()
        mean_temperature_set = self._weighted_average(
            df[df["mode"] == most_used_mode].temperature_set,
            df[df["mode"] == most_used_mode].duration,
        )

        return dict(
            mode=most_used_mode,
            target=mean_temperature_set,
            duration=most_used_mode_duration,
        )

    def _determine_set_temperature(
        self, target: float, ir_feature: IRFeature, precomputation: dict
    ) -> str:
        if precomputation["duration"] < LOW_AC_USAGE:
            return DEFAULT_SET_TEMPERATURE[precomputation["mode"]]
        return self._closest_set_point_to_target(
            target, ir_feature, precomputation["mode"]
        )

    @staticmethod
    def _closest_set_point_to_target(
        target: float, ir_feature: IRFeature, mode: str
    ) -> str:
        return min(
            ir_feature[mode]["temperature"]["value"],
            key=lambda t: abs(float(t) - target),
        )

    @staticmethod
    def _weighted_average(values, weights) -> float:
        return np.average(values, weights=list(weights))

    @staticmethod
    def _parse_state(state):
        state = state.copy()
        if state["mode"] in SUPPORTED_MODES:
            state["temperature_set"] = float(state["temperature_set"])
        return lower_dict(state)

    def log_(self, precomputation, setting):
        self.log("managed_manual", precomputation=precomputation, setting=setting)
