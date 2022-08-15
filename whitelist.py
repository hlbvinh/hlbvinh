import logging
import threading
from sqlite3 import Connection
from unittest import mock

from cassandra.cluster import Cluster
from dash import Dash
from pandas import DataFrame
from skynet.tests.test_control import MockEventA, MockEventB
from skynet.tests.test_control_util import TestComfort
from skynet.utils.nnet_climate_utils import (
    EmbeddingMLPRegressor as EmbeddingMLPRegressorClimate,
)

from scripts.__main__ import CLI
from scripts.status_monitor import ServiceHandler
from skynet.conftest import pytest_configure
from skynet.control.controller import Controller
from skynet.prediction.mode_filtering import (
    prevent_bad_heat_cool_selection,
    prevent_dry_mode_from_cooling_too_much,
    prevent_fan_mode_when_too_far_from_target,
    prevent_using_auto_mode,
)
from skynet.prediction.prediction_service import PredictionService
from skynet.tests.conftest import event_loop, filestorage, s3_model_store
from skynet.tests.test_mysql import db_with_data
from skynet.user.analytics_service import UserAnalyticsActor
from skynet.user.comfort_service import ComfortModelActor
from skynet.user.sample import NON_FEEDBACK_FEATURES
from skynet.utils import cache_util, compensation
from skynet.utils.async_util import multi_dict, multi_generator, multi_list
from skynet.utils.database.cassandra import CassandraSession
from skynet.utils.database.queries import (
    get_appliance_states_from_device,
    get_device_online_intervals,
    insert_ac_event_trigger,
)
from skynet.utils.events import (
    AutomatedDemandResponseEvent,
    ControlEvent,
    FeedbackEvent,
    IREvent,
    ModeFeedbackEvent,
    ModePreferenceEvent,
    NearbyUserEvent,
    SensorEvent,
    StateEvent,
)
from skynet.utils.ir_feature import APPLIANCE_PROPERTIES
from skynet.utils.nnet_utils import EmbeddingMLPRegressor
from skynet.utils.sample_store import SampleStore
from skynet.utils.sklearn_utils import BaseSeriesFold
from skynet.utils.status import StatusActor
from skynet.utils.storage import S3Storage
from skynet.utils.testing import gen_feature_matrix
from skynet.utils.types import ModePrefKey
from utils.comfort_model_offline_metrics import (
    compute_absolute_mae,
    compute_hour_of_day,
    plot_mae,
    plot_mae_per_feedback_type,
    plot_mae_per_hour_of_day,
)
from utils.micro_model_offline_metrics import (
    analysis_on_modes,
    analysis_on_number_of_samples,
    calculate_max_step_size,
    calculate_mean_range,
    calculate_mean_step_size,
    calculate_min_step_size,
    mae_score,
    number_of_inversion,
    set_temperature_variation_for_change_in_target_delta,
)
from utils.plotly_climate_model_baseline import update_graph as baseline_update_graph
from utils.plotly_climate_model_sensor_samples import (
    update_graph as sensor_samples_update_graph,
)

StatusActor.do_tell
UserAnalyticsActor.do_tell
ComfortModelActor.do_tell
StatusActor.do_tell


CLI.list_commands
CLI.get_command

Cluster.connection_class

logging.Logger.propagate

threading.Thread.daemon

TestComfort

pytest_configure

filestorage
s3_model_store

PredictionService.setup_resources

Connection.row_factory

Cluster.connection_class

Controller.get_away_mode_settings

CassandraSession.get_config

insert_ac_event_trigger
get_appliance_states_from_device
get_device_online_intervals

SampleStore.clear_all

NON_FEEDBACK_FEATURES

gen_feature_matrix

APPLIANCE_PROPERTIES

BaseSeriesFold.get_n_splits

DataFrame.index.names

ServiceHandler.initialize

ModePrefKey.__new__.__defaults__

mock.Mock.level

# this one is just used for tests at the moment
cache_util.get_invalid_sensors_count
S3Storage

# singledispatch, naming them all "_" leads to duplicate name error with mypy
compensation._compensate_features_df
compensation._compensate_features_dicts
compensation._compensate_sensors_df
compensation._compensate_sensors_dicts

db_with_data

SensorEvent, StateEvent, ControlEvent, FeedbackEvent, IREvent, ModePreferenceEvent, MockEventA, MockEventB, ModeFeedbackEvent, NearbyUserEvent, AutomatedDemandResponseEvent, DKVentilationOptionEvent

# for plotly scripts
baseline_update_graph
sensor_samples_update_graph
Dash.layout

EmbeddingMLPRegressorClimate.forward
EmbeddingMLPRegressorClimate.random_state
EmbeddingMLPRegressor.forward
EmbeddingMLPRegressor.random_state

event_loop

multi_dict, multi_generator, multi_list

set_temperature_variation_for_change_in_target_delta,
mae_score,
number_of_inversion,
calculate_mean_range,
calculate_mean_step_size,
calculate_max_step_size,
calculate_min_step_size,
analysis_on_number_of_samples,
analysis_on_modes

compute_absolute_mae,
compute_hour_of_day,
plot_mae,
plot_mae_per_feedback_type,
plot_mae_per_hour_of_day,

prevent_bad_heat_cool_selection
prevent_dry_mode_from_cooling_too_much
prevent_fan_mode_when_too_far_from_target
prevent_using_auto_mode
