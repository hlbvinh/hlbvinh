import pytest
from asynctest import mock

from ...control import deploy
from ...utils.types import BasicDeployment, Connections


@pytest.fixture
def default_db_service_msger(get_db_service_msger):
    return get_db_service_msger(code=200, response={})


@pytest.fixture
def connections(pool, rediscon, cassandra_session, default_db_service_msger):
    return Connections(
        pool=pool,
        redis=rediscon,
        session=cassandra_session,
        db_service_msger=default_db_service_msger,
    )


@pytest.fixture
def default_deploy(
    connections, device_id, appliance_id, state, ir_feature, target, feedback
):
    return deploy.Deploy(
        connections,
        device_id,
        appliance_id,
        state,
        ir_feature,
        target,
        feedback,
        testing=True,
        monitor=mock.MagicMock(),
    )


@pytest.fixture
def db_service_msger_with_error(get_db_service_msger):
    return get_db_service_msger(
        code=500, response={"reason": "db service not available"}
    )


@pytest.fixture
def connections_with_db_service_error(
    pool, rediscon, cassandra_session, db_service_msger_with_error
):
    return Connections(
        pool=pool,
        redis=rediscon,
        session=cassandra_session,
        db_service_msger=db_service_msger_with_error,
    )


@pytest.fixture
def deploy_with_db_service_error(default_deploy, connections_with_db_service_error):
    default_deploy.connections = connections_with_db_service_error
    return default_deploy


@pytest.fixture(params=[True, False])
def is_testing_controller(request):
    return request.param


@pytest.mark.asyncio
async def test_deploy_settings_success(default_deploy, is_testing_controller):
    default_deploy.testing = is_testing_controller
    code, result = await default_deploy.deploy_settings(BasicDeployment())
    assert code == 200
    assert result == {}


@pytest.mark.asyncio
async def test_deploy_settings_with_db_service_error(deploy_with_db_service_error):
    deploy = deploy_with_db_service_error
    deploy.testing = False
    code, result = await deploy.deploy_settings(BasicDeployment())
    assert code == 500
    assert result == {"reason": "db service not available"}


@pytest.fixture
def deployment_setting():
    return BasicDeployment(mode="cool")


@pytest.fixture
def fan_setting_deploy(default_deploy, deployment_setting):
    default_deploy.state["mode"] = "heat"
    assert deployment_setting.mode != default_deploy.current_mode
    return default_deploy


@pytest.mark.asyncio
async def test_regression_post_signal_fan_setting(
    rediscon, fan_setting_deploy, deployment_setting
):
    # make sure that we fetch the fan setting of the mode to be deployed and
    # not the current mode
    with mock.patch("skynet.control.util.get_fan_setting") as m:
        await fan_setting_deploy.deploy_settings(deployment_setting)
        call = mock.call(
            rediscon,
            fan_setting_deploy.appliance_id,
            deployment_setting.mode,
            fan_setting_deploy.ir_feature,
        )
        assert m.call_args == call
