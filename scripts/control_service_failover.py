import asyncio
import json
import logging
import socket
import subprocess

import click
import requests
import uvloop
import yaml
from aredis import StrictRedis as Redis
from aredis.exceptions import RedisError

from skynet.utils.async_util import run_in_executor
from skynet.utils.log_util import get_logger, init_logging_from_config
from skynet.utils.redis_util import RenewLock, get_redis
from skynet.utils.types import Config

log = get_logger("skynet")

CHECKING_FREQUENCY_SECONDS = 1
LOCK_EXPIRY_SECONDS = 2
EXTENDING_LOCK_FREQUENCY_SECONDS = LOCK_EXPIRY_SECONDS / 2


@click.command()
@click.option("--config", default="config.yml")
def main(config):
    uvloop.install()

    with open(config) as f:
        cnf = yaml.safe_load(f)
    init_logging_from_config("control_service_failover", cnf=cnf, loglevel=logging.INFO)

    asyncio.get_event_loop().run_until_complete(control_service_failover(cnf))


async def control_service_failover(cnf: Config) -> None:
    redis = get_redis(cnf)
    lock = create_lock(redis)

    while True:
        if await can_get_lock(lock):
            await keep_control_service_running(cnf, lock)
        stop_control_service(cnf)
        await asyncio.sleep(CHECKING_FREQUENCY_SECONDS)
        log.info("trying to acquire the lock")


def create_lock(redis: Redis) -> RenewLock:
    return RenewLock(redis, "skynet:lock:control_service", timeout=LOCK_EXPIRY_SECONDS)


async def can_get_lock(lock: RenewLock) -> bool:
    try:
        return await lock.acquire(blocking_timeout=0)
    except RedisError as e:
        log.info(f"cannot get lock: {e}")
        return False


async def keep_control_service_running(cnf: Config, lock: RenewLock) -> None:
    while True:
        start_control_service(cnf)
        await asyncio.sleep(EXTENDING_LOCK_FREQUENCY_SECONDS)
        if not await is_lock_renewed(lock):
            break
        log.info("succesfully renewed lock")


@run_in_executor
def start_control_service(cnf: Config) -> None:
    hostname = socket.gethostname()

    subject = f"{hostname} now providing control"
    msg = ""

    if is_control_service_stopped(cnf):
        send_message(cnf, subject, msg)
        run_supervisorctl("start", "control")


@run_in_executor
def stop_control_service(cnf: Config) -> None:
    hostname = socket.gethostname()

    subject = f"{hostname} is down/lost connectivity, stopping control"
    msg = ""
    if not is_control_service_stopped(cnf):
        send_message(cnf, subject, msg)
        run_supervisorctl("stop", "control")


def is_control_service_stopped(cnf: Config) -> bool:
    stdout = run_supervisorctl("status", "control")
    if all(state not in stdout for state in ["RUNNING", "STARTING", "STOPPED"]):
        log.error(f"{stdout}")
        hostname = socket.gethostname()
        subject = f"{hostname} supervisor status error"
        msg = f"{stdout}"
        send_message(cnf, subject, msg)
        return True
    return "STOPPED" in stdout


def run_supervisorctl(action: str, service: str) -> str:
    result = subprocess.run(
        ["supervisorctl", action, service], stdout=subprocess.PIPE, check=True
    )
    stdout = result.stdout.decode()
    log.info(f"{stdout}")
    return stdout


async def is_lock_renewed(lock: RenewLock) -> bool:
    try:
        return await lock.renew()
    except RedisError as e:
        log.info(f"cannot renew lock: {e}")
        return False


def send_message(cnf: Config, subject: str, msg: str) -> None:
    url = cnf["slack"]["webhook_url"]
    headers = {"content-type": "application/json"}
    payload = dict(attachments=[dict(color="warning", title=subject, text=msg)])
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    log.info(f"sent {msg}, response {response}")


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
