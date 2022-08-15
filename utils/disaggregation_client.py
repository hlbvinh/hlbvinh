import asyncio
import sys
from pprint import pprint
from time import time

import click
import yaml
from ambi_utils.zmq_micro_service.msg_util import AircomRequest
from ambi_utils.zmq_micro_service.zmq_actor import DealerActor

from skynet.utils.async_util import run_sync
from skynet.utils.log_util import get_logger, init_logging_from_config

log = get_logger("skynet")


@click.command()
@click.option("--config", default="config.yml")
@click.option("--location_id", default="3bd757cd-fc9f-407a-b0fd-f1de3f5a3dea")
@click.option("--start", default="2019-09-23")
@click.option("--end", default="2019-09-24")
def main(config, location_id, start, end):

    with open(config) as f:
        cnf = yaml.safe_load(f)

    init_logging_from_config("disaggregation_client", cnf=cnf)

    msger = DealerActor(
        ip=cnf["disaggregation_service"]["ip"],
        port=cnf["disaggregation_service"]["port"],
        log=log,
    )
    req = AircomRequest.from_dict(
        {
            "session_id": None,
            "params": {"location_id": location_id, "start": start, "end": end},
            "method": "DisaggregationActor",
        }
    )

    async def request():
        tic = time()
        response = await msger.ask(req)
        log.debug("took {:.5f} s".format(time() - tic))
        pprint(response)

    try:
        run_sync(request)
    except KeyboardInterrupt:
        log.info("Detected CTRL+C")
    except Exception as e:
        log.exception(e)
    finally:
        print("Shutting down")
        # asyncio.get_event_loop().stop()

        # loop = asyncio.get_event_loop()
        # # Stop loop:

        # # Find all running tasks:
        # pending = asyncio.Task.all_tasks()

        # # Run loop until tasks done:
        # loop.run_until_complete(asyncio.gather(*pending))
        # loop.stop()

        sys.exit(0)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
