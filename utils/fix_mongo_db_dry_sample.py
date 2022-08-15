import click
import yaml
from tqdm import tqdm

from skynet.sample.climate_sample_store import ClimateSampleStore
from skynet.utils.async_util import run_sync, multi
from skynet.utils.database.dbconnection import get_pool
from skynet.utils.mongo import Client


def query_temperature_set(appliance_state_id: int):
    q = """
    SELECT
        temperature AS temperature_set
    FROM
        ApplianceState
    WHERE
        appliance_state_id = %s
    """
    return q, (appliance_state_id,)


async def get_temperature_set(pool, appliance_state_ids):
    coroutines = [
        pool.execute(*query_temperature_set(appliance_state_id))
        for appliance_state_id in appliance_state_ids
    ]
    data = await multi(coroutines)
    return data


@click.command()
@click.option("--config", default="config.yml")
@click.option("--device_id", default="05DDFF393131573257228419")
@click.option("--mysql", default="viewer")
@click.option("--mongo", default="test")
def main(config, device_id, mysql, mongo):

    cnf = yaml.safe_load(open(config))
    mongo_client = Client(**cnf["mongo"][mongo])
    sample_store = ClimateSampleStore(mongo_client)
    db_cnf = cnf[mysql]
    pool = get_pool(**db_cnf)

    col = mongo_client._col(sample_store.sample_collection)
    bulk = col.initialize_ordered_bulk_op()
    counter = 0

    rs = []
    cursor = col.find({"mode": "dry"}, modifiers={"$snapshot": True})
    for record in tqdm(cursor, total=cursor.count()):
        rs.append(record)
        if len(rs) == 10:
            temps = run_sync(
                get_temperature_set, pool, [r["appliance_state_id"] for r in rs]
            )
            for r, temp in zip(rs, temps):
                if temp:
                    bulk.find({"_id": record["_id"]}).update(
                        {"$set": {"temperature_set": temp[0]["temperature_set"]}}
                    )
                    counter += 1
            rs = []

            if counter and (counter % 1000 == 0):
                bulk.execute()
                bulk = col.initialize_ordered_bulk_op()

    if counter != 0:
        temps = run_sync(
            get_temperature_set, pool, [r["appliance_state_id"] for r in rs]
        )
        for r, temp in zip(rs, temps):
            if temp:
                bulk.find({"_id": record["_id"]}).update(
                    {"$set": {"temperature_set": temp[0]["temperature_set"]}}
                )
            bulk.find({"_id": record["_id"]}).update(
                {"$set": {"temperature_set": temp}}
            )
        bulk.execute()


if __name__ == "__main__":
    main()
