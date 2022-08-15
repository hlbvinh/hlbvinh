import click
import yaml

from skynet.utils.mongo import Client
from skynet.sample.climate_sample_store import ClimateSampleStore
from skynet.utils import thermo


@click.command()
@click.option(
    "--task",
    required=True,
    default="run",
    type=click.Choice(["run", "clear", "set_watermarks"]),
)
@click.option("--config", default="config.yml")
@click.option("--mongo", default="production")
def main(task, config, mongo):

    cnf = yaml.safe_load(open(config))
    mongo_client = Client(**cnf["mongo"][mongo])
    sample_store = ClimateSampleStore(mongo_client)

    col = mongo_client._col(sample_store.sample_collection)
    bulk = col.initialize_ordered_bulk_op()
    counter = 0

    for record in col.find(modifiers={"$snapshot": True}):
        humidex_out = thermo.humidex(record["temperature_out"], record["humidity_out"])

        bulk.find({"_id": record["_id"]}).update({"$set": {"humidex_out": humidex_out}})
        counter += 1

        if counter % 1000 == 0:
            bulk.execute()
            bulk = col.initialize_ordered_bulk_op()
            print(counter, "done")

    if counter != 0:
        bulk.execute()


if __name__ == "__main__":
    main()
