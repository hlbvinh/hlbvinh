import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

from skynet.sample import climate_sample_store
from skynet.utils.log_util import get_logger, init_logging
from skynet.utils.mongo import Client

log = get_logger("skynet")
FIG_DIR = "./fig"


@click.command()
@click.option("--config", default="config.yml")
@click.option(
    "--device_id", default="6409FF383939473443067324", help="Device ID for predictions."
)
@click.option("--mongo", default="test")
def main(config, device_id, mongo):
    """Computes any number of useful things"""

    cnf = yaml.safe_load(open(config))
    mongo_cnf = cnf["mongo"][mongo]
    mongo_client = Client(**mongo_cnf)
    sample_store = climate_sample_store.ClimateSampleStore(mongo_client)
    init_logging("micro_model")

    def fetch():
        return sample_store.get(key={"mode": "cool", "device_id": device_id})

    X = fetch()
    samples = [sample for sample in X if len(sample["target"]) > 30]
    n = len(samples)
    nx = int(n / np.sqrt(n) + 1)
    ny = int(n / (nx - 1))
    print(nx, ny)
    _, axs = plt.subplots(nx, ny)
    axs_flat = [ax for row in axs for ax in row]

    for ax in axs_flat:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for sample, ax in zip(samples, axs_flat):
        print(sample)
        x, y = zip(*[(y["timestamp"], y["temperature"]) for y in sample["target"]])
        x = np.array((sample["timestamp"],) + x)
        y = np.array((sample["temperature"],) + y)
        y -= y[0]
        print(x)
        print(y)
        ax.plot(x, y)
        ax.set_ylim((-5, 5))

    plt.show()


if __name__ == "__main__":
    main()
