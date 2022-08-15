import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from datetime import datetime
from dateutil.relativedelta import relativedelta
from skynet import utils
from skynet.prediction import climate_model
from skynet.sample import climate_sample_store
from skynet.sample.climate_sample_store import ClimateSampleStore

cnf = yaml.safe_load(open("config.yml"))
db_cnf = cnf["local"]
mongo_cnf = cnf["mongo"]["test"]
mongo_client = utils.mongo.Client(**mongo_cnf)
sample_store = ClimateSampleStore(mongo_client)

N_MONTHS = 3

device_id = "05D6FF303830594143186915"

key = {"timestamp": {"$gt": datetime.utcnow() - relativedelta(days=N_MONTHS)}}

feats, targs = climate_sample_store.get_climate_samples(sample_store, key=key)
Xs, ys = climate_model.make_static_climate_dataset(feats, targs)

idx = np.where(Xs.device_id == "05D6FF303830594143186915")[0]

xh = Xs.iloc[idx].reset_index(drop=True)
yh = ys.iloc[idx].reset_index(drop=True)
yh.columns = ["target_{}".format(c) for c in yh.columns]

data = pd.concat(
    [pd.concat([xh, yh], axis=1)._get_numeric_data(), xh[["mode"]]], axis=1
)

plot_columns = [
    "humidex_out",
    "humidex",
    "humidity",
    "temperature",
    "temperature_set",
    "target_temperature",
    "mode",
]


sns.pairplot(data[plot_columns], hue="mode", palette="husl")

plt.savefig("dist.pdf")
