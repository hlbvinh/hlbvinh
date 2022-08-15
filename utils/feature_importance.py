import click
from operator import itemgetter
import pandas as pd
import yaml
from dateutil.parser import parse

from sklearn.ensemble import RandomForestClassifier
from skynet.user.store import UserSampleStore
from skynet.user.sample import COMFORT_FEATURES_WITH_TARGET
from skynet.utils.script_utils import get_connections
from skynet.utils import thermo


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mongo", default="test")
@click.option("--device_id", type=str, default="05D8FF373633594243106027")
@click.option("--user_id", type=str, default=None)
@click.option("--start", type=str, default=None)
@click.option("--end", type=str, default=None)
def main(config, mongo, device_id, user_id, start, end):
    with open(config) as f:
        cnf = yaml.safe_load(f)
    sample_store = UserSampleStore(get_connections(cnf, mongo=mongo).mongo)

    key = {"type": "user_feedback"}
    if device_id:
        key.update({"device_id": device_id})
    elif user_id:
        key.update({"user_id": user_id})
    else:
        print("Please provide either a device_id or a user_id")
        exit(0)
    if start and end:
        key.update({"timestamp": {"$gt": parse(start), "$lt": parse(end)}})

    data = pd.DataFrame(sample_store.get(key=key, limit=0))
    if data.empty:
        print("No data for the device/user.")
        exit(0)

    data["humidex_out"] = thermo.humidex(data["temperature_out"], data["humidity_out"])
    data["humidex"] = thermo.humidex(data["temperature"], data["humidity"])
    data = data[COMFORT_FEATURES_WITH_TARGET].dropna()

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    X, y = data.drop(columns=["device_id", "user_id", "feedback"]), data[["feedback"]]
    clf.fit(X, y)
    print(*sorted(zip(X.columns, clf.feature_importance_), key=itemgetter(1)))


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
