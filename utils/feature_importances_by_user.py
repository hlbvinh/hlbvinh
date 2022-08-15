import click
import pandas as pd
import yaml

from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skynet.user.store import UserSampleStore
from skynet.utils.mongo import Client
from skynet.utils.sklearn_utils import permutation_feature_importance


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mongo", default="test")
def main(config, mongo):

    cnf = yaml.safe_load(open(config))
    mongo_cnf = cnf["mongo"][mongo]
    mongo_client = Client(**mongo_cnf)
    sample_store = UserSampleStore(mongo_client)

    X = pd.DataFrame(sample_store.get())
    X = X.drop(
        [
            "type",
            "device_id",
            "timestamp",
            "toy_cos",
            "toy_sin",
            "tow_cos",
            "tow_sin",
            "pircount",
        ],
        1,
    )
    counts = X["user_id"].value_counts()
    X = X.loc[X["user_id"].isin(counts[counts > 50].index), :]
    X = X.dropna()

    # feature_groups = {k.replace('_', ' '): k for k in X.columns
    #                  if k not in ['tod_cos', 'tod_sin', 'user_id', 'feedback']}

    # feature_groups['time of day'] = ['tod_cos', 'tod_sin']
    feature_groups = {
        "time of day": ["tod_sin", "tod_cos"],
        "temperature": ["temperature"],
        "humidity": ["humidity"],
        "weather": ["temperature_out", "humidity_out"],
        "other": ["luminosity", "pirload"],
    }

    estimator = make_pipeline(SimpleImputer(), StandardScaler(), SVR())

    imps = X.groupby("user_id").apply(
        lambda x: permutation_feature_importance(
            estimator,
            x.drop(["feedback", "user_id"], 1),
            x["feedback"],
            5,
            feature_groups=feature_groups,
        )
    )

    imps = imps.sort("temperature", ascending=False)
    imps = imps[["temperature", "humidity", "time of day", "weather", "other"]]
    imps.to_csv("comfort_permutation_feature_importances_by_user.csv")


if __name__ == "__main__":
    main()
