import os
import subprocess

import click
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from typing import List, Optional

from sklearn.metrics import SCORERS
from skynet.utils.log_util import get_logger
from skynet.utils.log_util import init_logging

log = get_logger("skynet")

font = {"size": 6}

matplotlib.rc("font", **font)


@click.command()
@click.option("--config", default="config.yml")
@click.option("--mae_per_column", type=click.Choice(["device_id"]), default=None)
@click.option("--months", "-m", type=int, multiple=True, default=1)
@click.option("--n_jobs", type=int, default=1)
@click.option("--samples_score_dir", default="data/samples_optimization")
@click.option(
    "--scoring", type=click.Choice(SCORERS), default="neg_mean_absolute_error"
)
def main(config, mae_per_column, months, n_jobs, scoring, samples_score_dir):

    model_type = "comfort"

    init_logging("optimize_number_of_samples")

    if len(months) < 2:
        raise ValueError(
            "Need to provide at least two months inorder to "
            "compare. Use flag -m like -m 1 -m 2 for comparison between first"
            "and second month."
        )

    dir_path = f"./{samples_score_dir}/{model_type}"

    calculate_score_for_months(
        config, model_type, months, n_jobs, scoring, samples_score_dir, mae_per_column
    )

    score = fetch_score_for_months(model_type, months, dir_path)

    avg_device_id_score = None

    if mae_per_column:
        avg_device_id_score = fetch_score_by_id_for_months(
            model_type, months, dir_path, mae_per_column
        )
    else:
        log.info(
            "Average score per user cannot be calculated for this scoring"
            "metrics. Choose mean_absolute_error for average score per user."
        )

    plot_score_with_months(score, model_type, dir_path, scoring, avg_device_id_score)


def calculate_score_for_months(
    config: str,
    model_type: str,
    months: List[int],
    n_jobs: int,
    scoring: str,
    samples_score_dir: str,
    mae_per_column: Optional[str],
) -> None:
    call = (
        f"python scripts/comfort_model.py --config {config} "
        f"--mongo test --task score --scoring {scoring} "
        f"--samples_score_dir {samples_score_dir} --model_type {model_type} "
        f"--storage file --n_jobs {n_jobs} "
    )

    for month in months:
        subprocess.run(call + f"--month_limit {month}", shell=True)

    if mae_per_column:
        call = (
            f"python scripts/comfort_model.py --config {config} "
            f"--mongo test --task score --scoring {scoring} "
            f"--samples_score_dir {samples_score_dir} --model_type {model_type} "
            f"--storage file --n_jobs {n_jobs} --mae_per_column {mae_per_column} "
        )

        for month in months:
            subprocess.run(call + f"--month_limit {month}", shell=True)


def fetch_score_for_months(
    model_type: str, months: List[int], dir_path: str
) -> pd.DataFrame:
    all_files = os.listdir(dir_path)

    filenames = [
        filename
        for filename in all_files
        if f"{model_type}_model_mae_scores_for_sample_number" in filename
    ]

    if not filenames:
        log.info("No files to process. Exit.")
        exit(0)

    dataframes = [pd.read_pickle(f"{dir_path}/{filename}") for filename in filenames]

    score = pd.concat(dataframes, axis=0)

    score = process_score_for_months(score, months)

    return score


def process_score_for_months(score: pd.DataFrame, months: List[int]) -> pd.DataFrame:
    score["sample_limit"] = score["sample_limit"].apply(lambda x: int(x))
    score = score.sort_values(by="sample_limit")
    score = score.loc["mean", :]
    score["month"] = sorted(months)

    return score


def fetch_score_by_id_for_months(
    model_type: str, months: List[int], dir_path: str, mae_per_column: str
) -> pd.DataFrame:

    mae_per_column_for_each_sample_len = fetch_mae_score_per_column_for_each_sample_len(
        dir_path, model_type, mae_per_column=mae_per_column
    )

    train_mean_scores, test_mean_scores, sample_limit = [], [], []

    for i in range(0, len(mae_per_column_for_each_sample_len)):
        train_mean_scores.append(
            mae_per_column_for_each_sample_len[i].loc["train", "mae"]
        )
        test_mean_scores.append(
            mae_per_column_for_each_sample_len[i].loc["test", "mae"]
        )
        sample_limit.append(
            mae_per_column_for_each_sample_len[i].loc["test", "sample_limit"]
        )

    score = pd.DataFrame(
        {
            "test_score": test_mean_scores,
            "train_score": train_mean_scores,
            "sample_limit": sample_limit,
        }
    )

    score = score.sort_values("sample_limit")
    score["month"] = sorted(months)
    return score


def fetch_mae_score_per_column_for_each_sample_len(
    dir_path: str, model_type: str, mae_per_column: str = "device_id"
) -> List[List]:
    all_files = os.listdir(dir_path)

    mae_per_column_filenames = [
        filename
        for filename in all_files
        if f"{model_type}_model_mae_score_per_{mae_per_column}_for_sample_number_" in filename
    ]

    if not mae_per_column_filenames:
        log.info("No files found.")
        exit(0)

    dataframes_for_sample_lens = [
        pd.read_pickle(f"{dir_path}/{file}") for file in mae_per_column_filenames
    ]

    return dataframes_for_sample_lens


def plot_score_with_months(
    score: pd.DataFrame,
    model_type: str,
    dir_path: str,
    scoring: str,
    avg_score: Optional[pd.DataFrame] = None,
) -> None:

    if avg_score is not None:
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        ax1 = create_plot(ax1, score, model_type, scoring)
        ax2 = create_plot(ax2, avg_score, model_type, "average mae of devices")
    else:
        ax1 = plt.subplot(111)
        ax1 = create_plot(ax1, score, model_type, scoring)

    plt.savefig(dir_path + f"/{model_type}_model_{scoring}_months.png")
    log.info(f"Figure saved to {dir_path}/{model_type}_model_{scoring}_months.png")
    plt.show()


def create_plot(ax1, score: pd.DataFrame, model_type: str, scoring: str):
    xticklabels = score[["month", "sample_limit"]].apply(
        lambda x: str(x["month"]) + ", " + str(x["sample_limit"]), axis=1
    )

    ax1.scatter(
        score.loc[:, "month"].tolist(),
        score.loc[:, "train_score"].tolist(),
        label="Train",
    )
    ax1.scatter(
        score.loc[:, "month"].tolist(),
        score.loc[:, "test_score"].tolist(),
        label="Test",
    )

    ax1.set_title(f"{model_type} model score({scoring}) for previous months samples")
    ax1.set_ylabel(f"{scoring}")
    ax1.set_xlabel("Month, Number of samples")
    ax1.set_xticks(score.loc[:, "month"].tolist())
    ax1.set_xticklabels(xticklabels)
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
    return ax1


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
