import sys
import os
import shutil

from scripts.micro_models import OVERALL_SCORE_FILENAME, PER_COLUMN_SCORE_FILENAME

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import List
import click
from skynet.utils.log_util import get_logger, init_logging

TARGETS = ["temperature", "humidex", "humidity"]
log = get_logger("skynet")


@click.command()
@click.option("--dir_path", default="./data/climate")
def main(dir_path):
    init_logging("climate_model_score_per_mode")

    try:
        calculate_and_plot_mae_score_per_mode(dir_path)
    except OSError:
        print("Wrong directory. Check if the directory and files exist.")


def calculate_and_plot_mae_score_per_mode(dir_path):

    if not directory_and_files_exists(dir_path):
        log.info(f"{dir_path} or files doesn't exist")
        raise OSError
    else:
        check_and_create_necessary_sub_directories(dir_path)

    filenames_in_the_directory = os.listdir(f"{dir_path}")

    plot_overall_mae_score(dir_path, filenames_in_the_directory)

    mae = get_mae_score_per_target_per_column(dir_path, filenames_in_the_directory)

    plot_mae_for_all_possible_values_in_column_for_all_target(mae, dir_path)


def directory_and_files_exists(dir_path):
    if os.path.exists(f"{dir_path}"):
        filenames = os.listdir(f"{dir_path}")
        valid_filenames = [
            filename
            for filename in filenames
            if filename_starts_with("climate_model_mae_", filename)
        ]
        return bool(valid_filenames)
    else:
        return False


def plot_overall_mae_score(dir_path, filenames):

    try:
        scores = []

        for filename in filenames:
            if filename_starts_with(OVERALL_SCORE_FILENAME, filename):
                number_of_samples_per_mode = int(filename.split("_")[-1][:-4])
                score_dataframe = pd.read_csv(f"{dir_path}/{filename}")
                score_dataframe["sample_number"] = number_of_samples_per_mode
                scores.append(score_dataframe)

        if not scores:
            raise OSError(f"No files for overall score. Passing.")

        scores = pd.concat(scores).sort_values("sample_number")

        test_scores = scores.loc[scores["set"] == "test"]
        train_scores = scores.loc[scores["set"] == "train"]

        for target in TARGETS:
            plt.clf()
            plt.scatter(test_scores["sample_number"], test_scores[target], label="test")
            plt.scatter(
                train_scores["sample_number"], train_scores[target], label="train"
            )
            plt.title(f"Number of months vs MAE for {target}")
            plt.xlabel(f"Number of months per mode")
            plt.ylabel("Negative MAE")
            plt.legend()
            if os.path.exists(f"{dir_path}/climate_model_figures/general/"):
                plt.savefig(
                    f"{dir_path}/climate_model_figures/general/number_of_months(per_mode)_vs_MAE_of_{target}.png"
                )
                log.info(
                    f"Plot for overall mae score saved in /climate_model_figures/general/"
                )
            else:
                raise OSError(
                    f"Plots for overall MAE score vs Number of samples saved in {dir_path}/"
                    f"climate_model_figures/general."
                )

    except OSError as err:
        log.info(err.args)
        pass


def get_mae_score_per_target_per_column(
    dir_path: str, filenames: List[str]
) -> pd.DataFrame:

    try:
        valid_dataframes = []
        for filename in filenames:
            if filename_starts_with(PER_COLUMN_SCORE_FILENAME, filename):
                number_of_samples = int(filename.split("_")[-1][:-4])
                dataframe = pd.read_csv(f"{dir_path}/{filename}")
                dataframe = process_dataframe(dataframe, number_of_samples)
                valid_dataframes.append(dataframe)

        if not valid_dataframes:
            raise OSError("No files for climate model mae score per.")

        return pd.concat(valid_dataframes)

    except OSError as err:
        log.info(err.args)
        sys.exit(1)


def process_dataframe(score_df: pd.DataFrame, number_of_samples: int) -> pd.DataFrame:
    trans_score_df = score_df.set_index("set").transpose().reset_index()
    trans_score_df["target"] = trans_score_df["index"].apply(lambda x: x.split("--")[0])
    trans_score_df["mode"] = trans_score_df["index"].apply(lambda x: x.split("--")[-1])
    trans_score_df["total_samples"] = number_of_samples
    trans_score_df = trans_score_df[
        ["target", "mode", "train", "test", "total_samples"]
    ]
    melt_df = pd.melt(
        trans_score_df,
        id_vars=["target", "mode", "total_samples"],
        value_vars=["train", "test"],
    )
    pivot_df = melt_df.pivot_table(
        index=["mode", "total_samples", "set"], columns="target", values="value"
    ).reset_index()

    return pivot_df.sort_values("total_samples")


def plot_mae_for_all_possible_values_in_column_for_all_target(
    data_train_test: pd.DataFrame, dir_path: str
) -> None:
    log.info("Dropping rows with NAN value for train or test to plot it successfully.")
    plotable_dataframe = data_train_test.dropna(
        subset=["humidex", "humidity", "temperature"]
    )
    log.info(f"Using {len(plotable_dataframe)/len(data_train_test)*100}% of rows")
    for mode in plotable_dataframe["mode"].unique():
        plot_mae_per_mode_for_mode(plotable_dataframe, dir_path, mode=mode)
        for target in ["temperature", "humidex", "humidity"]:
            plot_mae_per_mode_for_target_separately(
                plotable_dataframe, dir_path, mode=mode, target=target
            )


def plot_mae_per_mode_for_target_separately(data, dir_path, mode=None, target=None):
    try:
        valid_data = data.loc[data["mode"] == mode].sort_values("total_samples")
        plt.clf()
        title, xlabel, ylabel, figurename = get_figure_details(
            mode, dir_path, target, separate_target=True
        )

        plt.scatter(
            valid_data["total_samples"], valid_data[target], alpha=0.5, label=target
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(figurename)

        log.info(
            f"Plot for overall mae for {mode} mode for {target} saved in /climate_model_figures/general/"
        )

    except OSError as err:
        log.info(
            f"No folder exist with name climate_model_figure/general/. Error message: {err}"
        )


def plot_mae_per_mode_for_mode(data, dir_path, mode=None):
    try:
        valid_data = data.loc[data["mode"] == mode]
        plt.clf()
        title, xlabel, ylabel, figurename = get_figure_details(
            mode, dir_path, target=None, separate_target=False
        )

        plt.scatter(
            valid_data["total_samples"],
            valid_data["humidity"],
            alpha=0.5,
            label="humidty",
        )
        plt.scatter(
            valid_data["total_samples"],
            valid_data["humidex"],
            alpha=0.5,
            label="humidex",
        )
        plt.scatter(
            valid_data["total_samples"],
            valid_data["temperature"],
            alpha=0.5,
            label="temperature",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(figurename)

        log.info(
            f"Plot for mae per mode for {mode} mode saved in /climate_model_figures/general/"
        )

    except OSError as err:
        log.info(
            f"No folder exist with name climate_model_figures/general/. Error message: {err}"
        )


def check_and_create_necessary_sub_directories(dir_path):

    if sub_directories_already_exist(dir_path):
        print("removing sub directories.")
        remove_sub_directories(dir_path)

    os.makedirs(f"{dir_path}/climate_model_figures")

    os.makedirs(f"{dir_path}/climate_model_figures/general")


def sub_directories_already_exist(dir_path):
    return os.path.exists(f"{dir_path}/climate_model_figures")


def remove_sub_directories(dir_path):
    if os.path.exists(f"{dir_path}/climate_model_figures"):
        shutil.rmtree(f"{dir_path}/climate_model_figures")
        log.info(
            f"Removing subdirectory climate_model_figures in path {dir_path} to "
            f"avoid conflicts with newly generated files."
        )


def filename_starts_with(string, filename):
    return re.match(string, filename)


def get_figure_details(mode, dir_path, target=None, separate_target=False):
    if separate_target:
        title = f"Mean MAE of {target} vs months for {mode} for train-test"
        xlabel = "Number of months"
        ylabel = f"Mean MAE {target} for {mode}"
        figname = f"{dir_path}/climate_model_figures/general/MAE_for_{mode}_vs_months_for_{target}_train-test.png"
    else:
        title = f"Mean MAE of vs months for {mode} for train-test"
        xlabel = "Number of months"
        ylabel = f"Mean MAE for {mode}"
        figname = f"{dir_path}/climate_model_figures/general/MAE_for_{mode}_vs_months_train-test.png"

    return title, xlabel, ylabel, figname


if __name__ == "__main__":
    main()
