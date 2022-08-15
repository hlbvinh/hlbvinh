import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("when running locally install matplotlib and seaborn")

COMPUTATIONS: List = []
PLOTS: Dict = {}


def add_computation(func):
    COMPUTATIONS.append(func)
    return func


@add_computation
def compute_absolute_mae(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mae"] = (df["y_pred"] - df["feedback"]).abs()
    return df


@add_computation
def compute_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.apply(hour, axis=1)
    return df


def hour(row: pd.Series) -> int:
    tod_sin = row["tod_cos"]
    tod_cos = row["tod_sin"]
    tod_tan = tod_sin / tod_cos
    theta = np.arctan(tod_tan)
    if tod_cos < 0:
        theta += np.pi
    elif tod_sin < 0:
        theta += 2 * np.pi
    return math.floor(theta * 24 / (2 * np.pi))


def generate_plots(dataframes: Dict, path: str) -> None:
    for computation in COMPUTATIONS:
        dataframes = {
            name: computation(dataframe) for name, dataframe in dataframes.items()
        }

    f = plt.figure(figsize=(8, 14))
    for i, (plot_name, plot_function) in enumerate(PLOTS.items()):
        f.add_subplot(len(PLOTS), 1, i + 1)
        plot_function(dataframes, plot_name)

    plt.savefig(f"{path}/comfort_model_metrics_plots.pdf")


def add_plot(name_of_plot):
    def plot_function(func):
        PLOTS[name_of_plot] = func
        return func

    return plot_function


@add_plot("sample")
def plot_mae(dataframes: Dict, plot_name: str) -> None:
    for name, df in dataframes.items():
        plt.hist(df["mae"], bins=50, label=name, alpha=0.6)
    set_title_and_axes(plot_name, is_group=False)


@add_plot("feedback type")
def plot_mae_per_feedback_type(dataframes: Dict, plot_name: str) -> None:
    for name, df in dataframes.items():
        plt.bar(*group_mae(df, "feedback"), label=name, alpha=0.6)
    set_title_and_axes(plot_name, is_group=True)


@add_plot("hour of day")
def plot_mae_per_hour_of_day(dataframes: Dict, plot_name: str) -> None:
    for name, df in dataframes.items():
        plt.bar(*group_mae(df, "hour"), label=name, alpha=0.6)
    set_title_and_axes(plot_name, is_group=True)


def group_mae(df: pd.DataFrame, group: str) -> Tuple:
    stats = df.groupby(group)["mae"].mean()
    return stats.index, stats.values


def set_title_and_axes(plot_name: str, is_group: bool):
    plt.title(f"mean absolute error (mae) per {plot_name}")
    if is_group:
        plt.xlabel(plot_name)
        plt.ylabel("mae")
    else:
        plt.xlabel("mae")
        plt.ylabel("count")
    plt.legend(loc="upper right")
