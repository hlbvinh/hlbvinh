from typing import Dict, Union

# pylint: disable=wrong-import-order
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    print("when running locally install matplotlib and seaborn")

import numpy as np
import pandas as pd

from skynet.prediction.mode_config import MODES

# pylint: enable=wrong-import-order

ERROR_QUANTITY_INDEX = {"error_humidex": 0, "error_humidity": 1, "error_temperature": 2}
SCORING_METRICS = {}
ANALYSES = {}


def add_as_scoring_metric(name_of_the_metric):
    def scoring_function(func):
        SCORING_METRICS[name_of_the_metric] = func
        return func

    return scoring_function


@add_as_scoring_metric("set_temperature_variation")
def set_temperature_variation_for_change_in_target_delta(X, change=1.0) -> pd.DataFrame:
    change_in_set_temperature = (
        X.groupby("sample_key")[
            ["prediction", "original_temperature_set", "temperature_set"]
        ]
        .apply(calculate_average_change_in_set_temperature, degree=change)
        .reset_index(name="scored_prediction")
    )
    return merge_dataframes(X, change_in_set_temperature)


def calculate_average_change_in_set_temperature(X: pd.DataFrame, degree: float):
    X = X.copy()

    X["target"] = estimate_original_target(X)
    set_temperature_for_pos_change = X.loc[(X["target"] + degree).abs().idxmin()][
        "temperature_set"
    ]
    set_temperature_for_neg_change = X.loc[(X["target"] - degree).abs().idxmin()][
        "temperature_set"
    ]
    return np.mean(
        [
            abs(
                set_temperature_for_pos_change - X["original_temperature_set"].values[0]
            ),
            abs(
                set_temperature_for_neg_change - X["original_temperature_set"].values[0]
            ),
        ]
    )


def estimate_original_target(X: pd.DataFrame) -> pd.Series:
    original_set_temperature_prediction = X.loc[
        (X["temperature_set"] - X["original_temperature_set"].values[0]).abs().idxmin
    ]["prediction"]
    return X["prediction"] - original_set_temperature_prediction


@add_as_scoring_metric("mae")
def mae_score(X: pd.DataFrame) -> pd.DataFrame:
    mae = (
        X[["sample_key", "mae"]]
        .drop_duplicates()
        .rename(columns={"mae": "scored_prediction"})
    )
    return merge_dataframes(X, mae)


@add_as_scoring_metric("inversion")
def number_of_inversion(X: pd.DataFrame) -> pd.DataFrame:
    return apply_to_prediction_of_each_sample_key(
        X, func=lambda x: (x.shift(1) > x).mean()
    )


@add_as_scoring_metric("mean_range")
def calculate_mean_range(X: pd.DataFrame) -> pd.DataFrame:
    return apply_to_prediction_of_each_sample_key(X, func=lambda x: x.max() - x.min())


@add_as_scoring_metric("mean_step_size")
def calculate_mean_step_size(X: pd.DataFrame) -> pd.DataFrame:
    return apply_to_prediction_of_each_sample_key(
        X, func=lambda x: x.sort_values().diff().mean()
    )


@add_as_scoring_metric("mean_max_step_size")
def calculate_max_step_size(X: pd.DataFrame) -> pd.DataFrame:
    return apply_to_prediction_of_each_sample_key(
        X, func=lambda x: x.sort_values().diff().max()
    )


@add_as_scoring_metric("mean_min_step_size")
def calculate_min_step_size(X: pd.DataFrame) -> pd.DataFrame:
    return apply_to_prediction_of_each_sample_key(
        X, func=lambda x: x.sort_values().diff().min()
    )


def apply_to_prediction_of_each_sample_key(X: pd.DataFrame, func) -> pd.DataFrame:
    scored_prediction = (
        X.groupby("sample_key")["prediction"]
        .apply(func)
        .reset_index(name="scored_prediction")
    )
    return merge_dataframes(X, scored_prediction)


def merge_dataframes(
    df_original: pd.DataFrame, df_average: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(
        df_original.drop(columns=["prediction", "temperature_set"]).drop_duplicates(),
        df_average,
        on="sample_key",
        how="inner",
    )


def add_analysis(name_of_the_analysis):
    def analysis_function(func):
        ANALYSES[name_of_the_analysis] = func
        return func

    return analysis_function


@add_analysis("analysis_on_number_of_samples")
def analysis_on_number_of_samples(dfs, scoring_metric_name, analysis_name, path):

    # Assuming same training and testing set for each dataframe
    fig, axes = plt.subplots(4, 1, figsize=(40, 60), sharex=True, sharey=True)

    for color_idx, (name, df) in enumerate(dfs):
        sample_ranges = [
            [0, 1],
            [1, df["number_of_samples"].quantile(0.1)],
            [
                df["number_of_samples"].quantile(0.1),
                df["number_of_samples"].quantile(0.2),
            ],
            [
                df["number_of_samples"].quantile(0.2),
                df["number_of_samples"].quantile(1),
            ],
        ]

        for idx, (minimum_number_of_samples, maximum_number_of_samples) in enumerate(
            sample_ranges
        ):
            subset_df = df.loc[
                (df["number_of_samples"] >= minimum_number_of_samples)
                & (df["number_of_samples"] < maximum_number_of_samples)
            ]
            sns.distplot(
                truncated_distribution(subset_df["scored_prediction"]),
                ax=axes[idx],
                label=name,
                color=random_color(color_idx),
                kde=True,
            )
            axes[idx].axvline(
                subset_df["scored_prediction"].mean(),
                linestyle="dashed",
                color=random_color(color_idx),
                linewidth=1,
            )
            axes[idx].tick_params(axis="both", which="major", labelsize=20)
            axes[idx].set_xlabel(
                f"Distribution of scoring metric for sample_id with number of samples between [{minimum_number_of_samples}, {maximum_number_of_samples}), total size {len(subset_df)}",
                fontsize=20,
            )
            axes[idx].legend()

    fig.savefig(
        f"{path}/{analysis_name}_for_{scoring_metric_name}_metrics_for_{', '.join([name for (name, _) in dfs])}.pdf",
        bbox_inches="tight",
    )


@add_analysis("analysis_on_mode")
def analysis_on_modes(dfs, scoring_metric_name, analysis_name, path):

    # Assuming same training and testing set for each dataframe
    fig, axes = plt.subplots(3, 1, figsize=(40, 60), sharex=True, sharey=True)

    for color_idx, (name, df) in enumerate(dfs):

        for idx, mode in enumerate(["cool", "heat", "dry"]):
            subset_df = df.loc[df["mode"] == mode]
            sns.distplot(
                truncated_distribution(subset_df["scored_prediction"]),
                ax=axes[idx],
                label=name,
                color=random_color(color_idx),
                kde=True,
            )
            axes[idx].axvline(
                subset_df["scored_prediction"].mean(),
                linestyle="dashed",
                color=random_color(color_idx),
                linewidth=1,
            )
            axes[idx].tick_params(axis="both", which="major", labelsize=20)
            axes[idx].set_xlabel(
                f"Distribution of scoring metric for sample_id with mode - {mode}, total samples {len(subset_df)}",
                fontsize=20,
            )
            axes[idx].legend()

    fig.savefig(
        f"{path}/{analysis_name}_for_{scoring_metric_name}_metrics_for_{', '.join([name for (name, _) in dfs])}.pdf",
        bbox_inches="tight",
    )


def random_color(x: int) -> str:
    return ["blue", "red", "green", "yellow", "grey", "brown", "orange"][x]


def truncated_distribution(X):
    return X.loc[X.between(X.quantile(0.05), X.quantile(0.95))]


def score_and_analyze(dataframes, path):
    for scoring_metric_name, scoring_metric_calculator in SCORING_METRICS.items():
        scored_dataframes = [
            [name, scoring_metric_calculator(dataframe)]
            for name, dataframe in sorted(dataframes.items())
        ]
        for analysis_name, analysis in ANALYSES.items():
            analysis(scored_dataframes, scoring_metric_name, analysis_name, path)


def create_offline_metric_graphs_and_graph_characteristics(
    predicted_df: pd.DataFrame, figure_dir: pd.DataFrame
) -> None:
    graph_characteristics_per_mode_for_all_graphs = []
    for mode in MODES + ["off"]:
        fig = plt.figure(figsize=(20, 19))
        ax = create_all_axes(fig)
        predicted_df_mode = predicted_df.loc[predicted_df["mode"] == mode]

        graph_characteristics_per_mode = []
        create_unsigned_absolute_error_graphs(predicted_df_mode, ax[0])
        graph_characteristics_per_mode.append(
            create_mae_graphs_and_characteristics(predicted_df_mode, ax[1])
        )
        graph_characteristics_per_mode.append(
            create_signed_absolute_error_graphs_and_characteristics(
                predicted_df_mode, ax[2], positive_signed=True
            )
        )
        graph_characteristics_per_mode.append(
            create_signed_absolute_error_graphs_and_characteristics(
                predicted_df_mode, ax[3], positive_signed=False
            )
        )
        characteristics_per_mode_df = pd.concat(
            graph_characteristics_per_mode, ignore_index=True
        )
        characteristics_per_mode_df["mode"] = mode
        graph_characteristics_per_mode_for_all_graphs.append(
            characteristics_per_mode_df
        )
        fig.savefig(f"{figure_dir}/figures_for_{mode}.png")

    characteristics_for_all_mode_df = pd.concat(
        graph_characteristics_per_mode_for_all_graphs, ignore_index=True
    )
    characteristics_for_all_mode_df.to_csv(
        f"{figure_dir}/figures_characteristics_for_all_modes.csv"
    )

    print(f"Figure and characteristics saved at {figure_dir}")


def create_unsigned_absolute_error_graphs(predicted_df: pd.DataFrame, ax) -> None:
    for key, index in ERROR_QUANTITY_INDEX.items():
        valid_df = predicted_df[key]
        valid_axis = ax[index]
        quantity = key.split("_")[-1]
        title = f"Unsigned Absolute Error for {quantity}"
        create_graph(valid_df, valid_axis, quantity, title, "normal")


def create_mae_graphs_and_characteristics(
    predicted_df: pd.DataFrame, ax
) -> pd.DataFrame:
    characteristics = []
    for key, index in ERROR_QUANTITY_INDEX.items():
        valid_df = np.abs(predicted_df[key])
        valid_axis = ax[index]
        quantity = key.split("_")[-1]
        title = f"MAE for {quantity}"
        create_graph(valid_df, valid_axis, quantity, title, "log")
        characteristics.append(create_graph_characteristics(valid_df, title, quantity))
    return pd.DataFrame.from_dict(characteristics)


def create_signed_absolute_error_graphs_and_characteristics(
    predicted_df: pd.DataFrame, ax, positive_signed: bool = True
) -> pd.DataFrame:
    characteristics = []
    for key, index in ERROR_QUANTITY_INDEX.items():
        valid_df = np.abs(
            predicted_df.loc[predicted_df[key] > 0][key]
            if positive_signed
            else predicted_df.loc[predicted_df[key] < 0][key]
        )
        valid_axis = ax[index]
        quantity = key.split("_")[-1]
        title = (
            f"{'Positive' if positive_signed else 'Negative'} "
            f"Absolute Error Distribution for {quantity}"
        )
        create_graph(valid_df, valid_axis, quantity, title, "log")
        characteristics.append(create_graph_characteristics(valid_df, title, quantity))
    return pd.DataFrame.from_dict(characteristics)


def set_axis(ax, qty, title, scale="log"):
    xlabel = f"{qty} error, {scale} scale"
    if scale == "log":
        ax.set(xscale="log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)


def create_bins(minimum: float, maximum: float) -> np.ndarray:
    floor, ceil = np.floor(np.log10(minimum)), np.ceil(np.log10(maximum))
    bins = []
    for i in np.arange(floor, ceil):
        bins.append(np.logspace(i, i + 1, 24))
    bins = np.unique(np.concatenate(bins, axis=0))
    return bins


def create_all_axes(fig):
    nrows = 4
    ncols = 3
    ax = [[0 * x * y for x in range(ncols)] for y in range(nrows)]
    for i in range(nrows):
        if i == 0 or i == 1:
            for j in range(ncols):
                ax[i][j] = fig.add_subplot(nrows, ncols, j + 1 + (i * ncols))
        else:
            for j in range(ncols):
                ax[i][j] = fig.add_subplot(
                    nrows, ncols, j + 1 + (i * ncols), sharex=ax[1][j]
                )
    return ax


def create_graph(df, axis, quantity, title, scale, bins=100):
    if scale == "log":
        bins = create_bins(df.min(), df.max())
    set_axis(axis, quantity, title, scale=scale)
    sns.distplot(df, bins=bins, kde=False, ax=axis)


def create_graph_characteristics(
    df: pd.DataFrame, title: str, target: str
) -> Dict[str, Union[str, float]]:
    char_dict = {
        "target": target,
        "title": title,
        "mean": np.mean(df),
        "median": np.median(df),
        "min": np.min(df),
        "max": np.max(df),
        "10 percentile": np.percentile(df, 10),
        "25 percentile": np.percentile(df, 25),
        "90 percentile": np.percentile(df, 90),
        "99 percentile": np.percentile(df, 99),
        "99.9 percentile": np.percentile(df, 99.9),
    }
    return char_dict
