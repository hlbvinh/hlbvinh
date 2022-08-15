import os
import pickle
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz


def create_timeseries_graph_for_modes_for_timerange(df):
    if not df.empty:
        print("Creating plot...")
        title = (
            f"{datetime.now(pytz.timezone('Asia/Hong_Kong'))}, "
            f"{df.iloc[0]['device_id']} with number total sensor samples: {len(df)}"
        )

        trace_temp = create_trace_tm(df, "t_temperature", "temperature_color")
        trace_humidex = create_trace_tm(df, "t_humidex", "humidex_color", yaxis="y2")
        trace_humidity = create_trace_tm(df, "t_humidity", "humidity_color", yaxis="y3")
        return create_plot_common(title, trace_humidex, trace_humidity, trace_temp)
    else:
        print("Empty dataframe. No data.")
        return go.Figure()


def create_baseline_graph(df):

    if not df.empty:
        print("Creating plot...")
        interested_df = df.groupby("t_timestamp")[
            "t_humidex", "t_temperature", "t_humidity"
        ].mean().reset_index()

        current_timestamp = datetime.utcnow()

        title = "Baseline"
        if "shape" not in interested_df.columns:
            interested_df["shape"] = "circle"
            interested_df["size"] = 9
            interested_df["text"] = " "
            interested_df["temperature_color"] = "#1f77b4"
            interested_df["humidex_color"] = "#ff7f0e"
            interested_df["humidity_color"] = "#d62728"
            interested_df["t_timestamp"] = interested_df["t_timestamp"].apply(
                lambda x: current_timestamp + timedelta(x.seconds)
            )

        trace_temp = create_trace_tm(
            interested_df, "t_temperature", "temperature_color"
        )
        trace_humidex = create_trace_tm(
            interested_df, "t_humidex", "humidex_color", yaxis="y2"
        )
        trace_humidity = create_trace_tm(
            interested_df, "t_humidity", "humidity_color", yaxis="y3"
        )
        return create_baseline_axes(title, trace_humidex, trace_humidity, trace_temp)
    else:
        print("Empty dataframe. No data.")
        return go.Figure()


def create_trace_tm(raw_samples, feature, color, yaxis=None):
    if yaxis is not None:
        trace = go.Scatter(
            x=raw_samples["t_timestamp"],
            y=raw_samples[feature],
            marker={
                "size": 8, "color": raw_samples[color], "symbol": raw_samples["shape"]
            },
            yaxis=yaxis,
            name=feature,
            mode="markers",
        )
    else:
        trace = go.Scatter(
            x=raw_samples["t_timestamp"],
            y=raw_samples[feature],
            marker={
                "size": 8, "color": raw_samples[color], "symbol": raw_samples["shape"]
            },
            text=raw_samples["text"] if raw_samples.iloc[0]["text"] else None,
            name=feature,
            mode="markers",
        )
    return trace


def create_plot_common(title, trace_humidex, trace_humidity, trace_temp):
    data = [trace_temp, trace_humidex, trace_humidity]
    layout = dict(
        title=title,
        height=900,
        width=1400,
        titlefont={"size": 11},
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=3, label="3h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=14, label="14d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=2, label="2m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(),
            type="date",
        ),
        yaxis=dict(
            title="Temp",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            dtick=1,
        ),
        yaxis2=dict(
            title="Humidex",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="free",
            overlaying="y",
            dtick=2,
            side="left",
            position=0.05,
        ),
        yaxis3=dict(
            title="Humidity",
            titlefont=dict(color="#d62728"),
            tickfont=dict(color="#d62728"),
            dtick=5,
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )
    fig = dict(data=data, layout=layout)

    return fig


def create_baseline_axes(title, trace_humidex, trace_humidity, trace_temp):
    data = [trace_temp, trace_humidex, trace_humidity]
    layout = dict(
        title=title,
        height=900,
        width=1500,
        titlefont={"size": 11},
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=3, label="3h", step="hour", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(),
            type="date",
        ),
        yaxis=dict(
            title="Temp",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            showgrid=True,
            dtick=0.5,
        ),
        yaxis2=dict(
            title="Humidex",
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="free",
            showgrid=True,
            overlaying="y",
            side="left",
            position=0.05,
            dtick=0.5,
        ),
        yaxis3=dict(
            title="Humidity",
            titlefont=dict(color="#d62728"),
            tickfont=dict(color="#d62728"),
            anchor="x",
            showgrid=True,
            overlaying="y",
            side="right",
            dtick=0.5,
        ),
    )
    fig = dict(data=data, layout=layout)

    return fig


def create_subset_of_samples_based_on_user_inputs(
    samples_df,
    mode,
    temperature_set,
    temperature_out,
    device_id=None,
    start_date=None,
    end_date=None,
):
    if device_id is not None:
        samples_df = samples_df.loc[samples_df["device_id"] == device_id]

    if start_date is not None and end_date is not None:
        samples_df = samples_df.loc[
            (samples_df["timestamp"] > start_date)
            & (samples_df["timestamp"] < end_date)
        ]

    subset_criteria_mode = (samples_df["mode"].isin(mode))

    subset_criteria_temperature_set = create_subset_condition_for(
        "temperature_set", temperature_set, samples_df, "st"
    )
    subset_criteria_temperature_out = create_subset_condition_for(
        "temperature_out", temperature_out, samples_df, "ot"
    )

    subset_criteria = subset_criteria_mode & subset_criteria_temperature_set & subset_criteria_temperature_out

    subset_samples = samples_df.loc[subset_criteria]

    if subset_samples is None:
        return pd.DataFrame()
    else:
        return subset_samples


def create_subset_condition_for(
    temperature_type, temperature_condition, df, temperature_type_identifier
):
    if not temperature_condition or len(temperature_condition) > 1:
        subset_criteria = True
    else:
        if temperature_condition[0] == f"ct>{temperature_type_identifier}":
            subset_criteria = (df["temperature"] >= df[str(temperature_type)])
        elif temperature_condition[0] == f"ct<{temperature_type_identifier}":
            subset_criteria = (df["temperature"] < df[str(temperature_type)])
        else:
            subset_criteria = True
    return subset_criteria


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2 ** 31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, "rb") as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj
