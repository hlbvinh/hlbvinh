# -*- coding: utf-8 -*-
import os
import sys
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from create_climate_model_plotly_dataset import (
    STORAGE_DIRECTORY,
    SENESOR_SAMPLES_FILENAME,
)
from climate_model_plotly_utils import *


samples_df = try_to_load_as_pickled_object_or_None(
    f"{STORAGE_DIRECTORY}/{SENESOR_SAMPLES_FILENAME}.pkl"
)

if samples_df is None:
    print("Loading dataframe failed. Specify the correnct folder.")
    sys.exit(1)

device_id_mode_df = samples_df[["device_id"]].drop_duplicates("device_id")

start_date = samples_df["timestamp"].min().date()

end_date = samples_df["timestamp"].max().date()

device_id_mode_df["label"] = device_id_mode_df["device_id"]
device_id_mode_df["value"] = device_id_mode_df["device_id"]

device_id_dict = device_id_mode_df[["label", "value"]].to_dict("records")
mode_dict = [{"label": value, "value": value} for value in samples_df["mode"].unique()]

app = dash.Dash()

app.layout = html.Div(
    children=[
        # Banner display
        html.Div([html.H2("Climate Model Sensor Samples", id="title")]),
        html.Div(
            style={"margin-bottom": "8px"},
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=dcc.Dropdown(
                                id="device_id",
                                options=device_id_dict,
                                placeholder="Select a device id",
                                searchable=True,
                            )
                        ),
                        html.Div(
                            children=dcc.Dropdown(
                                id="mode",
                                options=mode_dict,
                                placeholder="Select a mode",
                                searchable=True,
                                multi=True,
                            )
                        ),
                        html.Div(
                            children=dcc.Checklist(
                                id="set_temp",
                                options=[
                                    {
                                        "label": "Current Temp > Set Temp",
                                        "value": "ct>st",
                                    },
                                    {
                                        "label": "Current Temp < Set Temp",
                                        "value": "ct<st",
                                    },
                                ],
                                values=["ct>st", "ct<st"],
                            )
                        ),
                        html.Div(
                            children=dcc.Checklist(
                                id="out_temp",
                                options=[
                                    {
                                        "label": "Outside Temp less than Current Temp",
                                        "value": "ct>ot",
                                    },
                                    {
                                        "label": "Outside Temp more than Current Temp",
                                        "value": "ct<ot",
                                    },
                                ],
                                values=["ct>ot", "ct<ot"],
                            )
                        ),
                    ]
                )
            ],
        ),
        html.Div(
            style={"margin-bottom": "8px"},
            children=dcc.DatePickerRange(
                id="date-picker-range", start_date=start_date, end_date=end_date
            ),
        ),
        html.Div([dcc.Graph(id="timeseries-graph")], style={"height": "100vh"}),
    ]
)


@app.callback(
    dash.dependencies.Output("timeseries-graph", "figure"),
    [
        dash.dependencies.Input("device_id", "value"),
        dash.dependencies.Input("mode", "value"),
        dash.dependencies.Input("date-picker-range", "start_date"),
        dash.dependencies.Input("date-picker-range", "end_date"),
        dash.dependencies.Input("set_temp", "values"),
        dash.dependencies.Input("out_temp", "values"),
    ],
)
def update_graph(device_id, mode, start_date, end_date, set_temp, out_temp):

    if device_id and mode:
        new_samples_df = create_subset_of_samples_based_on_user_inputs(
            samples_df,
            mode,
            set_temp,
            out_temp,
            device_id=device_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not new_samples_df.empty:
            figure = create_timeseries_graph_for_modes_for_timerange(new_samples_df)
            return figure
        else:
            print("No data for this mode for this device_id")
            return go.Figure()

    else:
        return go.Figure()


if __name__ == "__main__":
    app.run_server(debug=True)
