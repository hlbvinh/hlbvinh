# -*- coding: utf-8 -*-
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from create_climate_model_plotly_dataset import STORAGE_DIRECTORY, BASELINE_FILENAME
from climate_model_plotly_utils import *

samples_df = pd.read_pickle(f"{STORAGE_DIRECTORY}/{BASELINE_FILENAME}.pkl")

if samples_df.empty:
    print("No data. Check directory.")
    sys.exit(1)

mode_dict = [{"label": value, "value": value} for value in samples_df["mode"].unique()]

app = dash.Dash()

app.layout = html.Div(
    children=[
        # Banner display
        html.Div([html.H2("Climate Model Sensor Samples Baseline", id="title")]),
        html.Div(
            style={"margin-bottom": "8px"},
            children=[
                html.Div(
                    children=[
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
                                        "label": "Going from High to Low say 30 -> 20",
                                        "value": "ct>st",
                                    },
                                    {
                                        "label": "Going from Low to High say 20 -> 30",
                                        "value": "ct<st",
                                    },
                                ],
                                values=["ct>st", "ct<st"],
                            )
                        ),
                        html.Div(
                            children=dcc.Checklist(
                                id="outside_temp",
                                options=[
                                    {
                                        "label": "Outside Temp more than Current Temp",
                                        "value": "ct<ot",
                                    },
                                    {
                                        "label": "Outside Temp less than Current Temp",
                                        "value": "ct>ot",
                                    },
                                ],
                                values=["ct<ot", "ct>ot"],
                            )
                        ),
                    ]
                )
            ],
        ),
        html.Div([dcc.Graph(id="baseline-graph")], style={"height": "100vh"}),
    ]
)


@app.callback(
    dash.dependencies.Output("baseline-graph", "figure"),
    [
        dash.dependencies.Input("mode", "value"),
        dash.dependencies.Input("set_temp", "values"),
        dash.dependencies.Input("outside_temp", "values"),
    ],
)
def update_graph(mode, set_temp, out_temp):
    if mode:
        subset_of_samples = create_subset_of_samples_based_on_user_inputs(
            samples_df, mode, set_temp, out_temp
        )
        if not subset_of_samples.empty:
            figure = create_baseline_graph(subset_of_samples)
            return figure
        else:
            print("No data for this mode")
            return go.Figure()

    else:
        return go.Figure()


if __name__ == "__main__":
    app.run_server(debug=True)
