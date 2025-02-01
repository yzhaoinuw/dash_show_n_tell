# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:49:08 2025

@author: yzhao
"""

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash.exceptions import PreventUpdate
from dash import Dash, dcc, html, ctx, Patch
from dash.dependencies import Input, Output, State

from dash_extensions import EventListener


class Data:
    def __init__(self, N=10240, frequency=512):
        self.N = N
        self.frequency = frequency
        self.time_start = 0
        self.time_end = int(np.ceil(N / frequency))
        self.time = np.linspace(self.time_start, self.time_end, num=self.N)
        self.signal = self.initialize_signal()
        self.labels = self.initialize_labels()

    def initialize_signal(self):
        x = np.arange(self.N)
        noise = np.random.normal(size=self.N)
        signal = np.sum(
            [
                0.5 * np.cos(1 / 3 * np.pi + 2 / 64 * np.pi * x),
                0.4 * np.sin(2 / 50 * np.pi * x),
                0.1 * noise,
            ],
            axis=0,
        )
        mask = np.ones(self.N)
        mask[1500:3000] = 0.05
        mask[6000:8000] = 0.5
        signal *= mask
        return signal

    def initialize_labels(self):
        signal_reshaped = np.reshape(self.signal, (-1, self.frequency))
        labels = np.zeros(self.time_end)
        features = np.max(abs(signal_reshaped), axis=1)
        label_1_ind = (0.1 < features) & (features <= 0.8)
        label_2_ind = features <= 0.1
        labels[label_1_ind] = 1
        labels[label_2_ind] = 2
        labels = np.expand_dims(labels, 0)
        return labels


def create_fig(data):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scattergl(
            x=data.time,
            y=data.signal,
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for i, color in enumerate(label_colors):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                mode="markers",
                marker=dict(size=8, color=color, symbol="square"),
                name=f"Label {i+1}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    labels = go.Heatmap(
        x0=0.5,
        dx=1,
        y0=0,
        dy=4,
        z=data.labels,
        hoverinfo="none",
        colorscale=colorscale,
        showscale=False,
        opacity=1,
        zmax=2,
        zmin=0,
        showlegend=False,
        xgap=0.2,  # add small gaps to serve as boundaries / ticks
    )
    fig.add_trace(labels, row=1, col=1)
    fig.update_layout(
        title=dict(
            text="Signal",
            font=dict(size=16),
            xanchor="center",
            x=0.5,
        ),
        autosize=True,
        margin=dict(t=30, l=20, r=20, b=30),
        height=400,
        hovermode="x unified",  # gives crosshair in one subplot
        xaxis=dict(tickformat="digits"),
        legend=dict(
            x=0.6,  # adjust these values to position the label legend
            y=1.1,
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="select",
        clickmode="event",
    )
    fig.update_xaxes(
        range=[data.time_start, data.time_end],
        title_text="<b>Time (s)</b>",
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[-2, 2], fixedrange=True, title_text="<b></b>", row=1, col=1)
    return fig


# %%
label_colors = ["rgb(124, 124, 251)", "rgb(251, 124, 124)", "rgb(123, 251, 123)"]
colorscale = [[0, label_colors[0]], [0.5, label_colors[1]], [1, label_colors[2]]]

graph = dcc.Graph(id="graph", config={"scrollZoom": True, "editable": False})
box_select_store = dcc.Store(id="box-select-store")
annotation_store = dcc.Store(id="annotation-store")
annotation_history_store = dcc.Store(id="annotation-history-store", data=[])
annotation_message = html.Div(id="annotation-message")
keyboard_event_listener = EventListener(
    id="keyboard", events=[{"event": "keydown", "props": ["key"]}]
)
undo_button = html.Button(
    "Undo Annotation", id="undo-button", style={"display": "none"}
)
# debug_message = html.Div(id="debug-message")

app = Dash(
    __name__, title="Time Series Labeling App", suppress_callback_exceptions=True
)
np.random.seed(0)
data = Data()
figure = create_fig(data)
graph.figure = figure
app.layout = html.Div(
    children=[
        graph,
        undo_button,
        box_select_store,
        annotation_store,
        annotation_history_store,
        annotation_message,
        keyboard_event_listener,
        # debug_message,
    ]
)

# switch_mode by pressing "m"
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, figure) {
        if (!keyboard_event || !figure) {
            return dash_clientside.no_update;
        }

        var key = keyboard_event.key;

        if (key === "m" || key === "M") {
            let updatedFigure = JSON.parse(JSON.stringify(figure));
            if (figure.layout.dragmode === "pan") {
                updatedFigure.layout.dragmode = "select"
            } else if (figure.layout.dragmode === "select") {
                var selections = figure.layout.selections;
                if (selections) {
                    if (selections.length > 0) {
                        updatedFigure.layout.selections = [];  // Remove the first selection (equivalent to pop(0) in Python)
                    }
                }
                updatedFigure.layout.dragmode = "pan"
            }
            return updatedFigure;
        }

        return dash_clientside.no_update;
    }
    """,
    Output("graph", "figure"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
)

# pan_figures using arrow keys
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, relayoutdata, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;
        var xaxisRange = figure.layout.xaxis.range;
        var x0 = xaxisRange[0];
        var x1 = xaxisRange[1];
        var newRange;

        if (key === "ArrowRight") {
            newRange = [x0 + (x1 - x0) * 0.1, x1 + (x1 - x0) * 0.1];
        } else if (key === "ArrowLeft") {
            newRange = [x0 - (x1 - x0) * 0.1, x1 - (x1 - x0) * 0.1];
        }
            
        if (newRange) {
            let updatedFigure = JSON.parse(JSON.stringify(figure));
            updatedFigure.layout = updatedFigure.layout || {};
            updatedFigure.layout.xaxis = updatedFigure.layout.xaxis || {};
            updatedFigure.layout.xaxis.range = newRange;
            
            relayoutdata['xaxis.range[0]'] = newRange[0];
            relayoutdata['xaxis.range[1]'] = newRange[1];
            
            return [updatedFigure, relayoutdata];
        }
        
        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "relayoutData"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "relayoutData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


@app.callback(
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure):
    selections = figure["layout"].get("selections")
    # dragmode = figure["layout"]["dragmode"]
    if not selections:
        return [], dash.no_update, ""

    patched_figure = Patch()
    # allow only at most one select box in all subplots
    if len(selections) > 1:
        selections.pop(0)

    patched_figure["layout"][
        "selections"
    ] = selections  # patial property update: https://dash.plotly.com/partial-properties#update

    # take the min as start and max as end so that how the box is drawn doesn't matter
    start, end = min(selections[0]["x0"], selections[0]["x1"]), max(
        selections[0]["x0"], selections[0]["x1"]
    )
    duration = len(figure["data"][-1]["z"][0])

    if end < 0 or start > duration:
        return [], patched_figure, ""

    start_round, end_round = round(start), round(end)
    start_round = max(start_round, 0)
    end_round = min(end_round, duration)
    if start_round == end_round:
        if (
            start_round - start > end - end_round
        ):  # spanning over two consecutive seconds
            end_round = np.ceil(start)
            start_round = np.floor(start)
        else:
            end_round = np.ceil(end)
            start_round = np.floor(end)

    start, end = start_round, end_round

    return (
        [start, end],
        patched_figure,
        "Draw a box to annotate. Press 1 for Blue, 2 for Coral, 3 for Green.",
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-store", "data"),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),  # a keyboard press
    State("keyboard", "event"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def update_labels(box_select_range, keyboard_press, keyboard_event, figure):
    if not (ctx.triggered_id == "keyboard" and box_select_range):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3"]:
        raise PreventUpdate

    label = int(label) - 1
    start, end = box_select_range
    # If the annotation does not change anything, don't add to history
    if (
        figure["data"][-1]["z"][0][start:end] == np.array([label] * (end - start))
    ).all():
        raise PreventUpdate

    patched_figure = Patch()
    prev_labels = figure["data"][-1]["z"][0][start:end]
    figure["data"][-1]["z"][0][start:end] = [label] * (end - start)
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]

    # remove box select after an update is made
    patched_figure["layout"]["selections"].clear()

    return patched_figure, (start, end, prev_labels)


@app.callback(
    Output("undo-button", "style"),
    Output("annotation-history-store", "data", allow_duplicate=True),
    Input("annotation-store", "data"),
    State("annotation-history-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def write_annotation_history(annotation, annotation_history, figure):
    """write to annotation history and make undo button availabe"""
    start, end, prev_labels = annotation
    annotation_history.append((start, end, prev_labels))
    if len(annotation_history) > 3:
        annotation_history.pop(0)
    return {"display": "block"}, annotation_history


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("undo-button", "style", allow_duplicate=True),
    Output("annotation-history-store", "data", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    State("annotation-history-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks, annotation_history, figure):
    prev_annotation = annotation_history.pop()
    (start, end, prev_labels) = prev_annotation
    prev_labels = np.array(prev_labels)
    patched_figure = Patch()
    figure["data"][-1]["z"][0][start:end] = prev_labels
    patched_figure["data"][-1]["z"][0] = figure["data"][-1]["z"][0]

    if not annotation_history:
        return patched_figure, {"display": "none"}, annotation_history
    return patched_figure, {"display": "block"}, annotation_history
