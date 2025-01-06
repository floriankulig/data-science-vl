import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from analysis import DTYPES

# Initialize the Dash app
app = dash.Dash(__name__)


# Load and prepare the data
def load_data():
    dfs = []
    years = range(2008, 2024)
    for year in years:
        file_path = f"data_raw/rad_{year}_tage_19_06_23_r.csv"
        df = pd.read_csv(file_path, dtype=DTYPES)
        df["datum"] = pd.to_datetime(df["datum"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["year"] = df["datum"].dt.year
    df["month"] = df["datum"].dt.month
    return df


# Layout of the app
def create_layout(df):
    return html.Div(
        [
            html.H1(
                "Fahrrad-Zählstellen Dashboard",
                style={
                    "textAlign": "center",
                    "color": "#2c3e50",
                    "marginBottom": 16,
                },
            ),
            # Control Panel
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Zählstelle auswählen:"),
                            dcc.Dropdown(
                                id="station-dropdown",
                                options=[
                                    {"label": i, "value": i}
                                    for i in sorted(df["zaehlstelle"].unique())
                                ],
                                value=df["zaehlstelle"].iloc[0],
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={
                            "width": "45%",
                            "display": "inline-block",
                            "paddingInline": "2.5%",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Jahr auswählen:"),
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=[
                                    {"label": str(year), "value": year}
                                    for year in sorted(df["year"].unique())
                                ],
                                value=df["year"].max(),
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={
                            "width": "45%",
                            "display": "inline-block",
                            "paddingInline": "2.5%",
                        },
                    ),
                ],
                style={
                    "marginBottom": 16,
                    "position": "sticky",
                    "top": 8,
                    "backgroundColor": "rgba(152, 153, 155, 0.15)",
                    "backdropFilter": "blur(16px)",
                    "zIndex": 99,
                    "padding": 16,
                    "borderRadius": 16,
                    "border": "1px solid #dfe6e9",
                },
            ),
            # Upper Row
            html.Div(
                [
                    dcc.Graph(
                        id="daily-graph",
                        style={"backgroundColor": "rgb(252, 253, 255)"},
                    )
                ],
                style={
                    "marginBottom": 16,
                    "backgroundColor": "rgb(252, 253, 255)",
                    "borderRadius": 16,
                    "border": "1px solid #dfe6e9",
                    "padding": 16,
                },
            ),
            # Bottom Row
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="monthly-graph")],
                        style={
                            "width": "70%",
                            "borderRadius": 16,
                            "backgroundColor": "rgb(252, 253, 255)",
                            "border": "1px solid #dfe6e9",
                            "padding": 16,
                        },
                    ),
                    html.Div(
                        [
                            html.H3("Statistiken", style={"textAlign": "center"}),
                            html.Div(id="statistics-panel"),
                        ],
                        style={
                            "width": "30%",
                            "borderRadius": 16,
                            "backgroundColor": "rgb(252, 253, 255)",
                            "border": "1px solid #dfe6e9",
                            "padding": 16,
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": 16,
                    "justifyContent": "space-between",
                },
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "position": "relative",
        },
    )


# Callback for updating both graphs
@app.callback(
    [
        Output("daily-graph", "figure"),
        Output("monthly-graph", "figure"),
        Output("statistics-panel", "children"),
    ],
    [Input("station-dropdown", "value"), Input("year-dropdown", "value")],
)
def update_graphs(selected_station, selected_year):
    # Filter data
    filtered_df = df[
        (df["zaehlstelle"] == selected_station) & (df["year"] == selected_year)
    ]

    # Calculate monthly averages
    monthly_avg = (
        filtered_df.groupby("month")["gesamt"].agg(["mean", "std"]).reset_index()
    )

    # Daily Graph
    daily_fig = px.scatter(
        filtered_df,
        x="datum",
        y="gesamt",
        title=f"Tägliche Fahrrad-Zählungen {selected_year}",
    )
    daily_fig.add_trace(
        go.Scatter(
            x=filtered_df["datum"],
            y=filtered_df["gesamt"].rolling(window=7).mean(),
            name="7-Tage Durchschnitt",
            line=dict(color="red"),
        )
    )
    daily_fig.update_layout(
        xaxis_title="Datum",
        yaxis_title="Anzahl Fahrräder",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Monthly Graph
    monthly_fig = go.Figure()
    monthly_fig.add_trace(
        go.Bar(
            x=monthly_avg["month"],
            y=monthly_avg["mean"],
            error_y=dict(type="data", array=monthly_avg["std"]),
            name="Monatlicher Durchschnitt",
        )
    )
    monthly_fig.update_layout(
        title=f"Monatliche Durchschnitte {selected_year}",
        xaxis_title="Monat",
        yaxis_title="Durchschnittliche Anzahl",
        xaxis=dict(
            tickmode="array",
            ticktext=[
                "Jan",
                "Feb",
                "Mär",
                "Apr",
                "Mai",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Okt",
                "Nov",
                "Dez",
            ],
            tickvals=list(range(1, 13)),
        ),
    )

    # Statistics Panel
    stats = filtered_df["gesamt"].agg(["mean", "std", "min", "max"]).round(2)
    stats_panel = html.Div(
        [
            html.Table(
                [
                    html.Tr(
                        [html.Td("Durchschnitt:"), html.Td(f"{stats['mean']:.0f}")]
                    ),
                    html.Tr(
                        [html.Td("Standardabweichung:"), html.Td(f"{stats['std']:.0f}")]
                    ),
                    html.Tr([html.Td("Minimum:"), html.Td(f"{stats['min']:.0f}")]),
                    html.Tr([html.Td("Maximum:"), html.Td(f"{stats['max']:.0f}")]),
                ],
                style={"margin": "auto"},
            )
        ]
    )

    return daily_fig, monthly_fig, stats_panel


# Load data and create app layout
df = load_data()
app.layout = create_layout(df)

if __name__ == "__main__":
    app.run_server(debug=True)
