import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Generate mock data (replace this with real data)
np.random.seed(42)
time = pd.date_range(
    start="2024-12-15 18:00", periods=1440, freq="3min"
)  # 72 hours at 3-min intervals
data = {
    "time": time,
    "bx": 5 * np.sin(0.01 * np.arange(len(time))) + np.random.normal(0, 0.5, len(time)),
    "by": 3 * np.cos(0.01 * np.arange(len(time))) + np.random.normal(0, 0.5, len(time)),
    "bz": 5 * np.sin(0.01 * np.arange(len(time))) + np.random.normal(0, 0.5, len(time)),
    "bt": 10 + np.random.normal(0, 0.5, len(time)),
    "phi_gsm": np.random.uniform(90, 270, len(time)),
    "density": np.random.uniform(1, 10, len(time)),
    "speed": np.random.uniform(300, 700, len(time)),
    "temperature": np.random.uniform(1e5, 1e6, len(time)),
}
df = pd.DataFrame(data)

# Create the Dash application
app = dash.Dash(__name__)


# Generate initial figure with empty vertical line
def generate_figure(hover_time=None):
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Magnetic Field (B, Bx, By, Bz)",
            "Phi GSM (deg)",
            "Density (1/cm³)",
            "Speed (km/s)",
            "Temperature (K)",
        ),
    )

    # Top plot: Magnetic Field (B, Bx, By, Bz)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bt"],
            mode="lines",
            name="Bt (nT)",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bx"],
            mode="lines",
            name="Bx (nT)",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["by"],
            mode="lines",
            name="By (nT)",
            line=dict(color="purple"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bz"],
            mode="lines",
            name="Bz (nT)",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # Second plot: Phi GSM
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["phi_gsm"],
            mode="lines",
            name="Phi GSM (deg)",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
    )

    # Third plot: Density
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["density"],
            mode="lines",
            name="Density (1/cm³)",
            line=dict(color="green"),
        ),
        row=3,
        col=1,
    )

    # Fourth plot: Speed
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["speed"],
            mode="lines",
            name="Speed (km/s)",
            line=dict(color="purple"),
        ),
        row=4,
        col=1,
    )

    # Fifth plot: Temperature
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["temperature"],
            mode="lines",
            name="Temperature (K)",
            line=dict(color="red"),
        ),
        row=5,
        col=1,
    )

    # Add vertical line across all subplots if hover_time exists
    if hover_time:
        for i in range(1, 6):
            fig.add_vline(
                x=hover_time,
                line_width=1,
                line_dash="dash",
                line_color="gray",
                row=i,
                col=1,
            )

    # Layout adjustments
    fig.update_layout(
        height=1200,
        hovermode="x",
        showlegend=False,
        xaxis5=dict(title="Time (UTC)"),  # Only add x-axis title for the last subplot
    )

    return fig


# App layout
app.layout = html.Div(
    [
        dcc.Graph(id="solar-wind-graph", config={"displayModeBar": True}),
        dcc.Store(id="hover-time"),  # Store hover time for cross-subplot vertical line
    ]
)


# Update figure when hovering
@app.callback(
    Output("solar-wind-graph", "figure"), Input("solar-wind-graph", "hoverData")
)
def update_hover_line(hoverData):
    if hoverData:
        hover_time = hoverData["points"][0]["x"]
        return generate_figure(hover_time)
    return generate_figure()


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
