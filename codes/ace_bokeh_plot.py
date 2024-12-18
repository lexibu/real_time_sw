# -*- coding: utf-8 -*-
import datetime
import importlib
import time
import geopack.geopack as gp
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, Span, CustomJS
from bokeh.layouts import column
import magnetopause_calculator as mp_calc

importlib.reload(mp_calc)


def plot_figures_ace_bokeh():
    print(
        "Code execution for ACE 2Hr started at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    ace_url_mag = "https://services.swpc.noaa.gov/text/ace-magnetometer.txt"
    ace_url_swp = "https://services.swpc.noaa.gov/text/ace-swepam.txt"

    ace_key_list_mag = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "bt",
        "lat_gsm",
        "lon_gsm",
    ]
    ace_key_list_swp = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "np",
        "vp",
        "Tp",
    ]

    df_ace_mag = pd.read_csv(
        ace_url_mag,
        sep=r"\s{1,}",
        skiprows=20,
        names=ace_key_list_mag,
        engine="python",
        dtype={"month": "string", "date": "string", "utctime": "string"},
    )
    df_ace_swp = pd.read_csv(
        ace_url_swp,
        sep=r"\s{1,}",
        skiprows=18,
        names=ace_key_list_swp,
        engine="python",
        dtype={"month": "string", "date": "string", "utctime": "string"},
    )
    df_ace_mag.replace([-999.9, -100000], np.nan, inplace=True)
    df_ace_swp.replace([-9999.9, -100000], np.nan, inplace=True)
    df_ace_mag.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_mag.year[i]}{df_ace_mag.month[i]}{df_ace_mag.date[i]}{df_ace_mag.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_mag.index))
        ]
    )

    df_ace_swp.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_swp.year[i]}{df_ace_swp.month[i]}{df_ace_swp.date[i]}{df_ace_swp.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_swp.index))
        ]
    )
    df_ace = pd.concat([df_ace_mag, df_ace_swp], axis=1)
    df_ace = df_ace.loc[:, ~df_ace.columns.duplicated()]
    df_ace["flux"] = df_ace.np * df_ace.vp * 1e-3
    df_ace["bm"] = np.sqrt(df_ace.bx_gsm**2 + df_ace.by_gsm**2 + df_ace.bz_gsm**2)
    df_ace["theta_c"] = np.arctan2(df_ace.by_gsm, df_ace.bz_gsm)

    df_ace["p_dyn"] = 1.6726e-6 * 1.15 * df_ace.np * df_ace.vp**2
    df_ace["unix_time"] = df_ace.index.astype(int) // 10**9
    for i in range(len(df_ace)):
        tilt_angle_gp = gp.recalc(df_ace.unix_time.iloc[i])
        df_ace.loc[df_ace.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    df_ace = mp_calc.mp_r_shue(df_ace)
    df_ace = mp_calc.mp_r_yang(df_ace)
    df_ace = mp_calc.mp_r_lin(df_ace)

    df_ace.reset_index(inplace=True)
    df_ace.rename(columns={"index": "time"}, inplace=True)

    source = ColumnDataSource(df_ace)

    # Create a shared span for vertical line
    span = Span(
        location=0,
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=1,
    )

    # Plot Magnetic Field
    p1 = figure(
        x_axis_type="datetime",
        title="Magnetic Field",
        outer_height=300,
        outer_width=800,
        toolbar_location="above",
    )
    p1.line("time", "bx_gsm", source=source, color="blue", legend_label="Bx")
    p1.line("time", "by_gsm", source=source, color="green", legend_label="By")
    p1.line("time", "bz_gsm", source=source, color="red", legend_label="Bz")
    p1.line("time", "bm", source=source, color="white", legend_label="|B|")
    p1.add_layout(span)
    hover1 = HoverTool()
    hover1.tooltips = [
        ("Time", "@time{%F %T}"),
        ("Bx", "@bx_gsm"),
        ("By", "@by_gsm"),
        ("Bz", "@bz_gsm"),
        ("|B|", "@bm"),
    ]
    hover1.formatters = {"@time": "datetime"}
    hover1.mode = "vline"
    p1.add_tools(hover1)
    p1.legend.location = "top_left"
    p1.background_fill_color = "#2b2b2b"
    p1.border_fill_color = "#2b2b2b"
    p1.title.text_color = "white"
    p1.legend.label_text_color = "white"
    p1.xaxis.axis_label_text_color = "white"
    p1.yaxis.axis_label_text_color = "white"
    p1.xaxis.major_label_text_color = "white"
    p1.yaxis.major_label_text_color = "white"

    # Plot Proton Density
    p2 = figure(
        x_axis_type="datetime",
        title="Proton Density",
        outer_height=300,
        outer_width=800,
        toolbar_location="above",
    )
    p2.line("time", "np", source=source, color="bisque", legend_label="np")
    p2.add_layout(span)
    hover2 = HoverTool()
    hover2.tooltips = [
        ("Time", "@time{%F %T}"),
        ("np", "@np"),
    ]
    hover2.formatters = {"@time": "datetime"}
    hover2.mode = "vline"
    p2.add_tools(hover2)
    p2.legend.location = "top_left"
    p2.background_fill_color = "#2b2b2b"
    p2.border_fill_color = "#2b2b2b"
    p2.title.text_color = "white"
    p2.legend.label_text_color = "white"
    p2.xaxis.axis_label_text_color = "white"
    p2.yaxis.axis_label_text_color = "white"
    p2.xaxis.major_label_text_color = "white"
    p2.yaxis.major_label_text_color = "white"

    # Save and show the plots
    output_file("ace_2hr_data.html")
    show(column(p1, p2))


if __name__ == "__main__":
    plot_figures_ace_bokeh()
