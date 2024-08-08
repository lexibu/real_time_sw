# -*- coding: utf-8 -*-
import datetime
import sched
import time
import sys

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

s = sched.scheduler(time.time, time.sleep)

# Set the dark mode for the plots
plt.style.use("dark_background")


def progress_bar(progress, total):
    """
    Function to display a progress bar in the terminal

    Parameters
    ----------
    progress : int
        The current progress value
    total : int
        The total number of steps to be completed

    Returns
    -------
    None
    """
    bar_length = 50  # Length of the progress bar
    block = int(round(bar_length * progress / total))
    filled_color = "\033[42m"  # Green background
    reset_color = "\033[0m"  # Reset color
    progress_display = (
        filled_color + " " * block + reset_color + "-" * (bar_length - block)
    )
    text = f"\rNext update in : [{progress_display}] {total - progress} seconds"
    sys.stdout.write(text)
    sys.stdout.flush()


def update_progress_bar(sc, current_step, total_steps):
    """
    Function to update the progress bar in the terminal at regular intervals using the sched module
    in Python standard library to schedule the next update of the progress bar at regular intervals
    of time until the progress is complete and then print

    Parameters
    ----------
    sc : sched.scheduler
        The scheduler object
    current_step : int
        The current step in the progress
    total_steps : int
        The total number of steps to be completed

    Returns
    -------
    None
    """
    progress_bar(current_step, total_steps)
    if current_step < total_steps:
        # Schedule the next update
        sc.enter(
            52 / total_steps,
            1,
            update_progress_bar,
            (sc, current_step + 1, total_steps),
        )
    # else:
    #     # Print a new line when the progress is complete
    #     print("")


def mp_r_shue(df):
    """
    Function to compute the magnetopause radius using the Shue et al., 1998 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    theta = np.arctan2(np.sqrt(df["z_gsm"] ** 2 + df["y_gsm"] ** 2), df["x_gsm"])
    # Check if all theta values are nan, if they are then set them to 0
    if np.isnan(theta).all():
        theta = np.zeros(len(theta))
    ro = (10.22 + 1.29 * np.tanh(0.184 * (df["bz_gsm"] + 8.14))) * (df["p_dyn"]) ** (
        -1 / 6.6
    )
    alpha = (0.58 - 0.007 * df["bz_gsm"]) * (1 + 0.024 * np.log(df["p_dyn"]))
    r = ro * (2 / (1 + np.cos(theta))) ** alpha
    df["r_shue"] = r
    return df


def mp_r_yang(df):
    """
    Function to compute the magnetopause radius using the Yang et al., 2011 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    for _, row in df.iterrows():
        bz = row["bz_gsm"]
        pdyn = row["p_dyn"]

        bzp = bz
        lim = -8.1 - 12.0 * np.log(pdyn + 1)
        if bzp < lim:
            bzp = lim

        a1 = 11.646
        a2 = 0.216
        # a3 = 0.122
        a4 = 6.215
        a5 = 0.578
        a6 = -0.009
        a7 = 0.012
        a7 = a7 * np.exp(-1 * pdyn / 30)
        alpha = (a5 + a6 * bzp) * (1 + a7 * pdyn)

        if bzp >= 0:
            ro = a1 * pdyn ** (-1.0 / a4)
        elif -8 <= bzp < 0:
            ro = (a1 + a2 * bzp) * pdyn ** (-1.0 / a4)
        else:
            ro = (a1 + a2 * bzp) * pdyn ** (-1.0 / a4)

        theta = 2 * np.pi * 0 / 360
        r = ro * (2 / (1 + np.cos(theta))) ** alpha

        df.loc[_, "r_yang"] = r
    return df


def mp_r_lin(df):
    """
    Function to compute the magnetopause radius using the Lin et al., 2008 model

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the solar wind parameters with the magnetopause radius computed
    """
    a0 = 12.544
    a1 = -0.194
    a2 = 0.305
    a3 = 0.0573
    a4 = 2.178
    a5 = 0.0571
    a6 = -0.999
    a7 = 16.473
    a8 = 0.00152
    a9 = 0.381
    a10 = 0.0431
    a11 = -0.00763
    a12 = -0.210
    a13 = 0.0405
    a14 = -4.430
    a15 = -0.636
    a16 = -2.600
    a17 = 0.832
    a18 = -5.328
    a19 = 1.103
    a20 = -0.907
    a21 = 1.450
    # sigma = 1.033

    pmag = 0  # magnetic pressure, assumed to be zero
    theta = 0
    phi = 0

    beta0 = a6 + a7 * (np.exp(a8 * df["bz_gsm"]) - 1) / (np.exp(a9 * df["bz_gsm"]) + 1)
    beta1 = a10
    beta2 = a11 + a12 * df["dipole_tilt"]
    beta3 = a13

    dn = a16 + a17 * df["dipole_tilt"] + a18 * df["dipole_tilt"] ** 2
    ds = a16 - a17 * df["dipole_tilt"] + a18 * df["dipole_tilt"] ** 2

    thetan = a19 + a20 * df["dipole_tilt"]
    thetas = a19 - a20 * df["dipole_tilt"]

    en = a21
    es = a21

    cn = a14 * df["p_dyn"] ** a15
    cs = cn

    psi_s = np.arccos(
        np.cos(theta) * np.cos(thetas)
        + np.sin(theta) * np.sin(thetas) * np.cos(phi - 3 * np.pi / 2)
    )
    psi_n = np.arccos(
        np.cos(theta) * np.cos(thetan)
        + np.sin(theta) * np.sin(thetan) * np.cos(phi - np.pi / 2)
    )

    ex = beta0 + beta1 * np.cos(phi) + beta2 * np.sin(phi) + beta3 * (np.sin(phi)) ** 2
    f = (np.cos(theta / 2) + a5 * np.sin(2 * theta) * (1 - np.exp(-theta))) ** ex
    r0 = (
        a0
        * (df["p_dyn"] + pmag) ** a1
        * (1 + a2 * (np.exp(a3 * df["bz_gsm"]) - 1) / (np.exp(a4 * df["bz_gsm"]) + 1))
    )
    r = r0 * f + cn * np.exp(dn * psi_n**en) + cs * np.exp(ds * psi_s**es)

    df["r_lin"] = r
    return df


def plot_figures_dsco_1day(sc=None):
    # for foo in range(1):
    """
    Download and upload data the ACE database hosted at
    https://services.swpc.noaa.gov/text/ace-swepam-1-day.json
    The data is then processed to compute the solar wind parameters and the magnetopause radius using
    the Shue et al., 1998 model, the Yang et al., 2011 model and the Lin et al., 2008 model.
    The data is then plotted and saved to a file in the Dropbox folder. The function is scheduled to
    run at regular intervals using the sched module in Python standard library to update the plots at
    regular intervals of time.

    Parameters
    ----------
    sc : sched.scheduler
        The scheduler object

    Returns
    -------
    df_dsco_hc : pandas.DataFrame
        The dataframe containing the solar wind parameters

    """
    # Set up the time to run the job
    s.enter(0, 1, update_progress_bar, (sc, 0, 52))
    s.enter(60, 1, plot_figures_dsco_1day, (sc,))

    # start = time.time()
    print(
        f"\nCode execution for DSCOVR 1day data started at at (UTC):"
        + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # URL of dscovr files
    dscovr_url_mag = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    dscovr_url_plas = (
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    )
    dscovr_url_eph = (
        "https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json"
    )

    dscovr_key_list_mag = [
        "time_tag",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "lon_gsm",
        "lat_gsm",
        "bt",
    ]
    dscovr_key_list_plas = ["time_tag", "np", "vp", "Tp"]
    dscovr_key_list_eph = [
        "time_tag",
        "x_gse",
        "y_gse",
        "z_gse",
        "vx_gse",
        "vy_gse",
        "vz_gse",
        "x_gsm",
        "y_gsm",
        "z_gsm",
        "vx_gsm",
        "vy_gsm",
        "vz_gsm",
    ]

    df_dsco_mag = pd.read_json(dscovr_url_mag, orient="columns")
    df_dsco_plas = pd.read_json(dscovr_url_plas, orient="columns")
    df_dsco_eph = pd.read_json(dscovr_url_eph, orient="columns")

    # Drop the first row of the dataframe to get rid of all strings
    df_dsco_mag.drop([0], inplace=True)
    df_dsco_plas.drop([0], inplace=True)
    df_dsco_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_dsco_mag.columns = dscovr_key_list_mag
    df_dsco_plas.columns = dscovr_key_list_plas
    df_dsco_eph.columns = dscovr_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_dsco_mag.index = pd.to_datetime(df_dsco_mag.time_tag)
    df_dsco_plas.index = pd.to_datetime(df_dsco_plas.time_tag)
    df_dsco_eph.index = pd.to_datetime(df_dsco_eph.time_tag)

    # Drop the time_tag column
    df_dsco_mag.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_plas.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_eph.drop(["time_tag"], axis=1, inplace=True)

    df_dsco_eph = df_dsco_eph[
        (
            df_dsco_eph.index
            >= np.nanmin([df_dsco_mag.index.min(), df_dsco_plas.index.min()])
        )
        & (
            df_dsco_eph.index
            <= np.nanmax([df_dsco_mag.index.max(), df_dsco_plas.index.max()])
        )
    ]

    df_dsco = pd.concat([df_dsco_mag, df_dsco_plas, df_dsco_eph], axis=1)

    # for key in df_dsco.keys():
    #     df_dsco[key] = pd.to_numeric(df_dsco[key])
    df_dsco = df_dsco.apply(pd.to_numeric)
    # Save the flux data to the dataframe
    df_dsco["flux"] = df_dsco.np * df_dsco.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dsco["bm"] = np.sqrt(df_dsco.bx_gsm**2 + df_dsco.by_gsm**2 + df_dsco.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_dsco["theta_c"] = np.arctan2(df_dsco.by_gsm, df_dsco.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dsco["p_dyn"] = 1.6726e-6 * 1.15 * df_dsco.np * df_dsco.vp**2

    # Get the unix time for all the time tags
    df_dsco["unix_time"] = df_dsco.index.astype(int) // 10**9

    # Compute the dipole tilt angle
    for i in range(len(df_dsco)):
        tilt_angle_gp = gp.recalc(df_dsco.unix_time.iloc[i])
        df_dsco.loc[df_dsco.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    # Compute the magnetopause radius using the Shue et al., 1998 model
    df_dsco = mp_r_shue(df_dsco)

    # Compute the magnetopause radius using the Yang et al., 2011 model
    df_dsco = mp_r_yang(df_dsco)

    # Compute the magnetopause radius using the Lin et al., 2008 model
    df_dsco = mp_r_lin(df_dsco)

    # Make a copy of the dataframe at original cadence
    df_dsco_hc = df_dsco.copy()

    # Compute 1 hour rolling average for each of the parameters and save it to the dataframe
    df_dsco = df_dsco.rolling("h", center=True).median()
    # Define the plot parameters
    # cmap = plt.cm.viridis
    # pad = 0.02
    # clabelpad = 10
    # labelsize = 22
    ticklabelsize = 20
    # cticklabelsize = 15
    # clabelsize = 15
    ticklength = 6
    tickwidth = 1.0
    # mticklength = 4
    # cticklength = 5
    # mcticklength = 4
    # labelrotation = 0
    xlabelsize = 20
    ylabelsize = 20
    alpha = 0.3
    bar_color = "k"

    ms = 2
    lw = 2
    # ncols = 2
    alpha = 0.3

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_dsco.index.max() - datetime.timedelta(minutes=30)
    t2 = df_dsco.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(
        num=None, figsize=(10, 13), dpi=200, facecolor="w", edgecolor="gray"
    )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
    )

    # Magnetic field plot
    gs = fig.add_gridspec(6, 1)
    axs1 = fig.add_subplot(gs[0, 0])
    axs1.plot(
        df_dsco.index.values, df_dsco.bx_gsm.values, "r-", lw=lw, ms=ms, label=r"$B_x$"
    )
    axs1.plot(
        df_dsco.index.values, df_dsco.by_gsm.values, "b-", lw=lw, ms=ms, label=r"$B_y$"
    )
    axs1.plot(
        df_dsco.index.values, df_dsco.bz_gsm.values, "g-", lw=lw, ms=ms, label=r"$B_z$"
    )
    axs1.plot(
        df_dsco.index.values,
        df_dsco.bm.values,
        "w-.",
        lw=lw,
        ms=ms,
        label=r"$|\vec{B}|$",
    )
    axs1.plot(df_dsco.index.values, -df_dsco.bm.values, "w-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_dsco.bm), 1.1 * np.nanmax(df_dsco.bm))

    axs1.set_xlim(df_dsco.index.min(), df_dsco.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
    # lgnd1 = axs1.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd1.legend_handles[0]._sizes = [labelsize]
    # Add a text in the plot right outside the plot along the right edge in the middle
    y_labels = [r"$|\vec{B}|$", r"$B_x$", r"$B_y$", r"$B_z$"]
    y_label_colors = ["w", "r", "b", "g"]
    for i, txt in enumerate(y_labels):
        axs1.text(
            1.01,
            -0.05 + 0.20 * (i + 1),
            txt,
            ha="left",
            va="center",
            transform=axs1.transAxes,
            fontsize=20,
            color=y_label_colors[i],
        )
    fig.suptitle("1 Day DSCOVR Real Time Data", fontsize=22)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    axs2.plot(
        df_dsco.index.values,
        df_dsco.np.values,
        color="bisque",
        ls="-",
        lw=lw,
        ms=ms,
        label=r"$n_p$",
    )
    axs2.plot(
        df_dsco_hc.index.values, df_dsco_hc.np.values, color="bisque", lw=1, alpha=alpha
    )
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.np.isnull().all():
        axs2.set_ylim([-1, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_dsco.np), 1.1 * np.nanmax(df_dsco.np))

    # lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd2.legend_handles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize, color="r")

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    axs3.plot(
        df_dsco.index.values, df_dsco.vp.values, "c-", lw=lw, ms=ms, label=r"$V_p$"
    )
    axs3.plot(
        df_dsco_hc.index.values, df_dsco_hc.vp.values, color="c", lw=1, alpha=alpha
    )
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.vp.isnull().all():
        axs3.set_ylim([-1, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_dsco.vp), 1.1 * np.nanmax(df_dsco.vp))

    # lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd3.legend_handles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize, color="c")

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    axs4.plot(
        df_dsco.index.values, df_dsco.flux.values, "w-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.plot(
        df_dsco_hc.index.values, df_dsco_hc.flux.values, color="w", lw=1, alpha=alpha
    )
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.flux.isnull().all():
        axs4.set_ylim([-1, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_dsco.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_dsco.flux), 3.3]),
        )

    # lgnd4 = axs4.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd4.legend_handles[0]._sizes = [labelsize]
    axs4.set_ylabel(
        r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize, color="w"
    )

    # Cusp latitude plot

    axs5 = fig.add_subplot(gs[4:, 0], sharex=axs1)

    min_rmp = np.nanmin(
        [
            np.nanmin(df_dsco.r_shue),
            np.nanmin(df_dsco.r_yang),
            np.nanmin(df_dsco.r_lin),
        ]
    )
    max_rmp = np.nanmax(
        [
            np.nanmax(df_dsco.r_shue),
            np.nanmax(df_dsco.r_yang),
            np.nanmax(df_dsco.r_lin),
        ]
    )

    axs5.plot(
        df_dsco_hc.index.values, df_dsco_hc.r_shue.values, color="w", lw=1, alpha=alpha
    )
    axs5.plot(
        df_dsco.index.values,
        df_dsco.r_shue.values,
        "w-",
        lw=lw,
        ms=ms,
        label=r"Shue",
    )

    axs5.plot(
        df_dsco_hc.index.values, df_dsco_hc.r_yang.values, color="b", lw=1, alpha=alpha
    )
    axs5.plot(
        df_dsco.index.values,
        df_dsco.r_yang.values,
        "b-",
        lw=lw,
        ms=ms,
        label=r"Yang",
    )

    axs5.plot(
        df_dsco_hc.index.values, df_dsco_hc.r_lin.values, color="g", lw=1, alpha=alpha
    )
    axs5.plot(
        df_dsco.index.values,
        df_dsco.r_lin.values,
        "g-",
        lw=lw,
        ms=ms,
        label=r"Lin",
    )
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_dsco.r_shue.isnull().all()
        and df_dsco.r_yang.isnull().all()
        and df_dsco.r_lin.isnull().all()
    ):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_rmp, 1.03 * max_rmp)

    # lgnd5 = axs5.legend(fontsize=labelsize, loc="best", ncol=4)
    # lgnd5.legend_handles[0]._sizes = [labelsize]

    # Add a text in the plot right outside the plot along the right edge in the middle for the y-axis
    y_labels = [r"Lin", r"Yang", r"Shue"]
    y_label_colors = ["g", "b", "w"]
    for i, txt in enumerate(y_labels):
        axs5.text(
            1.01,
            -0.05 + 0.10 * (i + 1),
            txt,
            ha="left",
            va="center",
            transform=axs5.transAxes,
            fontsize=20,
            color=y_label_colors[i],
        )

    axs5.set_xlabel(
        f"Time on {df_dsco.index.date[0]} (UTC) [HH:MM]", fontsize=xlabelsize
    )
    axs5.set_ylabel(r"Magnetopause Distance [$R_{\oplus}$]", fontsize=ylabelsize)

    # Set axis tick-parameters
    axs1.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        bottom=True,
        labelbottom=False,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )

    axs2.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=False,
        top=True,
        labeltop=False,
        right=True,
        labelright=True,
        bottom=True,
        labelbottom=False,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs2.yaxis.set_label_position("right")

    axs3.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        bottom=True,
        labelbottom=False,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )

    axs4.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=False,
        top=True,
        labeltop=False,
        right=True,
        labelright=True,
        bottom=True,
        labelbottom=False,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs4.yaxis.set_label_position("right")

    axs5.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        bottom=True,
        labelbottom=True,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs5.yaxis.set_label_position("left")
    date_form = DateFormatter("%H:%M")
    axs5.xaxis.set_major_formatter(date_form)

    figure_time = (
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    axs3.text(
        -0.1,
        0.5,
        f"Figure plotted on {figure_time[0:10]} at {figure_time[11:]} UTC",
        ha="right",
        va="center",
        transform=axs3.transAxes,
        fontsize=20,
        rotation="vertical",
    )
    fig_name = "/home/cephadrius/Dropbox/rt_sw/rt_sw_dsco_parameters_1day.png"
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300)

    # axs1.set_ylim([-22, 22])
    # axs2.set_ylim([0, 40])
    # axs3.set_ylim([250, 700])
    # axs4.set_ylim([0, 20])
    # axs5.set_ylim([60, 85])

    # plt.tight_layout()
    plt.close("all")
    print(
        "Figure saved at (UTC):"
        + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # print(f'It took {round(time.time() - start, 3)} seconds')
    # return df
    return df_dsco_hc


s.enter(0, 1, plot_figures_dsco_1day, (s,))
s.run()

# if __name__ == "__main__":
#     df_dsco_hc = plot_figures_dsco_1day()
