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

    for _, row in df.iterrows():
        bz = row["bz_gsm"]
        pdyn = row["p_dyn"]

        bzp = bz
        lim = -8.1 - 12.0 * np.log(pdyn + 1)
        if bzp < lim:
            bzp = lim

        a1 = 11.646
        a2 = 0.216
        a3 = 0.122
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
    sigma = 1.033

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


def plot_figures_ace_1day(sc=None):
    # for foo in range(1):
    """
    Download and upload data the ACE database hosted at https://services.swpc.noaa.gov/text
    """
    # Set up the time to run the job

    s.enter(0, 1, update_progress_bar, (sc, 0, 52))
    s.enter(60, 2, plot_figures_ace_1day, (sc,))

    # start = time.time()
    print(
        f"\nCode execution for ace 1day data started at at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # URL of ace files
    ace_url_mag = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    ace_url_plas = (
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    )
    ace_url_eph = "https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json"

    ace_key_list_mag = [
        "time_tag",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "lon_gsm",
        "lat_gsm",
        "bt",
    ]
    ace_key_list_plas = ["time_tag", "np", "vp", "Tp"]
    ace_key_list_eph = [
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

    df_ace_mag = pd.read_json(ace_url_mag, orient="columns")
    df_ace_plas = pd.read_json(ace_url_plas, orient="columns")
    df_ace_eph = pd.read_json(ace_url_eph, orient="columns")

    # Drop the first row of the dataframe to get rid of all strings
    df_ace_mag.drop([0], inplace=True)
    df_ace_plas.drop([0], inplace=True)
    df_ace_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_ace_mag.columns = ace_key_list_mag
    df_ace_plas.columns = ace_key_list_plas
    df_ace_eph.columns = ace_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_ace_mag.index = pd.to_datetime(df_ace_mag.time_tag)
    df_ace_plas.index = pd.to_datetime(df_ace_plas.time_tag)
    df_ace_eph.index = pd.to_datetime(df_ace_eph.time_tag)

    # Drop the time_tag column
    df_ace_mag.drop(["time_tag"], axis=1, inplace=True)
    df_ace_plas.drop(["time_tag"], axis=1, inplace=True)
    df_ace_eph.drop(["time_tag"], axis=1, inplace=True)

    df_ace_eph = df_ace_eph[
        (
            df_ace_eph.index
            >= np.nanmin([df_ace_mag.index.min(), df_ace_plas.index.min()])
        )
        & (
            df_ace_eph.index
            <= np.nanmax([df_ace_mag.index.max(), df_ace_plas.index.max()])
        )
    ]

    df_ace = pd.concat([df_ace_mag, df_ace_plas, df_ace_eph], axis=1)

    # for key in df_ace.keys():
    #     df_ace[key] = pd.to_numeric(df_ace[key])
    df_ace = df_ace.apply(pd.to_numeric)
    # Save the flux data to the dataframe
    df_ace["flux"] = df_ace.np * df_ace.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_ace["bm"] = np.sqrt(df_ace.bx_gsm**2 + df_ace.by_gsm**2 + df_ace.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_ace["theta_c"] = np.arctan2(df_ace.by_gsm, df_ace.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_ace["p_dyn"] = 1.6726e-6 * 1.15 * df_ace.np * df_ace.vp**2

    # Get the unix time for all the time tags
    df_ace["unix_time"] = df_ace.index.astype(int) // 10**9

    # Compute the dipole tilt angle
    for i in range(len(df_ace)):
        # tilt_angle_gp = gp.recalc(df_ace.unix_time[i])
        tilt_angle_gp = gp.recalc(df_ace.unix_time.iloc[i])
        df_ace.loc[df_ace.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    # Compute the magnetopause radius using the Shue et al., 1998 model
    df_ace = mp_r_shue(df_ace)

    # Compute the magnetopause radius using the Yang et al., 2011 model
    df_ace = mp_r_yang(df_ace)

    # Compute the magnetopause radius using the Lin et al., 2008 model
    df_ace = mp_r_lin(df_ace)

    # Make a copy of the dataframe at original cadence
    df_ace_hc = df_ace.copy()

    # Compute 1 hour rolling average for each of the parameters and save it to the dataframe
    df_ace = df_ace.rolling("h", center=True).median()
    # Define the plot parameters
    # cmap = plt.cm.viridis
    # pad = 0.02
    # clabelpad = 10
    labelsize = 22
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
    ncols = 2
    alpha = 0.3

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_ace.index.max() - datetime.timedelta(minutes=30)
    t2 = df_ace.index.max() - datetime.timedelta(minutes=40)

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
        df_ace.index.values, df_ace.bx_gsm.values, "r-", lw=lw, ms=ms, label=r"$B_x$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.by_gsm.values, "b-", lw=lw, ms=ms, label=r"$B_y$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.bz_gsm.values, "g-", lw=lw, ms=ms, label=r"$B_z$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.bm.values, "k-.", lw=lw, ms=ms, label=r"$|\vec{B}|$"
    )
    axs1.plot(df_ace.index.values, -df_ace.bm.values, "k-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_ace.bm), 1.1 * np.nanmax(df_ace.bm))

    axs1.set_xlim(df_ace.index.min(), df_ace.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
    # lgnd1 = axs1.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd1.legendHandles[0]._sizes = [labelsize]
    # Add a text in the plot right outside the plot along the right edge in the middle
    y_labels = [r"$|\vec{B}|$", r"$B_x$", r"$B_y$", r"$B_z$"]
    y_label_colors = ["k", "r", "b", "g"]
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

    fig.suptitle("1 Day ace Real Time Data", fontsize=22)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    axs2.plot(
        df_ace.index.values,
        df_ace.np.values,
        color="bisque",
        ls="-",
        lw=lw,
        ms=ms,
        label=r"$n_p$",
    )
    axs2.plot(
        df_ace_hc.index.values, df_ace_hc.np.values, color="bisque", lw=1, alpha=alpha
    )
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.np.isnull().all():
        axs2.set_ylim([-1, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_ace.np), 1.1 * np.nanmax(df_ace.np))

    # lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize, color="bisque")

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    axs3.plot(df_ace.index.values, df_ace.vp.values, "c-", lw=lw, ms=ms, label=r"$V_p$")
    axs3.plot(df_ace_hc.index.values, df_ace_hc.vp.values, color="c", lw=1, alpha=alpha)
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.vp.isnull().all():
        axs3.set_ylim([-1, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_ace.vp), 1.1 * np.nanmax(df_ace.vp))

    # lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd3.legend_handles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize, color="c")

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    axs4.plot(
        df_ace.index.values, df_ace.flux.values, "w-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.plot(
        df_ace_hc.index.values, df_ace_hc.flux.values, color="w", lw=1, alpha=alpha
    )
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.flux.isnull().all():
        axs4.set_ylim([-1, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_ace.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_ace.flux), 3.3]),
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
            np.nanmin(df_ace.r_shue),
            np.nanmin(df_ace.r_yang),
            np.nanmin(df_ace.r_lin),
        ]
    )
    max_rmp = np.nanmax(
        [
            np.nanmax(df_ace.r_shue),
            np.nanmax(df_ace.r_yang),
            np.nanmax(df_ace.r_lin),
        ]
    )

    axs5.plot(
        df_ace_hc.index.values, df_ace_hc.r_shue.values, color="r", lw=1, alpha=alpha
    )
    axs5.plot(
        df_ace.index.values,
        df_ace.r_shue.values,
        "r-",
        lw=lw,
        ms=ms,
        label=r"Shue",
    )

    axs5.plot(
        df_ace_hc.index.values, df_ace_hc.r_yang.values, color="b", lw=1, alpha=alpha
    )
    axs5.plot(
        df_ace.index.values,
        df_ace.r_yang.values,
        "b-",
        lw=lw,
        ms=ms,
        label=r"Yang",
    )

    axs5.plot(
        df_ace_hc.index.values, df_ace_hc.r_lin.values, color="g", lw=1, alpha=alpha
    )
    axs5.plot(
        df_ace.index.values,
        df_ace.r_lin.values,
        "g-",
        lw=lw,
        ms=ms,
        label=r"Lin",
    )
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_ace.r_shue.isnull().all()
        and df_ace.r_yang.isnull().all()
        and df_ace.r_lin.isnull().all()
    ):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_rmp, 1.03 * max_rmp)

    # lgnd5 = axs5.legend(fontsize=labelsize, loc="best", ncol=4)
    # lgnd5.legend_handles[0]._sizes = [labelsize]

    # Add a text in the plot right outside the plot along the right edge in the middle for the y-axis
    y_labels = [r"Lin", r"Yang", r"Shue"]
    y_label_colors = ["g", "b", "r"]
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
        f"Time on {df_ace.index.date[0]} (UTC) [HH:MM]", fontsize=xlabelsize
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
    fig_name = "/home/cephadrius/Dropbox/rt_sw/rt_sw_ace_parameters_1day.png"
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
    return df_ace_hc


s.enter(0, 1, plot_figures_ace_1day, (s,))
s.run()

# Print that the code has finished running and is waiting for the next update in 60 seconds
print(
    f"Code execution for ace 1day data finished at (UTC):"
    + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
)
# Display a progress bar for the next update


# if __name__ == "__main__":
#     df_ace_hc = plot_figures_ace_1day()