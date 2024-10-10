# -*- coding: utf-8 -*-
import datetime
import time
import importlib
import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from pathlib import Path

import magnetopause_calculator as mp_calc
import misc_codes as m_codes

# Reload the module to get the latest changes
importlib.reload(mp_calc)
importlib.reload(m_codes)

# s = sched.scheduler(time.time, time.sleep)

# Set the dark mode for the plots
plt.style.use("dark_background")


def plot_figures_ace_1day(sc=None):
    """
    Download and upload data the ACE database hosted at https://services.swpc.noaa.gov/text/ace-swepam-1-day.json
    """
    # Set up the time to run the job

    # s.enter(0, 1, m_codes.update_progress_bar, (sc, 0, 52))
    # s.enter(60, 2, plot_figures_ace_1day, (sc,))

    # start = time.time()
    print(
        "\nCode execution for ace 1day data started at at (UTC):"
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
    df_ace = mp_calc.mp_r_shue(df_ace)

    # Compute the magnetopause radius using the Yang et al., 2011 model
    df_ace = mp_calc.mp_r_yang(df_ace)

    # Compute the magnetopause radius using the Lin et al., 2008 model
    df_ace = mp_calc.mp_r_lin(df_ace)

    # Make a copy of the dataframe at original cadence
    df_ace_hc = df_ace.copy()

    # Compute 1 hour rolling average for each of the parameters and save it to the dataframe
    df_ace = df_ace.rolling("h", center=True).median()
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
    xlabelsize = 24
    ylabelsize = 24
    alpha = 0.3
    bar_color = "turquoise"

    ms = 2
    lw = 2
    # ncols = 2
    alpha = 0.3

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_ace.index.max() - datetime.timedelta(minutes=30)
    t2 = df_ace.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(
        num=None, figsize=(12, 15), dpi=200, facecolor="w", edgecolor="gray"
    )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
    )

    # Magnetic field plot
    gs = fig.add_gridspec(7, 1)
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
        df_ace.index.values, df_ace.bm.values, "w-.", lw=lw, ms=ms, label=r"$|\vec{B}|$"
    )
    axs1.plot(df_ace.index.values, -df_ace.bm.values, "w-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_ace.bm), 1.1 * np.nanmax(df_ace.bm))

    # In a textbox, add the average value of the magnetic field in the plot at the top
    # right corner
    avg_bm = np.nanmean(df_ace.bm)

    axs1.text(
        0.98,
        0.95,
        r"$\langle |\vec{B}| \rangle = %.2f$ nT" % avg_bm,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs1.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    axs1.set_xlim(df_ace.index.min(), df_ace.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
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

    # In a textbox, add the average value of the density in the plot at the top
    # right corner
    avg_np = np.nanmean(df_ace.np)

    axs2.text(
        0.98,
        0.95,
        r"$\langle n_p \rangle = %.2f$" % avg_np,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs2.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

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

    # In a textbox, add the average value of the speed in the plot at the top
    # right corner
    avg_vp = np.nanmean(df_ace.vp)

    axs3.text(
        0.98,
        0.95,
        r"$\langle V_p \rangle = %.2f$" % avg_vp,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs3.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize, color="w")

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

    # In a textbox, add the average value of the flux in the plot at the top
    # right corner
    avg_flux = np.nanmean(df_ace.flux)

    axs4.text(
        0.98,
        0.95,
        r"$\langle \Phi \rangle = %.2f$" % avg_flux,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs4.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    axs4.set_ylabel(
        r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize, color="w"
    )

    # Add the dynamic pressure plot
    axs5 = fig.add_subplot(gs[4:5, 0], sharex=axs1)
    axs5.plot(
        df_ace.index.values,
        df_ace.p_dyn.values,
        "m-",
        lw=lw,
        ms=ms,
        label=r"Dynamic Pressure",
    )
    axs5.plot(
        df_ace_hc.index.values, df_ace_hc.p_dyn.values, color="m", lw=1, alpha=alpha
    )
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.p_dyn.isnull().all():
        axs5.set_ylim([0, 1])
    else:
        axs5.set_ylim(0.9 * np.nanmin(df_ace.p_dyn), 1.1 * np.nanmax(df_ace.p_dyn))

    # In a textbox, add the average value of the dynamic pressure in the plot at the top
    # right corner
    avg_p_dyn = np.nanmean(df_ace.p_dyn)

    axs5.text(
        0.98,
        0.95,
        r"$\langle P_{\rm{dyn}} \rangle = %.2f$" % avg_p_dyn,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs5.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    axs5.set_yscale("linear")
    axs5.set_ylabel(r"Dynamic Pressure [nPa]", fontsize=ylabelsize, color="w")

    # Cusp latitude plot
    axs6 = fig.add_subplot(gs[5:7, 0], sharex=axs1)

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

    axs6.plot(
        df_ace.index.values,
        df_ace.r_shue.values,
        "w-",
        lw=lw,
        ms=ms,
        label=r"Shue",
    )
    axs6.plot(
        df_ace_hc.index.values, df_ace_hc.r_shue.values, color="w", lw=1, alpha=alpha
    )

    axs6.plot(
        df_ace.index.values,
        df_ace.r_yang.values,
        "b-",
        lw=lw,
        ms=ms,
        label=r"Yang",
    )
    axs6.plot(
        df_ace_hc.index.values, df_ace_hc.r_yang.values, color="b", lw=1, alpha=alpha
    )

    axs6.plot(
        df_ace.index.values,
        df_ace.r_lin.values,
        "g-",
        lw=lw,
        ms=ms,
        label=r"Lin",
    )
    axs6.plot(
        df_ace_hc.index.values, df_ace_hc.r_lin.values, color="g", lw=1, alpha=alpha
    )
    axs6.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_ace.r_shue.isnull().all()
        and df_ace.r_yang.isnull().all()
        and df_ace.r_lin.isnull().all()
    ):
        axs6.set_ylim([-1, 1])
    else:
        axs6.set_ylim(0.97 * min_rmp, 1.03 * max_rmp)

    # In a textbox, add the average value of the magnetopause distance in the plot at the top
    # right corner
    avg_rmp_shue = np.nanmean(df_ace.r_shue)
    avg_rmp_yang = np.nanmean(df_ace.r_yang)
    avg_rmp_lin = np.nanmean(df_ace.r_lin)

    axs6.text(
        0.98,
        0.95,
        r"$\langle R_{\rm{s}} \rangle = %.2f$" % avg_rmp_shue + "\n"
        r"$\langle R_{\rm{y}} \rangle = %.2f$" % avg_rmp_yang + "\n"
        r"$\langle R_{\rm{l}} \rangle = %.2f$" % avg_rmp_lin,
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs6.transAxes,
        fontsize=24,
        color="w",
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    # Add a text in the plot right outside the plot along the right edge in the middle for the y-axis
    y_labels = [r"Lin", r"Yang", r"Shue"]
    y_label_colors = ["g", "b", "w"]
    for i, txt in enumerate(y_labels):
        axs6.text(
            -0.03,
            -0.05 + 0.10 * (i + 1),
            txt,
            ha="right",
            va="center",
            transform=axs6.transAxes,
            fontsize=20,
            color=y_label_colors[i],
        )
    axs6.set_ylabel(r"Magnetopause Distance [$R_{\oplus}$]", fontsize=ylabelsize)

    axs6.set_xlabel(
        f"Time on {df_ace.index.date[0]} (UTC) [HH:MM]", fontsize=xlabelsize
    )
    # Set axis ticw-parameters
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
        labelbottom=False,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs5.yaxis.set_label_position("left")

    axs6.tick_params(
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
        color="w",
        colors="w",
    )
    axs6.yaxis.set_label_position("right")

    date_form = DateFormatter("%H:%M")
    axs6.xaxis.set_major_formatter(date_form)

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
        fontsize=24,
        rotation="vertical",
    )

    # Properly define the folder and figure name
    folder_name = "~/Dropbox/rt_sw/"
    folder_name = Path(folder_name).expanduser()
    # cd into the folder using Path
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    fig_name = "rt_sw_ace_parameters_1day.png"

    fig_name = folder_name / fig_name
    print(f"Figure saved at: {fig_name}")
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


# s.enter(0, 1, plot_figures_ace_1day, (s,))
# s.run()

# Print that the code has finished running and is waiting for the next update in 60 seconds
print(
    "Code execution for ace 1day data finished at (UTC):"
    + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
)
# Display a progress bar for the next update


if __name__ == "__main__":
    df_ace_hc = plot_figures_ace_1day()
