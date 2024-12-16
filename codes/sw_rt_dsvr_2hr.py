# -*- coding: utf-8 -*-
import datetime
import time
import importlib
import matplotlib as mpl
import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from pathlib import Path
import magnetopause_calculator as mp_calc

# Reload the module to get the latest changes
importlib.reload(mp_calc)

# s = sched.scheduler(time.time, time.sleep)

mpl.use("Agg")
# Set the dark mode for the plots
plt.style.use("dark_background")

# Set the font style to Times New Roman
font = {
    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
    "weight": "normal",
    "size": 20,
}
plt.rc("font", **font)
plt.rc("text", usetex=False)
mpl.rcParams["pgf.texsystem"] = "pdflatex"


def plot_figures_dsco(sc=None):
    # for xxx in range(1):
    """
    Download and upload data the DSCOVR database hosted at https://services.swpc.noaa.gov/text
    """
    # Set up the time to run the job
    # s.enter(60, 1, plot_figures_dsco, (sc,))

    print(
        "Code execution for DSCOVR 2Hr started at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # URL of plas and magnetometer files
    dscovr_url_mag = (
        "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"
    )
    dscovr_url_plas = (
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json"
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
    # dscovr_key_list_eph = ["time_tag", "x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse",
    #                       "x_gsm", "y_gsm", "z_gsm", "vx_gsm", "vy_gsm", "vz_gsm"]

    df_dsco_mag = pd.read_json(dscovr_url_mag, orient="columns")
    df_dsco_plas = pd.read_json(dscovr_url_plas, orient="columns")
    # df_dsco_eph = pd.read_json(dscovr_url_eph, orient='columns')

    # Drop the first row of the dataframe to get rid of all strings
    df_dsco_mag.drop([0], inplace=True)
    df_dsco_plas.drop([0], inplace=True)
    # df_dsco_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_dsco_mag.columns = dscovr_key_list_mag
    df_dsco_plas.columns = dscovr_key_list_plas
    # df_dsco_eph.columns = dscovr_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_dsco_mag.index = pd.to_datetime(df_dsco_mag.time_tag)
    df_dsco_plas.index = pd.to_datetime(df_dsco_plas.time_tag)
    # df_dsco_eph.index = pd.to_datetime(df_dsco_eph.time_tag)

    # Drop the time_tag column
    df_dsco_mag.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_plas.drop(["time_tag"], axis=1, inplace=True)
    # df_dsco_eph.drop(["time_tag"], axis=1, inplace=True)

    # df_dsco_eph = df_dsco_eph[(df_dsco_eph.index >=
    #                           np.nanmin([df_dsco_mag.index.min(), df_dsco_plas.index.min()])) &
    #                          (df_dsco_eph.index <=
    #                           np.nanmax([df_dsco_mag.index.max(), df_dsco_plas.index.max()]))]

    # df_dsco = pd.concat([df_dsco_mag, df_dsco_plas, df_dsco_eph], axis=1)
    df_dsco = pd.concat([df_dsco_mag, df_dsco_plas], axis=1)

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
        # tilt_angle_gp = gp.recalc(df_dsco.unix_time[i])
        tilt_angle_gp = gp.recalc(df_dsco.unix_time.iloc[i])
        df_dsco.loc[df_dsco.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    # Compute the magnetopause radius using the Shue et al., 1998 model
    df_dsco = mp_calc.mp_r_shue(df_dsco)

    # Compute the magnetopause radius using the Yang et al., 2011 model
    df_dsco = mp_calc.mp_r_yang(df_dsco)

    # Compute the magnetopause radius using the Lin et al., 2008 model
    df_dsco = mp_calc.mp_r_lin(df_dsco)

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

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_dsco.index.max() - datetime.timedelta(minutes=30)
    t2 = df_dsco.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(
        num=None, figsize=(12, 15), dpi=200, facecolor="w", edgecolor="gray"
    )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
    )
    fig.suptitle("2 Hours DSCOVR Real Time Data", fontsize=24)

    # Magnetic field plot
    gs = fig.add_gridspec(7, 1)
    axs1 = fig.add_subplot(gs[0, 0])
    (_,) = axs1.plot(
        df_dsco.index.values,
        df_dsco.bx_gsm.values,
        "b-",
        lw=0.5 * lw,
        ms=ms,
        label=r"$B_x$",
    )
    (_,) = axs1.plot(
        df_dsco.index.values,
        df_dsco.by_gsm.values,
        "g-",
        lw=0.5 * lw,
        ms=ms,
        label=r"$B_y$",
    )
    (_,) = axs1.plot(
        df_dsco.index.values,
        df_dsco.bz_gsm.values,
        "r-",
        lw=1.5 * lw,
        ms=ms,
        label=r"$B_z$",
    )
    (_,) = axs1.plot(
        df_dsco.index.values,
        df_dsco.bm.values,
        "w-.",
        lw=0.5 * lw,
        ms=ms,
        label=r"$|\vec{B}|$",
    )
    (_,) = axs1.plot(
        df_dsco.index.values, -df_dsco.bm.values, "w-.", lw=0.5 * lw, ms=ms
    )
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    # Add a white line at y=0
    axs1.axhline(0, color="w", lw=1, ls="--")

    if df_dsco.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_dsco.bm), 1.1 * np.nanmax(df_dsco.bm))

    # In a textbox, add the average value of the magnetic field in the plot at the top
    # right corner
    avg_bm = np.nanmean(df_dsco.bm)

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
    axs1.set_xlim(df_dsco.index.min(), df_dsco.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)

    # Add a text in the plot right outside the plot along the right edge in the middle
    y_labels = [r"$|\vec{B}|$", r"$B_x$", r"$B_y$", r"$B_z$"]
    y_label_colors = ["w", "b", "g", "r"]
    for i, txt in enumerate(y_labels):
        axs1.text(
            1.01,
            -0.05 + 0.20 * (i + 1),
            txt,
            ha="left",
            va="center",
            transform=axs1.transAxes,
            fontsize=24,
            color=y_label_colors[i],
        )

    # axs1.text(0.98, 0.95, f'{model_type}', horizontalalignment='right', verticalalignment='center',
    #          transform=axs1.transAxes, fontsize=18)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    (_,) = axs2.plot(
        df_dsco.index.values,
        df_dsco.np.values,
        color="bisque",
        ls="-",
        lw=lw,
        ms=ms,
        label=r"$n_p$",
    )
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.np.isnull().all():
        axs2.set_ylim([0, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_dsco.np), 1.1 * np.nanmax(df_dsco.np))

    # In a textbox, add the average value of the density in the plot at the top
    # right corner
    avg_np = np.nanmean(df_dsco.np)

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
    (_,) = axs3.plot(
        df_dsco.index.values, df_dsco.vp.values, "c-", lw=lw, ms=ms, label=r"$V_p$"
    )
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.vp.isnull().all():
        axs3.set_ylim([0, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_dsco.vp), 1.1 * np.nanmax(df_dsco.vp))

    # In a textbox, add the average value of the speed in the plot at the top
    # right corner
    avg_vp = np.nanmean(df_dsco.vp)

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
    (_,) = axs4.plot(
        df_dsco.index.values, df_dsco.flux.values, "w-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.flux.isnull().all():
        axs4.set_ylim([0, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_dsco.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_dsco.flux), 3.3]),
        )

    # In a textbox, add the average value of the flux in the plot at the top
    # right corner
    avg_flux = np.nanmean(df_dsco.flux)

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

    #  Add a horizontal line at y=2.5 and label it
    axs4.axhline(2.5, color="r", lw=2, ls="--")
    # Add a text right above the horizontal line
    axs4.text(
        0.05,
        0.5,
        r"$\Phi_{\rm{th}} = 2.5$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs4.transAxes,
        fontsize=24,
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    axs4.set_yscale("linear")
    axs4.set_ylabel(
        r"Flux $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize, color="w"
    )

    # Add the dynamic pressure plot
    axs5 = fig.add_subplot(gs[4:5, 0], sharex=axs1)
    axs5.plot(
        df_dsco.index.values,
        df_dsco.p_dyn.values,
        "m-",
        lw=lw,
        ms=ms,
        label=r"Dynamic Pressure",
    )
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.p_dyn.isnull().all():
        axs5.set_ylim([0, 1])
    else:
        axs5.set_ylim(0.9 * np.nanmin(df_dsco.p_dyn), 1.1 * np.nanmax(df_dsco.p_dyn))

    # In a textbox, add the average value of the dynamic pressure in the plot at the top
    # right corner
    avg_p_dyn = np.nanmean(df_dsco.p_dyn)

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
    axs5.set_ylabel(
        r"Dynamic Pressure [nPa]", fontsize=ylabelsize, color="w", labelpad=20
    )

    # Cusp latitude plot
    axs6 = fig.add_subplot(gs[5:7, 0], sharex=axs1)

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

    axs6.plot(
        df_dsco.index.values,
        df_dsco.r_shue.values,
        "w-",
        lw=lw,
        ms=ms,
        label=r"Shue",
    )

    axs6.plot(
        df_dsco.index.values,
        df_dsco.r_yang.values,
        "b-",
        lw=lw,
        ms=ms,
        label=r"Yang",
    )

    axs6.plot(
        df_dsco.index.values,
        df_dsco.r_lin.values,
        "g-",
        lw=lw,
        ms=ms,
        label=r"Lin",
    )
    axs6.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_dsco.r_shue.isnull().all()
        and df_dsco.r_yang.isnull().all()
        and df_dsco.r_lin.isnull().all()
    ):
        axs6.set_ylim([-1, 1])
    else:
        axs6.set_ylim(0.97 * min_rmp, 1.03 * max_rmp)

    # In a textbox, add the average value of the magnetopause distance in the plot at the top
    # right corner
    avg_rmp_shue = np.nanmean(df_dsco.r_shue)
    avg_rmp_yang = np.nanmean(df_dsco.r_yang)
    avg_rmp_lin = np.nanmean(df_dsco.r_lin)

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
            -0.05,
            -0.05 + 0.10 * (i + 1),
            txt,
            ha="right",
            va="center",
            transform=axs6.transAxes,
            fontsize=24,
            color=y_label_colors[i],
        )
    axs6.set_ylabel(r"Magnetopause Distance [$R_{\oplus}$]", fontsize=ylabelsize)

    axs6.set_xlabel(
        f"Time on {df_dsco.index.date[0]} (UTC) [HH:MM]", fontsize=xlabelsize
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
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    fig_name = "sw_dscovr_parameters_2hr.png"
    fig_name = folder_name / fig_name
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300)
    plt.close("all")
    print(
        "Figure saved for DSCOVR at (UTC):"
        + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # print(f'It took {round(time.time() - start, 3)} seconds')
    return None


# s.enter(0, 1, plot_figures_dsco, (s,))
# s.run()

if __name__ == "__main__":
    plot_figures_dsco()
