#!/home/vetinari/.cache/pypoetry/virtualenvs/codes-fO0b3aYA-py3.10/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from pathlib import Path
import os

font = {
    "family": "sans-serif",
    "sans-serif": ["Helvetica"],
    "weight": "normal",
    "size": 20,
}
plt.rc("font", **font)
plt.rc("text", usetex=False)
mpl.rcParams["pgf.texsystem"] = "pdflatex"

re = 6371.0  # radius of earth in km


# ========================================
# Analytic surface functions for bow shock
def jelinek_plane_bs(pdyn, r0=15.02, l_turning=1.17, e=6.55):
    """
    Bow shock model from Jelinek et al 2012. Assumes GSE Z=0.

    Parameters:
        pdyn (float): Solar wind dynamic pressure (nPa)
        r0 (float): Bow shock average standoff distance tuning parameter (RE)
        l (float): Lambda tuning parameter
        e (float): Epsilon tuning parameter
    """
    n = 200
    result = np.zeros([2, n])
    result[1, :] = np.linspace(-80, 80, num=n)
    for i in range(n):
        result[0, :] = r0 * (pdyn ** (-1 / e)) - (result[1, :] ** 2) * (
            l_turning**2
        ) / (4 * r0 * (pdyn ** (-1 / e)))

    return result


# Analytic surface functions for bow shock
def jelinek_plane_mp(pdyn, r0=12.82, l_turning=1.54, e=5.26):
    """
    Bow shock model from Jelinek et al 2012. Assumes GSE Z=0.

    Parameters:
        pdyn (float): Solar wind dynamic pressure (nPa)
        r0 (float): Bow shock average standoff distance tuning parameter (RE)
        l_turning (float): Lambda tuning parameter
        e (float): Epsilon tuning parameter
    """
    n = 200
    result = np.zeros([2, n])
    result[1, :] = np.linspace(-80, 80, num=n)
    for i in range(n):
        result[0, :] = r0 * (pdyn ** (-1 / e)) - (result[1, :] ** 2) * (
            l_turning**2
        ) / (4 * r0 * (pdyn ** (-1 / e)))

    return result


# ========================================

user_name = os.getlogin()
initial_dir = f"/home/{user_name}/Desktop/git/real_time_sw/data/lunar_data/"
dir = os.path.expanduser(initial_dir)

# df = pd.read_csv(dir+'LEXI_Lunar_Pos.txt',header=0,sep=" ")
df = pd.read_csv(dir + "LEXI_Lunar_Pos.txt", header=3, delim_whitespace=True)

xgse = df["X"].to_numpy()
ygse = df["Y"].to_numpy()

df["moon_time"] = df["Date"].str.cat(df["Time"], sep=" ")
datetime_obj = pd.to_datetime(df["moon_time"], format="%y/%m/%d %H:%M:%S", utc=True)

# get current time from computer in UTC
now = datetime.datetime.now(datetime.timezone.utc)

# find index in the array of lunar coordinates closest to current time
closest_idx = (datetime_obj - now).abs().idxmin()
# Get the numebr of 2 minutes interval in 2 weeks
dindx = int(14.7 * 24 * 60 / 2)

# trim to plot +/- 2 weeks from current time
xplt = xgse[closest_idx - dindx : closest_idx + dindx]
yplt = ygse[closest_idx - dindx : closest_idx + dindx]

# Set constants for plotting
pdyn = 2.5  # nPa
mp_val = jelinek_plane_mp(pdyn)
bs_val = jelinek_plane_bs(pdyn)

fig = plt.figure(figsize=(10, 10))
# Set the plot theme to dark
plt.style.use("dark_background")

plt.axes().set_aspect("equal")

plt.plot(yplt / re, xplt / re, linewidth=1.5, label="Moon Trajectory", color="orange")
plt.xlabel(r"Y (GSE) [$R_E$]")
plt.ylabel(r"X (GSE) [$R_E$]")

plt.plot(mp_val[1, :], mp_val[0, :], c="w", linewidth=1)
plt.plot(bs_val[1, :], bs_val[0, :], c="w", linewidth=1)

theta = np.arctan2(ygse[closest_idx], xgse[closest_idx]) * 180 / np.pi
if theta < 0:
    theta = theta + 360.0
plt.text(60, 70, "Angle Moon to E-S line: " + "{:.2f}".format(theta) + " deg")

plt.ylim((-70, 80))
plt.xlim((70, -70))

# plot Earth
circle1 = plt.Circle((0, 0), 1.0, color="g", fill=True, linewidth=4)
fig = plt.gcf()
fig.gca().add_artist(circle1)

plt.scatter(ygse[closest_idx] / re, xgse[closest_idx] / re, s=130, color="red", alpha=1)

plt.title(f"Moon Position at {now.strftime('%Y-%m-%d %H:%M:%S')}")

# Properly define the folder and figure name
folder_name = "~/Dropbox/rt_sw/"
folder_name = Path(folder_name).expanduser()
Path(folder_name).mkdir(parents=True, exist_ok=True)

fig_name = folder_name / "moon_pos_temp.png"

plt.savefig(
    fig_name,
    bbox_inches="tight",
    dpi=300,
    transparent=False,
    pad_inches=0.05,
    facecolor="black",
)
