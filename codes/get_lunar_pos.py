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

# Define earth radius in meters
R_earth = 6.371e6
R_earth_km = R_earth / 1e3


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


# Analytic surface functions for Magnetopause
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


user_name = os.getlogin()
initial_dir = f"/home/{user_name}/Desktop/git/real_time_sw/data/lunar_data/"
dir = os.path.expanduser(initial_dir)

# df = pd.read_csv(dir+'LEXI_Lunar_Pos.txt',header=0,sep=" ")
df = pd.read_csv(dir + "LEXI_Lunar_Pos.txt", header=3, delim_whitespace=True)

xgse = df["X"].to_numpy()
ygse = df["Y"].to_numpy()
zgse = df["Z"].to_numpy()

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
zplt = zgse[closest_idx - dindx : closest_idx + dindx]

moon_pos = np.array([xgse[closest_idx], ygse[closest_idx], zgse[closest_idx]])

# Set constants for plotting
pdyn = 2.5  # nPa
mp_val = jelinek_plane_mp(pdyn)
bs_val = jelinek_plane_bs(pdyn)


# Load the CSV files
mag_dir = f"/home/{user_name}/Desktop/git/real_time_sw/data/"
mag_dir = os.path.expanduser(mag_dir)
df_mag = pd.read_csv(mag_dir + "mag_ra_dec.csv")

# Convert 'epoch_utc' column in 'df_mag' to datetime and set the timezone to UTC
df_mag["epoch_utc"] = pd.to_datetime(df_mag["epoch_utc"])
df_mag["epoch_utc"] = df_mag["epoch_utc"].dt.tz_localize("UTC")

# Set 'epoch_utc' as the index for both df_magframes
df_mag = df_mag.set_index("epoch_utc")

# Get the closest index in the DataFrame to "now"
closest_idx_mag = abs((df_mag.index - now).total_seconds()).argmin()

# Get the x, y and z coordinate of lexi in GSE
x_lexi_gse = df_mag.iloc[closest_idx_mag]["e_to_l_x"]
y_lexi_gse = df_mag.iloc[closest_idx_mag]["e_to_l_y"]
z_lexi_gse = df_mag.iloc[closest_idx_mag]["e_to_l_z"]

lexi_pos = np.array([x_lexi_gse, y_lexi_gse, z_lexi_gse])

# Get the x, y and z coordinate of the magnetopause in GSE
x_mp_gse = df_mag.iloc[closest_idx_mag]["e_to_mag_track_x"]
y_mp_gse = df_mag.iloc[closest_idx_mag]["e_to_mag_track_y"]
z_mp_gse = df_mag.iloc[closest_idx_mag]["e_to_mag_track_z"]

mp_pos = np.array([x_mp_gse, y_mp_gse, z_mp_gse])

fig = plt.figure(figsize=(10, 10))
# Set the plot theme to dark
plt.style.use("dark_background")

plt.axes().set_aspect("equal")

plt.plot(
    yplt / R_earth_km,
    xplt / R_earth_km,
    linewidth=1.5,
    label="Moon Trajectory",
    color="orange",
)
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

plt.scatter(
    ygse[closest_idx] / R_earth_km,
    xgse[closest_idx] / R_earth_km,
    s=130,
    color="red",
    alpha=0.8,
    zorder=10,
)

plt.title(f"Moon Position at {now.strftime('%Y-%m-%d %H:%M:%S')}")


moon_pos = np.array([xgse[closest_idx], ygse[closest_idx]]) / R_earth_km
mp_pos = np.array([x_mp_gse, y_mp_gse]) / R_earth_km

# Calculate the unit vector from the Moon to the magnetopause
direction = mp_pos - moon_pos
direction_unit = direction / np.linalg.norm(direction)

# Angle of the cone (in degrees)
cone_angle = 4.5
cone_angle_rad = np.radians(cone_angle)

# Generate the two boundary vectors of the cone
rotation_matrix1 = np.array(
    [
        [np.cos(cone_angle_rad), -np.sin(cone_angle_rad)],
        [np.sin(cone_angle_rad), np.cos(cone_angle_rad)],
    ]
)
rotation_matrix2 = np.array(
    [
        [np.cos(-cone_angle_rad), -np.sin(-cone_angle_rad)],
        [np.sin(-cone_angle_rad), np.cos(-cone_angle_rad)],
    ]
)

boundary_vector1 = np.dot(rotation_matrix1, direction_unit)
boundary_vector2 = np.dot(rotation_matrix2, direction_unit)

# Extend the boundaries to the magnetopause
boundary_line1 = moon_pos + boundary_vector1 * 200
boundary_line2 = moon_pos + boundary_vector2 * 200

# Plotting
# plt.figure(figsize=(8, 8))
# # plt.scatter(*earth_pos, color="blue", label="Earth", s=100)
# plt.scatter(*moon_pos, color="gray", label="Moon", s=100)
# plt.scatter(*mp_pos, color="red", label="Magnetopause", s=100)

# Fill the cone region
cone_x = [moon_pos[0], boundary_line1[0], boundary_line2[0]]
cone_y = [moon_pos[1], boundary_line1[1], boundary_line2[1]]
plt.fill(cone_y, cone_x, color="c", alpha=0.1)

# Plot the cone boundaries
plt.plot(
    [moon_pos[1], boundary_line1[1]],
    [moon_pos[0], boundary_line1[0]],
    "w--",
    lw=0.6,
    alpha=0.5,
)
plt.plot(
    [moon_pos[1], boundary_line2[1]],
    [moon_pos[0], boundary_line2[0]],
    "w--",
    lw=0.6,
    alpha=0.5,
)

# Connect the Moon and magnetopause for visualization
plt.plot(
    [moon_pos[1], mp_pos[1]],
    [moon_pos[0], mp_pos[0]],
    "g-",
    lw=0.1,
    alpha=0.5,
)
# Properly define the folder and figure name
folder_name = "~/Dropbox/rt_sw/"
folder_name = Path(folder_name).expanduser()
Path(folder_name).mkdir(parents=True, exist_ok=True)

fig_name = folder_name / "moon_pos.png"

plt.savefig(
    fig_name,
    bbox_inches="tight",
    dpi=300,
    transparent=False,
    pad_inches=0.05,
    facecolor="black",
)
