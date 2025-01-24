import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.time import Time
import pandas as pd
import os


# Function to convert RA/Dec (J2000) to a unit vector in GSE coordinates
def ra_dec_to_gse(ra, dec):
    skycoord = SkyCoord(ra=ra, dec=dec, unit="deg", frame=FK5(equinox=Time("J2000")))
    return skycoord.cartesian.xyz.value  # Unit vector in the same coordinate system


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


start_time = "2025-03-03 00:00:00"
# Convert the start_time string to a datetime object
start_time = pd.to_datetime(start_time, format="%Y-%m-%d %H:%M:%S", utc=True)
# Get the username of the system
user_name = os.getlogin()
initial_dir = f"/home/{user_name}/Desktop/git/real_time_sw/data/lunar_data/"
dir = os.path.expanduser(initial_dir)

# df = pd.read_csv(dir+'LEXI_Lunar_Pos.txt',header=0,sep=" ")
df = pd.read_csv(dir + "LEXI_Lunar_Pos.txt", header=3, sep="\s+")
# Combine the Date (DD/MM/YY) and Time (HH:MM:SS) columns and connvert them to datetime
df["moon_time"] = df["Date"].str.cat(df["Time"], sep=" ")
# Convert the combined string to datetime
df["moon_time"] = pd.to_datetime(df["moon_time"], format="%y/%m/%d %H:%M:%S")

# Convert the datetime to UTC
df["moon_time"] = df["moon_time"].dt.tz_localize("UTC")
# Convert the datetime to a pandas datetime index
df["moon_time"] = pd.to_datetime(df["moon_time"], utc=True)
# Drop the original Date and Time columns
df.drop(columns=["Date", "Time"], inplace=True)
# Set the moon_time column as the index
df.set_index("moon_time", inplace=True)

# Get the closest index in the DataFrame to the start_time
closest_index = abs((df.index - start_time).total_seconds()).argmin()
dindx = int(15 * 24 * 60 / 2)

# Get the corresponding row
closest_row = df.iloc[closest_index]
# Extract the GSE coordinates
x_moon = closest_row["X"]
y_moon = closest_row["Y"]
z_moon = closest_row["Z"]

# Inputs
moon_gse = np.array(
    [x_moon, y_moon, z_moon]
)  # Lunar position in GSE (replace x_moon, etc. with actual values)
earth_gse = np.array([0, 0, 0])  # Earth position in GSE

# Set constants for plotting
pdyn = 2.5  # nPa
mp_val = jelinek_plane_mp(pdyn)
bs_val = jelinek_plane_bs(pdyn)


# RA/Dec of the look direction
ra = 200.29  # Right Ascension in degrees (replace with actual value)
dec = -11.16  # Declination in degrees (replace with actual value)

# Convert RA/Dec to a GSE direction unit vector
look_direction_gse = ra_dec_to_gse(ra, dec)

# Earth Radius in GSE coordinates
r_e = 6.371e3  # km

# Scale everything by earth radius
moon_gse = moon_gse / r_e
earth_gse = earth_gse / r_e

# Create an arrow to represent the look direction in GSE coordinates extending from the Moon and
# spanning to the whole GSE sphere
arrow_length = 100  # Length of the arrow
arrow_end = moon_gse + look_direction_gse * arrow_length
# Plotting
# Set the plot style to dark background
plt.style.use("dark_background")
# Set up the 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d", frame_on=False, elev=90, azim=180)

# Plot the Earth
ax.scatter(earth_gse[0], earth_gse[1], earth_gse[2], color="g", label="Earth", s=100)
# Plot the Moon
ax.scatter(moon_gse[0], moon_gse[1], moon_gse[2], color="b", label="Moon", s=100)

# Plot the bs and mp
ax.plot(
    mp_val[0, :],
    mp_val[1, :],
    mp_val[0, :] * 0,
    linewidth=1,
    label="MP",
    color="w",
)
ax.plot(
    bs_val[0, :],
    bs_val[1, :],
    bs_val[0, :] * 0,
    linewidth=1,
    label="BS",
    color="w",
)

# Plot the lunar trajectory over the past 15 days and the next 15 days
plt.plot(
    df["X"][closest_index - dindx : closest_index + dindx].values / r_e,
    df["Y"][closest_index - dindx : closest_index + dindx].values / r_e,
    df["Z"][closest_index - dindx : closest_index + dindx].values / r_e,
    linewidth=1.5,
    label="Moon Trajectory",
    color="orange",
)

# Plot the look direction
ax.quiver(
    moon_gse[0],
    moon_gse[1],
    moon_gse[2],
    look_direction_gse[0],
    look_direction_gse[1],
    look_direction_gse[2],
    length=arrow_length,
    color="r",
    label="Look Direction",
)
# Set labels and title
ax.set_xlabel("X (GSE)")
ax.set_ylabel("Y (GSE)")
ax.set_zlabel("Z (GSE)")
ax.set_title("Look Direction in GSE Coordinates")
# Add a legend
# ax.legend()

# Flip the y-axis
ax.invert_yaxis()

# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Hide the z-axis
ax.zaxis.set_visible(False)
# Hide the z-label and z-ticks
ax.zaxis.label.set_visible(False)
ax.zaxis.set_ticks([])
# Hide the grid
ax.grid(False)
# Make the plot 2D by projecting the points onto the XY plane
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide the x-axis line
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide the y-axis line
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Hide the z-axis line

# Set the view angle to be z-down, with y-axis at the bottom and x-axis at the left
ax.view_init(elev=90, azim=180)

# Set the limits of the axes
ax.set_xlim([-80, 80])
ax.set_ylim([-80, 80])
ax.set_zlim([-80, 80])

# Show the plot
plt.savefig("look_direction_test.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
