# Analytic surface functions for bow shock
# -*- coding: utf-8 -*-
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import warnings
import glob
import imageio as iio
from matplotlib.collections import PolyCollection


warnings.simplefilter(action="ignore", category=FutureWarning)


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


def get_lunar_position(now=datetime.datetime.now(datetime.timezone.utc), pdyn=2.5):

    # Properly define the folder and figure name
    folder_name = "../figures/rt_sw/movie_frames/"
    # Check if folder exists, if not create it
    folder_name = Path(folder_name).expanduser()
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Add the time to the figure name
    fig_name = folder_name / f"{now.strftime('%Y%m%d_%H%M')}.png"
    # If file already exists, return
    if os.path.exists(fig_name):
        print(f"File {fig_name} already exists. Skipping...")
        return
    else:
        print(f"Generating plot for {now.strftime('%Y-%m-%d %H:%M:%S')}")
    # Define earth radius in meters
    R_earth = 6.371e6
    R_earth_km = R_earth / 1e3

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
    # now = datetime.datetime.now(datetime.timezone.utc)

    # find index in the array of lunar coordinates closest to current time
    closest_idx = (datetime_obj - now).abs().idxmin()

    # Get the closest index for next day
    closest_idx_day1 = (
        (datetime_obj - (now + datetime.timedelta(days=7))).abs().idxmin()
    )
    closest_idx_day2 = (
        (datetime_obj - (now + datetime.timedelta(days=14))).abs().idxmin()
    )
    closest_idx_day3 = (
        (datetime_obj - (now + datetime.timedelta(days=21))).abs().idxmin()
    )
    # Get the numebr of 2 minutes interval in 2 weeks
    dindx = int(14.7 * 24 * 60 / 2)

    # trim to plot +/- 2 weeks from current time
    xplt = xgse[closest_idx - dindx : closest_idx + dindx]
    yplt = ygse[closest_idx - dindx : closest_idx + dindx]

    moon_pos = np.array([xgse[closest_idx], ygse[closest_idx], zgse[closest_idx]])

    # Set constants for plotting
    # pdyn = 2.5  # nPa
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
        ls="--",
        label="Moon Trajectory",
        color="gray",
        alpha=0.5,
    )
    plt.xlabel(r"Y (GSE) [$R_E$]")
    plt.ylabel(r"X (GSE) [$R_E$]")

    plt.plot(mp_val[1, :], mp_val[0, :], c="c", linewidth=1)
    # Add a text along the line
    plt.text(
        -42,
        -50,
        "Magnetopause",
        color="c",
        fontsize=20,
        ha="right",
        va="center",
        rotation=-75,
    )
    plt.plot(bs_val[1, :], bs_val[0, :], c="w", linewidth=1)
    # Add a text along the line
    plt.text(
        58.7,
        -65,
        "Bow Shock",
        color="w",
        fontsize=20,
        ha="left",
        va="bottom",
        rotation=69,
    )

    # x_combined = np.concatenate((bs_val[1, :], mp_val[1, ::-1]))
    # y_combined = np.concatenate((bs_val[0, :], mp_val[0, ::-1]))
    # gradient = np.gradient(y_combined, x_combined)
    # colors = plt.cm.plasma(gradient / gradient.max())

    # # Create a PolyCollection with gradient colors
    # verts = np.array([x_combined, y_combined]).T
    # polygon = PolyCollection(
    #     [verts], array=gradient, cmap="viridis", edgecolor="none", alpha=0.6
    # )
    # plt.gca().add_collection(polygon)
    # plt.autoscale()
    # plt.colorbar(polygon, ax=plt.gca(), orientation="vertical", label="Gradient")

    # Fill the area between the bow shock and magnetopause
    plt.fill(
        np.concatenate((bs_val[1, :], mp_val[1, ::-1])),
        np.concatenate((bs_val[0, :], mp_val[0, ::-1])),
        color="coral",
        alpha=0.2,
    )
    theta = np.arctan2(ygse[closest_idx], xgse[closest_idx]) * 180 / np.pi
    if theta < 0:
        theta = theta + 360.0
    # Add this angle to the plot
    plt.text(
        0.02,
        0.98,
        r"$\theta_{moon, Earth-sun-line} = %.1f^\circ$" % theta,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=20,
        bbox=dict(facecolor="gray", alpha=0.5),
    )
    # Add pdyn value to the plot
    plt.text(
        0.98,
        0.98,
        r"$P_{dyn} = %.1f$ nPa" % pdyn,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=20,
        bbox=dict(facecolor="gray", alpha=0.5),
    )

    # Find the maximum value of mp_val at y=0
    mp_val_max = np.nanmax(mp_val[0, :])
    bs_val_max = np.nanmax(bs_val[0, :])
    # Define positions for the arrows
    arrow_start_mp = (0, mp_val_max)
    arrow_start_bs = (0, bs_val_max)

    # Add arrows and labels
    plt.annotate(
        r"%.1f$R_E$" % mp_val_max,
        xy=arrow_start_mp,
        xytext=(
            arrow_start_mp[1] - 15.15,
            arrow_start_mp[0] - 15.15,
        ),
        arrowprops=dict(
            facecolor="aqua",
            shrink=0.01,
            width=1,
            headwidth=7,
            alpha=0.2,
            zorder=1,
            edgecolor="aqua",
        ),
        color="aqua",
        fontsize=20,
        bbox=dict(facecolor="gray", alpha=0.0),
        ha="center",
        va="center",
        alpha=0.2,
    )

    plt.annotate(
        r"%.1f$R_E$" % bs_val_max,
        xy=arrow_start_bs,
        xytext=(
            40,
            arrow_start_bs[0] + 15.15,
        ),
        arrowprops=dict(
            facecolor="w",
            shrink=0.05,
            width=1,
            headwidth=7,
            alpha=0.5,
            zorder=1,
            edgecolor="w",
        ),
        color="w",
        fontsize=20,
        bbox=dict(facecolor="gray", alpha=0.0),
        ha="center",
        va="center",
        alpha=0.6,
    )

    # arrow = patches.FancyArrowPatch(
    #     arrow_start_bs,
    #     (40, bs_val_max + 4),
    #     connectionstyle="arc3,rad=0.4",  # Curvature (rad > 0 makes a counter-clockwise curve)
    #     color="w",
    #     linewidth=1,
    #     arrowstyle="->",
    #     mutation_scale=15,
    #     alpha=0.6,
    # )
    # plt.gca().add_patch(arrow)

    plt.ylim((-70, 80))
    plt.xlim((70, -70))

    # plot Earth
    # circle1 = plt.Circle((0, 0), 1.0, color="g", fill=True, linewidth=4)
    # fig = plt.gcf()
    # fig.gca().add_artist(circle1)
    # Add unicode for earth
    plt.text(
        0,
        0,
        "\u25CF",
        fontsize=30,
        ha="center",
        va="center",
        color="g",
        zorder=1,
    )

    # plot moon
    # plt.scatter(
    #     ygse[closest_idx] / R_earth_km,
    #     xgse[closest_idx] / R_earth_km,
    #     s=130,
    #     color="red",
    #     alpha=0.8,
    #     zorder=10,
    # )
    plt.text(
        ygse[closest_idx] / R_earth_km,
        xgse[closest_idx] / R_earth_km,
        "\u263D",
        fontsize=30,
        ha="center",
        va="center",
        color="white",
    )

    plt.text(
        ygse[closest_idx_day1] / R_earth_km,
        xgse[closest_idx_day1] / R_earth_km,
        "\u263D",
        fontsize=30,
        ha="center",
        va="center",
        color="white",
        alpha=0.2,
    )
    plt.text(
        ygse[closest_idx_day2] / R_earth_km,
        xgse[closest_idx_day2] / R_earth_km,
        "\u263D",
        fontsize=30,
        ha="center",
        va="center",
        color="white",
        alpha=0.2,
    )
    plt.text(
        ygse[closest_idx_day3] / R_earth_km,
        xgse[closest_idx_day3] / R_earth_km,
        "\u263D",
        fontsize=30,
        ha="center",
        va="center",
        color="white",
        alpha=0.2,
    )

    plt.title(f"Moon Position at {now.strftime('%Y-%m-%d %H:%M')}")

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

    plt.fill(cone_y, cone_x, color="bisque", alpha=0.1)
    # Add a text with color bisque and alpha=0.1
    plt.text(
        0.02,
        0.9,
        "Projected Look Direction",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        color="bisque",
        fontsize=20,
    )

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

    plt.savefig(
        fig_name,
        bbox_inches="tight",
        dpi=300,
        transparent=False,
        pad_inches=0.05,
        facecolor="black",
    )
    plt.close()


def gif_maker(
    file_list, vid_name, mode="I", skip_rate=10, vid_type="mp4", duration=0.05, fps=25
):
    """
    Make a gif from a list of images.

    Parameters
    ----------
    file_list : list
        List of image files.
    vid_name : str
        Name of the gif file.
    mode : str, optional
        Mode of the gif. The default is "I".
    skip_rate : int, optional
        Skip rate of the gif. The default is 10.
    vid_type : str, optional
        Type of the video. The default is "mp4".
    duration : float, optional
        Duration for which each image is displayed in gif. The default is 0.05.
    fps : int, optional
        Frames per second for mp4 video. The default is 25.

    Raises
    ------
    ValueError
        If the skip_rate is not an integer.
    ValueError
        If the duration is not a float.
    ValueError
        If the file_list is empty.
    ValueError
        If vid_name is empty.

    Returns
    -------
    None.
    """
    # From the video name, get the directory and make sure that it exists using pathlib
    Path(vid_name).parent.mkdir(parents=True, exist_ok=True)
    if file_list is None:
        raise ValueError("file_list is None")
    if vid_name is None:
        raise ValueError("vid_name is None. Please provide the name of the gif/video")
    if len(file_list) == 0:
        raise ValueError("file_list is empty")
    # if len(file_list) >= 1501:
    #     # Check if the skip_rate is an integer
    #     if skip_rate != int(skip_rate):
    #         raise ValueError("skip_rate must be an integer")
    #     file_list = file_list[-1500::skip_rate]
    if vid_type == "gif":
        if duration != float(duration):
            raise ValueError("duration must be a float")
    if vid_type == "mp4":
        if fps != int(fps):
            raise ValueError("Frame rate (fps) must be an integer")

    count = 0
    if vid_type == "gif":
        with iio.get_writer(vid_name, mode=mode, duration=duration) as writer:
            for filename in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                try:
                    img = iio.imread(filename)
                    writer.append_data(img)
                except Exception as e:
                    print(e)
                    pass
    elif vid_type == "mp4":
        with iio.get_writer(vid_name, mode=mode, fps=fps) as writer:
            for filename in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                try:
                    img = iio.imread(filename)
                    writer.append_data(img)
                except Exception as e:
                    print(e)
                    pass
    writer.close()

    # Print that the video is created along with the time of creation in UTC
    print(
        f"Video created at (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    )

    # Copy the file to a specific location
    # os.system(f"cp {vid_name} ~/Dropbox/rt_sw/")


# Generate the plots
# Set the time interval for the movie
time_interval = 15  # minutes
total_duration = 3  # days
plotting_start_time = datetime.datetime(
    2025, 9, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
)
# Loop through the time interval and generate the plots
for i in range(int(total_duration * 24 * 60 / time_interval)):
    now = plotting_start_time + datetime.timedelta(minutes=i * time_interval)
    # Print the progress
    # print(f"Generating plot for {now.strftime('%Y-%m-%d %H:%M:%S')}")
    get_lunar_position(now)
    # Add a progress bar to show the progress of the image generation
    print(f"Generating plot - {i+1} of {int(total_duration * 24 * 60 / time_interval)}")


file_list = sorted(glob.glob("../figures/rt_sw/movie_frames/*.png"))
start_time = "2025-03-02 00:00"
end_time = "2025-03-08 00:00"
start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M")
end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M")
new_file_list = []
for file in file_list:
    file_timestamp_str = file.split("/")[-1].split(".")[0]
    file_timestamp = datetime.datetime.strptime(file_timestamp_str, "%Y%m%d_%H%M")
    if start_time <= file_timestamp <= end_time:
        new_file_list.append(file)


gif_maker(
    new_file_list,
    f"../figures/rt_sw/movie/lunar_position_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.mp4",
    mode="I",
    skip_rate=1,
    vid_type="mp4",
    duration=0.05,
    fps=25,
)
