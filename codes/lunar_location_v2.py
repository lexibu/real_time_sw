import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from astropy.coordinates import get_moon, GCRS, ITRS
from astropy.time import Time
import astropy.units as u


def calculate_gsm_coordinates(time):
    # Get Moon's position in GCRS (Geocentric Celestial Reference System)
    moon_gcrs = get_moon(time)

    # Convert to ITRS (International Terrestrial Reference System)
    moon_itrs = moon_gcrs.transform_to(ITRS(obstime=time))

    # GSM conversion (this is simplified, a more accurate method requires external packages or complex calculations)
    # Assuming a basic transformation for the GSM system:
    # For accurate GSM conversion, you'd need the solar and magnetic field data and perform a detailed transformation.

    moon_position_gsm = np.array(
        [moon_itrs.x.value, moon_itrs.y.value, moon_itrs.z.value]
    )

    return moon_position_gsm


def plot_positions(earth_pos, moon_pos, time_str):
    fig, ax = plt.subplots()

    # Earth at the origin
    ax.scatter(earth_pos[0], earth_pos[1], color="blue", label="Earth", s=300)

    # Moon position
    ax.scatter(moon_pos[0], moon_pos[1], color="gray", label="Moon", s=100)

    ax.set_xlabel("GSM X (km)")
    ax.set_ylabel("GSM Y (km)")
    ax.set_title(f"Positions in GSM Coordinates at {time_str}")

    ax.legend()
    ax.grid(True)

    plt.savefig(f"gsm_positions_{time_str}.png")
    plt.close()


# Main loop to update every hour
while True:
    current_time = datetime.utcnow()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")

    # Convert to Astropy time
    astropy_time = Time(current_time)

    # Calculate positions in GSM
    earth_position_gsm = np.array([0, 0, 0])  # Earth is at the origin
    moon_position_gsm = calculate_gsm_coordinates(astropy_time)

    # Plot the positions
    plot_positions(earth_position_gsm, moon_position_gsm, time_str)

    # Wait for one hour
    time.sleep(10)
