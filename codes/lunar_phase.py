import matplotlib.pyplot as plt
import ephem
import numpy as np
import time


def plot_earth_moon():
    # Set up observer
    observer = ephem.Observer()
    observer.lat = "0"  # Equator
    observer.lon = "0"  # Prime Meridian

    # Get Earth and Moon
    moon = ephem.Moon(observer)
    sun = ephem.Sun(observer)

    # Calculate positions
    moon.compute(observer)
    sun.compute(observer)

    # Calculate phase of the Moon
    moon_phase = moon.phase  # Percentage illumination

    # Convert to Cartesian coordinates for plotting
    moon_distance = moon.earth_distance * ephem.meters_per_au  # distance in meters
    moon_x = moon_distance * np.cos(moon.ra) * np.cos(moon.dec)
    moon_y = moon_distance * np.sin(moon.ra) * np.cos(moon.dec)

    # Scale all distance by earth radius (for plotting), where the earth radius is 6400 km
    earth_radius = 6.371e6
    moon_x /= earth_radius
    moon_y /= earth_radius
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Earth
    earth_radius_scaled = 1  # in meters
    earth_circle = plt.Circle((0, 0), earth_radius_scaled, color="blue", label="Earth")
    ax.add_artist(earth_circle)

    # Plot Moon
    ax.plot(moon_x, moon_y, "o", label=f"Moon (Phase: {moon_phase:.1f}%)")

    # Set plot limits and labels
    limit = 4e8 / earth_radius  # Roughly the distance from Earth to Moon
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Distance (meters)")
    ax.set_title("Real-time Earth and Moon Position")
    ax.legend()

    # Save the image
    plt.savefig("earth_moon_plot.png")
    plt.close()


if __name__ == "__main__":
    plot_earth_moon()
    # time.sleep(3600)  # Wait for 1 hour before updating again
