import matplotlib.pyplot as plt
import numpy as np
import ephem
import time


plt.style.use("dark_background")


def plot_earth_moon():
    # Set up observer
    observer = ephem.Observer()
    observer.lat = "0"  # Equator
    observer.lon = "0"  # Prime Meridian

    # Get Moon and Sun positions
    moon = ephem.Moon(observer)
    sun = ephem.Sun(observer)

    # Compute positions
    moon.compute(observer)
    sun.compute(observer)

    # Calculate Moon's phase
    moon_phase = moon.phase  # Percentage illumination

    # Convert to Cartesian coordinates for plotting
    moon_distance = moon.earth_distance * ephem.meters_per_au  # distance in meters
    moon_x = moon_distance * np.cos(moon.ra) * np.cos(moon.dec)
    moon_y = moon_distance * np.sin(moon.ra) * np.cos(moon.dec)
    moon_z = moon_distance * np.sin(moon.dec)

    # Scale all distance by earth radius (for plotting), where the earth radius is 6400 km
    earth_radius = 6.371e6
    moon_x /= earth_radius
    moon_y /= earth_radius

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Earth
    # Plot Earth
    earth_radius_scaled = 1  # in meters
    earth_circle = plt.Circle((0, 0), earth_radius_scaled, color="blue", label="Earth")
    ax.add_artist(earth_circle)

    # Moon's phase visualization
    moon_radius = 10  # Moon's radius in meters
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = moon_radius * np.cos(theta)
    y_circle = moon_radius * np.sin(theta)

    # Shade the Moon according to its phase
    moon_illumination = (moon_phase / 100.0) * 2 * np.pi
    x_shadow = moon_radius * np.cos(np.linspace(moon_illumination, 2 * np.pi, 100))
    y_shadow = moon_radius * np.sin(np.linspace(moon_illumination, 2 * np.pi, 100))

    ax.fill(x_circle, y_circle, color="w", alpha=0.5)  # Moon
    ax.fill(x_shadow, y_shadow, color="k")  # Shadow part of the Moon

    # Plot Moon's position (for reference)
    ax.plot(0, moon_radius, "o", color="gray", label=f"Moon (Phase: {moon_phase:.1f}%)")

    # Set plot limits and labels
    limit = 20  # Adjusted for visibility
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Distance (meters)")
    ax.set_title("Real-time Earth and Moon Position")
    ax.legend()

    # Save the image
    plt.savefig("earth_moon_plot.png")
    plt.close()
    # plt.show()


if __name__ == "__main__":
    # while True:
    plot_earth_moon()
    #     time.sleep(3600)  # Wait for 1 hour before updating again
