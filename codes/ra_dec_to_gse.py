import astropy.coordinates as ac
import astropy.units as u
import datetime
import numpy as np
import sunpy.coordinates as sc
import matplotlib.pyplot as plt

distance_lenth = np.logspace(10, 20, 100)

# Make a plot of the unit vector in the look direction
plt.figure(figsize=(8, 8))
plt.title("Unit Vector in the Look Direction (GSE)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
# Get a randoom value between 0 and 360
# theta =
ra = np.random.uniform(0, 360) * u.deg  # Right Ascension in degrees
dec = np.random.uniform(-90, 90) * u.deg  # Declination in degrees
for i, distance in enumerate(distance_lenth):

    # Given RA, Dec, and Distance

    distance = distance * u.R_earth  # Distance in Earth radii

    # Convert from spherical to Cartesian coordinates in ICRS
    coord_icrs = ac.SkyCoord(ra=ra, dec=dec, distance=distance, frame="icrs")

    # Convert ICRS to GSE at the specified time
    time = datetime.datetime(2025, 3, 3, 0, 0, 0)  # Time for the transformation
    gse_coord = coord_icrs.transform_to(sc.GeocentricSolarEcliptic(obstime=time))

    # Extract the x, y, z coordinates in GSE
    x_gse = gse_coord.cartesian.x
    y_gse = gse_coord.cartesian.y
    z_gse = gse_coord.cartesian.z

    # Print the GSE coordinates (in Earth radii)
    # print(f"GSE Coordinates (x, y, z) at {time}:")
    # print(f"x = {x_gse:.4f} Earth radii")
    # print(f"y = {y_gse:.4f} Earth radii")
    # print(f"z = {z_gse:.4f} Earth radii")

    # Get the unit vector in the look direction
    look_direction_gse = np.array([x_gse.value, y_gse.value, z_gse.value])
    look_direction_gse /= np.linalg.norm(look_direction_gse)

    # Print the unit vector in the look direction
    # print("\nUnit vector in the look direction (GSE):")
    # Plot the each component of the unit vector in three different colors
    plt.plot(distance, abs(look_direction_gse[0]), "r.")
    plt.plot(distance, abs(look_direction_gse[1]), "g.")
    plt.plot(distance, abs(look_direction_gse[2]), "b.")


plt.xscale("log")
plt.yscale("log")
plt.savefig("unit_vector_look_direction.png")
