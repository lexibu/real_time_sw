import astropy.coordinates, sunpy.coordinates, astropy.units as u, datetime
import numpy as np

my_gse_coord = astropy.coordinates.SkyCoord(
    x=10 * u.R_earth,
    y=0 * u.R_earth,
    z=0 * u.R_earth,
    representation_type="cartesian",
    frame=sunpy.coordinates.GeocentricSolarEcliptic(
        obstime=datetime.datetime(2025, 3, 3)
    ),
)

my_icrs_coord = my_gse_coord.transform_to("icrs")

my_gse_coord_2 = astropy.coordinates.SkyCoord(
    x=10 * u.R_earth,
    y=0 * u.R_earth,
    z=0 * u.R_earth,
    representation_type="cartesian",
    frame=sunpy.coordinates.GeocentricSolarEcliptic(
        obstime=datetime.datetime(2026, 3, 3)
    ),
)

my_icrs_coord_2 = my_gse_coord_2.transform_to("icrs")

print(my_gse_coord)
print(my_icrs_coord)

print(my_gse_coord_2)
print(my_icrs_coord_2)


# # Convert RA and DEC to degrees
ra = my_icrs_coord.ra.value
dec = my_icrs_coord.dec.value

# print(f"RA: {ra}, DEC: {dec}")


my_icrs_coord = astropy.coordinates.SkyCoord(
    ra=200.291689115616 * u.deg,
    dec=-11.1626773128907 * u.deg,
    distance=1 * u.m,
    frame="icrs",
)

new_coord = my_gse_coord.transform_to(
    sunpy.coordinates.GeocentricSolarEcliptic(
        obstime=datetime.datetime(
            2026,
            5,
            4,
            18,
            0,
        )
    )
)

ra_new = new_coord.transform_to("icrs").ra.value
dec_new = new_coord.transform_to("icrs").dec.value

# print(f"RA: {ra_new}, DEC: {dec_new}")
